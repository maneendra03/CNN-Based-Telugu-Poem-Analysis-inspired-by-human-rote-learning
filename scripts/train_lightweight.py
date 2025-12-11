#!/usr/bin/env python
"""
Lightweight Training Script for MacBook M2
Optimized for 8GB RAM with MPS acceleration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.pretrained_backbone import GPT2PoemGenerator
from src.preprocessing.tokenizer import PoemTokenizer
from src.preprocessing.text_cleaner import TextCleaner


class SimplePoemDataset(Dataset):
    """Lightweight dataset for memory efficiency."""
    
    def __init__(self, poems, tokenizer, max_length=128):
        self.poems = poems
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cleaner = TextCleaner()
    
    def __len__(self):
        return len(self.poems)
    
    def __getitem__(self, idx):
        poem = self.poems[idx]
        text = poem.get('text', '') if isinstance(poem, dict) else poem
        
        # Clean and truncate
        text = self.cleaner.clean(text)[:500]  # Limit length for memory
        
        # Encode
        encoding = self.tokenizer.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Labels are shifted input_ids
        labels = input_ids.clone()
        labels[labels == self.tokenizer.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def train_lightweight():
    """Train with optimized settings for M2 Mac 8GB."""
    
    print("=" * 60)
    print("🎯 Lightweight Training for MacBook M2 Pro (8GB)")
    print("=" * 60)
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS (Metal) acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Settings optimized for 8GB RAM - FULL TRAINING
    config = {
        'batch_size': 4,          # Small batch for memory
        'max_length': 128,        # Shorter sequences
        'epochs': 5,              # More epochs for better accuracy
        'learning_rate': 3e-5,    # Lower LR for stability
        'warmup_steps': 200,
        'log_every': 100,
        'save_every': 2000,       # Save less frequently to save disk
        'max_poems': 12000,       # USE FULL DATASET
    }
    
    print(f"\nConfig: {config}")
    
    # Load data
    print("\n📚 Loading data...")
    data_path = Path("data/processed/train.json")
    
    if not data_path.exists():
        print("Error: Training data not found. Run dataset processing first.")
        return
    
    with open(data_path) as f:
        all_poems = json.load(f)
    
    # Use subset for faster training
    poems = all_poems[:config['max_poems']]
    print(f"Using {len(poems):,} poems for training")
    
    # Load model
    print("\n🧠 Loading GPT-2...")
    model = GPT2PoemGenerator(model_name='gpt2', freeze_backbone=True)
    
    # Unfreeze the language model head for training
    for param in model.gpt2.lm_head.parameters():
        param.requires_grad = True
    
    # Also unfreeze last transformer layer for better fine-tuning
    for param in model.gpt2.transformer.h[-1].parameters():
        param.requires_grad = True
    
    model.to(device)
    
    # Use model's tokenizer
    tokenizer = type('Tokenizer', (), {'tokenizer': model.tokenizer})()
    
    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Create dataset
    dataset = SimplePoemDataset(poems, tokenizer, max_length=config['max_length'])
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Required for MPS
        pin_memory=False
    )
    
    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['learning_rate']
    )
    
    # Scheduler
    total_steps = len(dataloader) * config['epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop
    print("\n🚀 Starting training...")
    print(f"Total batches per epoch: {len(dataloader)}")
    
    model.train()
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model.gpt2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1
            
            # Log
            if (batch_idx + 1) % config['log_every'] == 0:
                avg_loss = epoch_loss / batch_count
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{config['epochs']} | "
                      f"Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
            
            # Save checkpoint
            if global_step % config['save_every'] == 0:
                checkpoint_path = Path("checkpoints")
                checkpoint_path.mkdir(exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'loss': loss.item()
                }, checkpoint_path / f"checkpoint_step_{global_step}.pt")
                print(f"  💾 Saved checkpoint at step {global_step}")
        
        # End of epoch
        avg_epoch_loss = epoch_loss / batch_count
        print(f"\n📊 Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f}")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"  ⭐ New best model saved!")
    
    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_loss': avg_epoch_loss
    }, "checkpoints/final_model.pt")
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: checkpoints/")
    
    # Test generation
    print("\n📝 Testing generation...")
    model.eval()
    
    test_prompt = "Roses are red,"
    generated = model.generate(
        prompt=test_prompt,
        max_length=50,
        temperature=0.8
    )
    print(f"\nPrompt: {test_prompt}")
    print(f"Generated:\n{generated}")


if __name__ == "__main__":
    train_lightweight()
