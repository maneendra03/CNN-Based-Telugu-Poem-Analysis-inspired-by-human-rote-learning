#!/usr/bin/env python
"""
Telugu Poem Training Script
Training script for Telugu poem generation using IndicBERT/MuRIL.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.telugu_backbone import create_telugu_generator
from src.preprocessing.telugu_cleaner import TeluguTextCleaner


class TeluguPoemDataset(Dataset):
    """Dataset for Telugu poems."""
    
    def __init__(self, poems, tokenizer, max_length=128):
        self.poems = poems
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cleaner = TeluguTextCleaner()
    
    def __len__(self):
        return len(self.poems)
    
    def __getitem__(self, idx):
        poem = self.poems[idx]
        text = poem.get('text', '') if isinstance(poem, dict) else poem
        
        # Clean Telugu text
        text = self.cleaner.clean(text)
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Labels for language modeling
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def train_telugu():
    """Train Telugu poem generator."""
    
    print("=" * 60)
    print("🎯 Telugu Poem Training (తెలుగు కవిత్వ శిక్షణ)")
    print("=" * 60)
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS (Metal) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Config - Optimized for 8GB RAM Mac
    config = {
        'model_type': 'distilmbert',  # PUBLIC model, smaller (134M params)
        'batch_size': 2,              # Reduced for memory
        'max_length': 64,             # Reduced for memory
        'epochs': 5,
        'learning_rate': 3e-5,
        'log_every': 10,
        'save_every': 100,
    }
    
    print(f"\nConfig: {config}")
    
    # Load or create Telugu dataset
    data_path = Path("data/processed/telugu_poems.json")
    
    if not data_path.exists():
        print("\n📚 Creating sample Telugu dataset...")
        from src.preprocessing.telugu_cleaner import create_sample_telugu_dataset
        poems = create_sample_telugu_dataset(str(data_path))
    else:
        with open(data_path, encoding='utf-8') as f:
            poems = json.load(f)
    
    print(f"Loaded {len(poems)} Telugu poems")
    
    # Create model
    print(f"\n🧠 Loading {config['model_type']} model...")
    model = create_telugu_generator(config['model_type'], freeze_backbone=True)
    model.to(device)
    
    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Create dataset
    dataset = TeluguPoemDataset(poems, model.tokenizer, max_length=config['max_length'])
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['learning_rate']
    )
    
    total_steps = len(dataloader) * config['epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop
    print(f"\n🚀 Starting Telugu training...")
    print(f"Total batches per epoch: {len(dataloader)}")
    
    model.train()
    global_step = 0
    best_loss = float('inf')
    
    checkpoints_dir = Path("checkpoints/telugu")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['epochs']):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            
            if loss is None:
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1
            
            if (batch_idx + 1) % config['log_every'] == 0:
                avg_loss = epoch_loss / batch_count
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{config['epochs']} | "
                      f"Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
            
            if global_step % config['save_every'] == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'loss': loss.item()
                }, checkpoints_dir / f"checkpoint_step_{global_step}.pt")
                print(f"  💾 Saved checkpoint at step {global_step}")
        
        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
            print(f"\n📊 Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f}")
            
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(model.state_dict(), checkpoints_dir / "best_model.pt")
                print(f"  ⭐ New best model saved!")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_loss': best_loss
    }, checkpoints_dir / "final_model.pt")
    
    print("\n" + "=" * 60)
    print("✅ Telugu Training Complete!")
    print("=" * 60)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {checkpoints_dir}/")
    
    # Test generation
    print("\n📝 Testing Telugu generation...")
    model.eval()
    
    test_prompts = [
        "చందమామ రావే",
        "తెలుగు భాష",
        "నా తెలుగు తల్లికి"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        generated = model.generate(prompt, max_length=50, temperature=0.8)
        print(generated)


if __name__ == "__main__":
    train_telugu()
