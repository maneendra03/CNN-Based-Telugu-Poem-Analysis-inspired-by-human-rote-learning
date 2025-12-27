"""
Enhanced Training Pipeline for Telugu Poem Generation
======================================================
Professional training with:
1. Mixed precision training (AMP)
2. Gradient accumulation for larger effective batch sizes
3. Learning rate scheduling with warmup
4. Validation metrics and early stopping
5. Checkpoint management
6. Repetition-aware loss functions
7. Coverage loss for attention diversity

Author: Telugu Poem Generation System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
import math
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = 'bert-base-multilingual-cased'
    freeze_encoder: bool = True
    
    # Training
    epochs: int = 50
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Scheduler
    scheduler_type: str = 'cosine'  # 'cosine', 'onecycle', 'constant'
    
    # Mixed precision
    use_amp: bool = True
    
    # Regularization
    label_smoothing: float = 0.1
    dropout: float = 0.2
    
    # Losses
    coverage_weight: float = 0.1
    repetition_loss_weight: float = 0.2
    
    # Validation
    val_every_n_steps: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpoints
    checkpoint_dir: str = 'checkpoints/telugu_v3'
    save_every_n_steps: int = 500
    max_checkpoints: int = 5
    
    # Data
    max_length: int = 128
    num_workers: int = 4


class TeluguPoemDataset(Dataset):
    """Dataset for Telugu poems."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 128,
        add_special_tokens: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.poems = json.load(f)
        
        # Handle different data formats
        if isinstance(self.poems, dict) and 'poems' in self.poems:
            self.poems = self.poems['poems']
        
        # Extract text
        self.texts = []
        for poem in self.poems:
            if isinstance(poem, str):
                self.texts.append(poem)
            elif isinstance(poem, dict):
                text = poem.get('text', poem.get('content', poem.get('poem', '')))
                if text:
                    self.texts.append(text)
        
        print(f"📚 Loaded {len(self.texts)} poems from {data_path}")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=self.add_special_tokens
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Labels (shifted for LM)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class RepetitionAwareLoss(nn.Module):
    """
    Custom loss that penalizes repetitive outputs.
    Combines cross-entropy with repetition penalty.
    """
    
    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        label_smoothing: float = 0.1,
        repetition_penalty_weight: float = 0.2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.label_smoothing = label_smoothing
        self.rep_weight = repetition_penalty_weight
        
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: [batch, seq, vocab]
            labels: [batch, seq]
            input_ids: [batch, seq] original input for repetition calc
        """
        batch_size, seq_len = labels.shape
        
        # Cross-entropy loss
        ce_loss = self.ce_loss(
            logits.view(-1, self.vocab_size),
            labels.view(-1)
        )
        
        # Repetition penalty loss
        rep_loss = torch.tensor(0.0, device=logits.device)
        if input_ids is not None and self.rep_weight > 0:
            # Get predictions
            preds = logits.argmax(dim=-1)  # [batch, seq]
            
            for b in range(batch_size):
                # Count token frequencies
                valid_mask = labels[b] != -100
                valid_preds = preds[b][valid_mask]
                
                if len(valid_preds) > 0:
                    # Calculate entropy of token distribution
                    unique, counts = torch.unique(valid_preds, return_counts=True)
                    probs = counts.float() / counts.sum()
                    entropy = -(probs * torch.log(probs + 1e-10)).sum()
                    
                    # Penalty for low entropy (more repetition)
                    max_entropy = math.log(len(unique) + 1)
                    rep_loss = rep_loss + (1 - entropy / max_entropy)
            
            rep_loss = rep_loss / batch_size
        
        total_loss = ce_loss + self.rep_weight * rep_loss
        
        metrics = {
            'ce_loss': ce_loss.item(),
            'rep_loss': rep_loss.item() if isinstance(rep_loss, torch.Tensor) else rep_loss
        }
        
        return total_loss, metrics


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
    
    def reset(self):
        self.counter = 0
        self.should_stop = False


class CheckpointManager:
    """Manages checkpoint saving and loading."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Path] = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object],
        epoch: int,
        step: int,
        loss: float,
        config: TrainingConfig,
        is_best: bool = False
    ) -> Path:
        """Save checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch{epoch}_step{step}_{timestamp}.pt"
        path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'config': asdict(config)
        }
        
        torch.save(checkpoint, path)
        self.checkpoints.append(path)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"   💾 Saved best model (loss: {loss:.4f})")
        
        # Remove old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_path = self.checkpoints.pop(0)
            if old_path.exists() and 'best' not in old_path.name:
                old_path.unlink()
        
        return path
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        path: Optional[str] = None
    ) -> Dict:
        """Load checkpoint."""
        if path is None:
            # Load best model
            path = self.checkpoint_dir / "best_model.pt"
        
        checkpoint = torch.load(path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"📥 Loaded checkpoint from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
        
        return checkpoint


class EnhancedTrainer:
    """
    Enhanced trainer for Telugu poem generation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True
            )
        
        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': config.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            eps=1e-8
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * config.epochs // config.gradient_accumulation_steps
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        if config.scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config.learning_rate,
                total_steps=total_steps,
                pct_start=config.warmup_ratio,
                anneal_strategy='cos'
            )
        elif config.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=warmup_steps,
                T_mult=2,
                eta_min=config.learning_rate * 0.01
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.loss_fn = RepetitionAwareLoss(
            vocab_size=model.vocab_size,
            label_smoothing=config.label_smoothing,
            repetition_penalty_weight=config.repetition_loss_weight
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            config.max_checkpoints
        )
        
        # Metrics tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train(self) -> Dict[str, List[float]]:
        """Run training."""
        print("\n" + "="*60)
        print("🚀 Starting Enhanced Training")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Mixed precision: {self.config.use_amp}")
        print("="*60)
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Train epoch
            train_loss = self._train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = None
            if self.val_loader:
                val_loss = self._validate()
                self.val_losses.append(val_loss)
                
                # Check for best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                # Save checkpoint
                self.checkpoint_manager.save(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.global_step,
                    val_loss,
                    self.config,
                    is_best=is_best
                )
                
                # Early stopping
                if self.early_stopping(val_loss):
                    print(f"\n⏹️ Early stopping triggered at epoch {epoch}")
                    break
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1}/{self.config.epochs}")
            print(f"   Train Loss: {train_loss:.4f}")
            if val_loss:
                print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Time: {epoch_time:.1f}s")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print(f"   Best Val Loss: {self.best_val_loss:.4f}")
        print("="*60)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def _train_epoch(self, epoch: int) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}",
            leave=False
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss']
                    
                    # Add coverage loss
                    if 'coverage_loss' in outputs:
                        loss = loss + self.config.coverage_weight * outputs['coverage_loss']
                    
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward with scaler
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss']
                
                if 'coverage_loss' in outputs:
                    loss = loss + self.config.coverage_weight * outputs['coverage_loss']
                
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Validation during training
                if (self.val_loader and 
                    self.global_step % self.config.val_every_n_steps == 0):
                    val_loss = self._validate()
                    self.model.train()
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.checkpoint_manager.save(
                            self.model, self.optimizer, self.scheduler,
                            epoch, self.global_step, val_loss,
                            self.config, is_best=True
                        )
            
            # Update progress bar
            progress.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _validate(self) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs['loss'].item()
            num_batches += 1
        
        return total_loss / num_batches


def train_enhanced_model(
    model,
    train_data_path: str,
    val_data_path: Optional[str] = None,
    config: Optional[TrainingConfig] = None
):
    """
    Convenience function to train enhanced model.
    
    Args:
        model: TeluguPoemGeneratorV3 instance
        train_data_path: Path to training data JSON
        val_data_path: Path to validation data JSON
        config: Training configuration
    """
    if config is None:
        config = TrainingConfig()
    
    # Create datasets
    train_dataset = TeluguPoemDataset(
        train_data_path,
        model.tokenizer,
        config.max_length
    )
    
    val_dataset = None
    if val_data_path:
        val_dataset = TeluguPoemDataset(
            val_data_path,
            model.tokenizer,
            config.max_length
        )
    
    # Create trainer
    trainer = EnhancedTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    # Train
    return trainer.train()


if __name__ == "__main__":
    print("Enhanced Trainer Module")
    print("="*60)
    print("Usage:")
    print("  from src.training.enhanced_trainer import train_enhanced_model")
    print("  from src.models.enhanced_generator import create_enhanced_generator")
    print()
    print("  model = create_enhanced_generator()")
    print("  results = train_enhanced_model(model, 'data/processed/telugu_train.json')")
