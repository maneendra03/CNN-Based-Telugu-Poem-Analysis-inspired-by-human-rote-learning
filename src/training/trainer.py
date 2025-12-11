"""
Trainer Module
Handles training loop, optimization, and logging for the Poem Learner model.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from typing import Dict, Optional, Callable, List
from pathlib import Path
import json

# Import local modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.poem_learner import PoemLearner
from src.training.losses import PoemLoss, PerplexityMetric
from src.utils.helpers import (
    set_seed, get_device, AverageMeter, EarlyStopping
)


class Trainer:
    """
    Trainer for the Poem Learner model.
    
    Features:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Checkpoint management
    - Logging (console, file, W&B optional)
    """
    
    def __init__(
        self,
        model: PoemLearner,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Trainer.
        
        Args:
            model: PoemLearner model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            device: Device to train on
        """
        self.config = config or {}
        self.device = device or get_device()
        
        # Model
        self.model = model.to(self.device)
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Training config
        training_config = self.config.get('training', {})
        self.epochs = training_config.get('epochs', 100)
        self.learning_rate = training_config.get('learning_rate', 1e-4)
        self.weight_decay = training_config.get('weight_decay', 0.01)
        self.warmup_steps = training_config.get('warmup_steps', 1000)
        self.max_grad_norm = training_config.get('max_grad_norm', 1.0)
        self.eval_every = training_config.get('eval_every', 1)
        self.save_every = self.config.get('logging', {}).get('save_every', 5)
        
        # Mixed precision
        self.use_amp = self.config.get('device', {}).get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        scheduler_type = training_config.get('scheduler', 'cosine')
        total_steps = len(train_loader) * self.epochs
        
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )
        else:
            self.scheduler = LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
            )
        
        # Loss function
        vocab_size = self.config.get('vocab_size', model.vocab_size)
        self.criterion = PoemLoss(vocab_size=vocab_size)
        
        # Early stopping
        patience = training_config.get('early_stopping_patience', 10)
        self.early_stopping = EarlyStopping(patience=patience, mode='min')
        
        # Logging
        log_config = self.config.get('logging', {})
        self.log_dir = Path(log_config.get('log_dir', 'logs'))
        self.checkpoint_dir = Path(log_config.get('checkpoint_dir', 'checkpoints'))
        self.log_every = log_config.get('log_every', 100)
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # W&B logging (optional)
        self.use_wandb = log_config.get('use_wandb', False)
        self.wandb_run = None
        
        # Metrics tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history: List[Dict] = []
    
    def train(self) -> Dict:
        """
        Run training loop.
        
        Returns:
            Training history dictionary
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        # Initialize W&B if enabled
        if self.use_wandb:
            self._init_wandb()
        
        for epoch in range(self.epochs):
            # Training epoch
            train_metrics = self._train_epoch(epoch)
            
            # Validation
            if self.val_loader is not None and (epoch + 1) % self.eval_every == 0:
                val_metrics = self._validate()
                
                # Early stopping check
                if self.early_stopping(val_metrics['loss']):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self._save_checkpoint('best_model.pt', epoch, val_metrics)
            else:
                val_metrics = {}
            
            # Periodic checkpoint
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch, train_metrics)
            
            # Log epoch summary
            self._log_epoch(epoch, train_metrics, val_metrics)
        
        # Final save
        self._save_checkpoint('final_model.pt', self.epochs - 1, train_metrics)
        
        return self._get_training_summary()
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        loss_meter = AverageMeter('loss')
        lm_loss_meter = AverageMeter('lm_loss')
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch.get('target_ids', input_ids).to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            style_labels = batch.get('style_labels', None)
            if style_labels is not None:
                style_labels = style_labels.to(self.device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    target_ids=target_ids,
                    attention_mask=attention_mask
                )
                
                losses = self.criterion(
                    logits=outputs['logits'],
                    target_ids=target_ids,
                    style_logits=outputs.get('style_logits'),
                    style_labels=style_labels
                )
                
                loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Update meters
            loss_meter.update(loss.item(), input_ids.size(0))
            lm_loss_meter.update(losses['lm_loss'].item(), input_ids.size(0))
            
            self.global_step += 1
            
            # Log progress
            if (batch_idx + 1) % self.log_every == 0:
                self._log_step(epoch, batch_idx, loss_meter, lm_loss_meter)
        
        elapsed = time.time() - start_time
        
        return {
            'loss': loss_meter.avg,
            'lm_loss': lm_loss_meter.avg,
            'time': elapsed,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        loss_meter = AverageMeter('val_loss')
        perplexity_metric = PerplexityMetric()
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch.get('target_ids', input_ids).to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                target_ids=target_ids,
                attention_mask=attention_mask
            )
            
            losses = self.criterion(
                logits=outputs['logits'],
                target_ids=target_ids
            )
            
            loss_meter.update(losses['total_loss'].item(), input_ids.size(0))
            perplexity_metric.update(outputs['logits'], target_ids)
        
        metrics = {
            'loss': loss_meter.avg,
            'perplexity': perplexity_metric.compute()
        }
        
        # Get memorization metrics
        mem_metrics = self.model.get_memorization_metrics()
        metrics.update(mem_metrics)
        
        return metrics
    
    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        
        print(f"Loaded checkpoint from {path}")
        return checkpoint['epoch'], checkpoint['metrics']
    
    def _log_step(
        self,
        epoch: int,
        batch_idx: int,
        loss_meter: AverageMeter,
        lm_loss_meter: AverageMeter
    ):
        """Log training step."""
        lr = self.scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch} | Step {batch_idx} | "
            f"Loss: {loss_meter.avg:.4f} | "
            f"LM Loss: {lm_loss_meter.avg:.4f} | "
            f"LR: {lr:.2e}"
        )
        
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log({
                'train/loss': loss_meter.avg,
                'train/lm_loss': lm_loss_meter.avg,
                'train/learning_rate': lr,
                'global_step': self.global_step
            })
    
    def _log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log epoch summary."""
        msg = f"\n{'='*60}\n"
        msg += f"Epoch {epoch + 1}/{self.epochs}\n"
        msg += f"Train Loss: {train_metrics['loss']:.4f}\n"
        
        if val_metrics:
            msg += f"Val Loss: {val_metrics['loss']:.4f}\n"
            msg += f"Val Perplexity: {val_metrics.get('perplexity', 0):.2f}\n"
            if 'retention_score' in val_metrics:
                msg += f"Memory Retention: {val_metrics['retention_score']:.4f}\n"
        
        msg += f"Time: {train_metrics['time']:.1f}s\n"
        msg += f"{'='*60}\n"
        
        print(msg)
        
        # Save to history
        self.training_history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics
        })
        
        # Log to W&B
        if self.use_wandb and self.wandb_run:
            import wandb
            log_dict = {f'train/{k}': v for k, v in train_metrics.items()}
            log_dict.update({f'val/{k}': v for k, v in val_metrics.items()})
            wandb.log(log_dict)
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            
            project = self.config.get('logging', {}).get('wandb_project', 'poem-learner')
            self.wandb_run = wandb.init(
                project=project,
                config=self.config,
                name=f"run_{time.strftime('%Y%m%d_%H%M%S')}"
            )
        except ImportError:
            print("Warning: wandb not installed. Skipping W&B logging.")
            self.use_wandb = False
    
    def _get_training_summary(self) -> Dict:
        """Get summary of training."""
        return {
            'total_epochs': len(self.training_history),
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.training_history[-1]['train']['loss'] if self.training_history else None,
            'history': self.training_history
        }


if __name__ == "__main__":
    # Test trainer with dummy data
    from torch.utils.data import TensorDataset
    
    # Create dummy model
    model = PoemLearner(
        vocab_size=1000,
        embedding_dim=128,
        hidden_dim=256
    )
    
    # Create dummy data
    batch_size = 8
    seq_len = 50
    num_samples = 100
    
    input_ids = torch.randint(1, 1000, (num_samples, seq_len))
    target_ids = torch.randint(1, 1000, (num_samples, seq_len))
    attention_mask = torch.ones(num_samples, seq_len)
    
    dataset = TensorDataset(input_ids, target_ids, attention_mask)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda x: {
            'input_ids': torch.stack([b[0] for b in x]),
            'target_ids': torch.stack([b[1] for b in x]),
            'attention_mask': torch.stack([b[2] for b in x])
        }
    )
    
    # Create trainer
    config = {
        'training': {
            'epochs': 2,
            'learning_rate': 1e-3
        },
        'logging': {
            'log_every': 5
        }
    }
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=config
    )
    
    # Run training
    summary = trainer.train()
    print("\nTraining Summary:")
    print(f"Final train loss: {summary['final_train_loss']:.4f}")
