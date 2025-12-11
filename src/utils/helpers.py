"""
Utility functions and helpers for the Poem Learner project.
"""

import os
import random
import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(use_cuda: bool = True, cuda_device: int = 0) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        use_cuda: Whether to use CUDA if available
        cuda_device: Which CUDA device to use
        
    Returns:
        torch.device object
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_device}')
        print(f"Using CUDA device: {torch.cuda.get_device_name(cuda_device)}")
    elif use_cuda and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS device")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the config YAML file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_number(num: int) -> str:
    """
    Format a large number with K, M, B suffixes.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string
    """
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return str(num)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update the meter.
        
        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            mode: "min" for minimizing (loss), "max" for maximizing (accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if we should stop training.
        
        Args:
            score: Current validation score
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def create_attention_mask(
    seq_lengths: torch.Tensor,
    max_length: int
) -> torch.Tensor:
    """
    Create attention mask for variable length sequences.
    
    Args:
        seq_lengths: Tensor of sequence lengths [batch_size]
        max_length: Maximum sequence length
        
    Returns:
        Attention mask [batch_size, max_length]
    """
    batch_size = seq_lengths.size(0)
    mask = torch.arange(max_length).expand(batch_size, -1)
    mask = mask < seq_lengths.unsqueeze(1)
    return mask.float()


def pad_sequence(
    sequences: list,
    padding_value: int = 0
) -> torch.Tensor:
    """
    Pad a list of sequences to the same length.
    
    Args:
        sequences: List of tensors
        padding_value: Value to use for padding
        
    Returns:
        Padded tensor [batch_size, max_length]
    """
    max_len = max(seq.size(0) for seq in sequences)
    batch_size = len(sequences)
    
    padded = torch.full((batch_size, max_len), padding_value)
    for i, seq in enumerate(sequences):
        padded[i, :seq.size(0)] = seq
    
    return padded


if __name__ == "__main__":
    # Test utilities
    set_seed(42)
    device = get_device()
    
    # Test AverageMeter
    meter = AverageMeter("loss")
    for i in range(10):
        meter.update(random.random())
    print(meter)
    
    # Test EarlyStopping
    early_stop = EarlyStopping(patience=3)
    scores = [1.0, 0.9, 0.85, 0.85, 0.85, 0.85]
    for score in scores:
        if early_stop(score):
            print(f"Early stopping triggered at score {score}")
            break
    
    print(f"Best score: {early_stop.best_score}")
