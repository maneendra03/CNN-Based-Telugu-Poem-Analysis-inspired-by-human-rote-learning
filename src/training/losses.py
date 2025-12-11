"""
Loss Functions for Poem Learning
Includes standard losses and specialized losses for poetry quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PoemLoss(nn.Module):
    """
    Combined loss function for poem learning and generation.
    
    Components:
    - Language modeling loss (cross-entropy)
    - Style classification loss
    - Knowledge-based quality loss
    - Memorization regularization
    """
    
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        label_smoothing: float = 0.1,
        style_weight: float = 0.5,
        quality_weight: float = 0.3,
        memory_weight: float = 0.2,
        rhyme_weight: float = 0.5,
        ignore_index: int = -100
    ):
        """
        Initialize PoemLoss.
        
        Args:
            vocab_size: Vocabulary size
            pad_token_id: Padding token ID
            label_smoothing: Label smoothing factor
            style_weight: Weight for style loss
            quality_weight: Weight for quality loss
            memory_weight: Weight for memory regularization
            rhyme_weight: Weight for rhyme consistency loss
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        
        self.pad_token_id = pad_token_id
        self.style_weight = style_weight
        self.quality_weight = quality_weight
        self.memory_weight = memory_weight
        self.rhyme_weight = rhyme_weight
        
        # Language modeling loss
        self.lm_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        
        # Style classification loss
        self.style_loss = nn.CrossEntropyLoss()
        
        # Quality regression loss
        self.quality_loss = nn.MSELoss()
        
    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        style_logits: Optional[torch.Tensor] = None,
        style_labels: Optional[torch.Tensor] = None,
        quality_pred: Optional[torch.Tensor] = None,
        quality_target: Optional[torch.Tensor] = None,
        memory_strength: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            logits: Predicted logits [batch, seq_len, vocab_size]
            target_ids: Target token IDs [batch, seq_len]
            style_logits: Style prediction logits [batch, num_styles]
            style_labels: Style labels [batch]
            quality_pred: Predicted quality [batch, 1]
            quality_target: Target quality [batch, 1]
            memory_strength: Memory strength values for regularization
            
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # Language modeling loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = target_ids.view(-1)
        
        lm_loss = self.lm_loss(logits_flat, targets_flat)
        losses['lm_loss'] = lm_loss
        
        total_loss = lm_loss
        
        # Style classification loss
        if style_logits is not None and style_labels is not None:
            s_loss = self.style_loss(style_logits, style_labels)
            losses['style_loss'] = s_loss
            total_loss = total_loss + self.style_weight * s_loss
        
        # Quality prediction loss
        if quality_pred is not None and quality_target is not None:
            q_loss = self.quality_loss(quality_pred, quality_target)
            losses['quality_loss'] = q_loss
            total_loss = total_loss + self.quality_weight * q_loss
        
        # Memory regularization (encourage stable memorization)
        if memory_strength is not None:
            # Encourage memory strength to converge to stable values
            memory_reg = memory_strength.var()
            losses['memory_reg'] = memory_reg
            total_loss = total_loss + self.memory_weight * memory_reg
        
        losses['total_loss'] = total_loss
        
        return losses


class PerplexityMetric:
    """Metric for computing perplexity."""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.total_tokens = 0
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update with batch predictions."""
        # Flatten
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        # Create mask for non-padding tokens
        mask = targets != self.pad_token_id
        
        # Compute cross-entropy for each token
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply mask
        masked_log_probs = target_log_probs * mask.float()
        
        self.total_loss += -masked_log_probs.sum().item()
        self.total_tokens += mask.sum().item()
    
    def compute(self) -> float:
        """Compute perplexity."""
        if self.total_tokens == 0:
            return float('inf')
        avg_loss = self.total_loss / self.total_tokens
        return torch.exp(torch.tensor(avg_loss)).item()


class RhymeLoss(nn.Module):
    """
    Loss for encouraging rhyme consistency.
    Compares embeddings of line-ending words.
    """
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
        self.cosine = nn.CosineSimilarity(dim=-1)
    
    def forward(
        self,
        line_end_embeddings: torch.Tensor,
        rhyme_scheme: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rhyme consistency loss.
        
        Args:
            line_end_embeddings: Embeddings of line-ending words [batch, num_lines, dim]
            rhyme_scheme: Rhyme scheme encoding [batch, num_lines]
                         Lines with same value should rhyme
            
        Returns:
            Rhyme consistency loss
        """
        batch_size, num_lines, dim = line_end_embeddings.shape
        
        loss = 0.0
        count = 0
        
        for b in range(batch_size):
            for i in range(num_lines):
                for j in range(i + 1, num_lines):
                    should_rhyme = rhyme_scheme[b, i] == rhyme_scheme[b, j]
                    
                    sim = self.cosine(
                        line_end_embeddings[b, i],
                        line_end_embeddings[b, j]
                    )
                    
                    if should_rhyme:
                        # Maximize similarity
                        loss += (1 - sim)
                    else:
                        # Minimize similarity (with margin)
                        loss += F.relu(sim - self.margin)
                    
                    count += 1
        
        return loss / max(count, 1)


class MeterLoss(nn.Module):
    """
    Loss for encouraging meter/rhythm consistency.
    Uses predicted syllable stress patterns.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        predicted_stress: torch.Tensor,
        target_pattern: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute meter consistency loss.
        
        Args:
            predicted_stress: Predicted stress patterns [batch, num_syllables]
            target_pattern: Target meter pattern [batch, pattern_len] (will be tiled)
            
        Returns:
            Meter consistency loss
        """
        batch_size, num_syllables = predicted_stress.shape
        pattern_len = target_pattern.size(1)
        
        # Tile pattern to match prediction length
        num_repeats = (num_syllables + pattern_len - 1) // pattern_len
        tiled_pattern = target_pattern.repeat(1, num_repeats)[:, :num_syllables]
        
        # Binary cross-entropy (0=unstressed, 1=stressed)
        loss = F.binary_cross_entropy_with_logits(
            predicted_stress, tiled_pattern.float()
        )
        
        return loss


if __name__ == "__main__":
    # Test losses
    batch_size = 4
    seq_len = 30
    vocab_size = 10000
    num_styles = 10
    
    # Random predictions and targets
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    style_logits = torch.randn(batch_size, num_styles)
    style_labels = torch.randint(0, num_styles, (batch_size,))
    
    # Create loss
    loss_fn = PoemLoss(vocab_size=vocab_size)
    
    # Compute loss
    losses = loss_fn(
        logits=logits,
        target_ids=targets,
        style_logits=style_logits,
        style_labels=style_labels
    )
    
    print("Losses:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
    
    # Test perplexity
    ppl = PerplexityMetric()
    ppl.update(logits, targets)
    print(f"\nPerplexity: {ppl.compute():.2f}")
