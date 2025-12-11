"""
Memory & Attention Module
Simulates human rote learning through specialized memory cells and attention mechanisms.

Key Innovation:
- Memory cells that track "memorized" patterns
- Attention that weights familiar/repeated patterns higher
- Decay mechanism simulating forgetting
- Retention scoring for measuring memorization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class RoteLearningMemory(nn.Module):
    """
    Memory module that simulates human rote learning behavior.
    
    Key features:
    1. External memory cells that store learned patterns
    2. Repetition-based reinforcement (patterns seen more often are stronger)
    3. Decay mechanism (unused patterns fade)
    4. Similarity-based retrieval
    """
    
    def __init__(
        self,
        input_dim: int,
        memory_size: int = 256,
        num_memory_cells: int = 16,
        num_heads: int = 4,
        dropout: float = 0.2,
        decay_rate: float = 0.1
    ):
        """
        Initialize Rote Learning Memory.
        
        Args:
            input_dim: Dimension of input features
            memory_size: Size of each memory cell
            num_memory_cells: Number of memory cells (like "slots" for patterns)
            num_heads: Number of attention heads for memory access
            dropout: Dropout rate
            decay_rate: Rate at which memories decay (simulates forgetting)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.memory_size = memory_size
        self.num_memory_cells = num_memory_cells
        self.decay_rate = decay_rate
        
        # Learnable memory bank (persistent patterns)
        self.memory_bank = nn.Parameter(
            torch.randn(num_memory_cells, memory_size) * 0.02
        )
        
        # Memory strength (how well each pattern is "memorized")
        # Higher strength = more reinforced through repetition
        self.register_buffer(
            'memory_strength',
            torch.ones(num_memory_cells)
        )
        
        # Query, Key, Value projections for memory access
        self.query_proj = nn.Linear(input_dim, memory_size)
        self.key_proj = nn.Linear(memory_size, memory_size)
        self.value_proj = nn.Linear(memory_size, memory_size)
        
        # Multi-head attention for memory retrieval
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=memory_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(memory_size, input_dim)
        
        # Gate for controlling memory influence
        self.memory_gate = nn.Sequential(
            nn.Linear(input_dim + memory_size, input_dim),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        update_memory: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with memory access and update.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            update_memory: Whether to update memory strengths
            
        Returns:
            Tuple of:
            - Memory-augmented features [batch_size, seq_len, input_dim]
            - Dictionary with memory statistics
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to query
        query = self.query_proj(x)  # [batch, seq, memory_size]
        
        # Prepare memory for attention
        memory = self.memory_bank.unsqueeze(0).expand(batch_size, -1, -1)
        keys = self.key_proj(memory)
        values = self.value_proj(memory)
        
        # Weight values by memory strength (simulates stronger memories)
        strength_weights = self.memory_strength.unsqueeze(0).unsqueeze(-1)
        values = values * strength_weights
        
        # Retrieve from memory using attention
        retrieved, attn_weights = self.memory_attention(
            query, keys, values
        )  # retrieved: [batch, seq, memory_size]
        
        # Project back to input dimension
        retrieved = self.output_proj(retrieved)
        
        # Compute gating (how much to use memory vs. input)
        gate_input = torch.cat([x, self.query_proj(x)], dim=-1)
        gate = self.memory_gate(gate_input)
        
        # Combine input with retrieved memory
        output = gate * x + (1 - gate) * retrieved
        
        # Apply layer norm and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        # Update memory strengths based on usage (rote learning)
        if update_memory and self.training:
            self._update_memory_strength(attn_weights)
        
        # Compute memory statistics
        stats = {
            'attention_weights': attn_weights.mean(dim=1),  # [batch, num_cells]
            'memory_strength': self.memory_strength.clone(),
            'gate_values': gate.mean().item()
        }
        
        return output, stats
    
    def _update_memory_strength(self, attn_weights: torch.Tensor):
        """
        Update memory strength based on attention (usage).
        Frequently accessed memories become stronger (rote learning).
        
        Args:
            attn_weights: Attention weights [batch, seq, num_cells]
        """
        # Average attention across batch and sequence
        usage = attn_weights.mean(dim=(0, 1))  # [num_cells]
        
        # Reinforce used memories
        reinforcement = usage * (1 - self.decay_rate)
        
        # Apply decay to all memories
        new_strength = self.memory_strength * (1 - self.decay_rate) + reinforcement
        
        # Clamp to valid range
        self.memory_strength.copy_(new_strength.clamp(0.1, 2.0))
    
    def get_retention_score(self) -> float:
        """
        Get overall memory retention score.
        Higher score = more patterns well-memorized.
        """
        return self.memory_strength.mean().item()
    
    def reset_memory_strength(self):
        """Reset memory strengths to initial values."""
        self.memory_strength.fill_(1.0)


class RepetitionAttention(nn.Module):
    """
    Attention mechanism that specifically focuses on repetitive patterns.
    Key for rote learning - repeated patterns get higher attention.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        dropout: float = 0.2,
        repetition_bonus: float = 0.5
    ):
        """
        Initialize Repetition Attention.
        
        Args:
            input_dim: Input dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            repetition_bonus: Bonus weight for repeated patterns
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.repetition_bonus = repetition_bonus
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        # QKV projections
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        
        # Pattern similarity detector
        self.similarity_detector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with repetition-aware attention.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of:
            - Output features [batch_size, seq_len, input_dim]
            - Repetition scores [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Detect repetitive patterns
        repetition_scores = self._compute_repetition_scores(x)
        
        # Add repetition bonus to attention
        repetition_bonus = repetition_scores.unsqueeze(1).unsqueeze(-1)
        repetition_bonus = repetition_bonus.expand(-1, self.num_heads, -1, seq_len)
        attn_scores = attn_scores + self.repetition_bonus * repetition_bonus
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~mask.bool(), float('-inf'))
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        output = torch.matmul(attn_probs, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.input_dim)
        output = self.out_proj(output)
        
        return output, repetition_scores
    
    def _compute_repetition_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute how "repetitive" each position is.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            Repetition scores [batch_size, seq_len]
        """
        # Compute pairwise similarities
        x_norm = F.normalize(x, dim=-1)
        similarities = torch.matmul(x_norm, x_norm.transpose(-2, -1))
        
        # Mask self-similarity
        eye = torch.eye(x.size(1), device=x.device)
        similarities = similarities * (1 - eye)
        
        # High similarity with other positions = high repetition
        max_similarity, _ = similarities.max(dim=-1)
        
        return max_similarity


class MemoryAttentionModule(nn.Module):
    """
    Complete Memory & Attention module combining:
    1. Rote Learning Memory
    2. Repetition Attention
    3. LSTM Memory Cells (for temporal patterns)
    
    This module is the core innovation for simulating human rote learning.
    """
    
    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 512,
        memory_size: int = 256,
        num_memory_cells: int = 16,
        num_heads: int = 8,
        dropout: float = 0.2,
        decay_rate: float = 0.1
    ):
        """
        Initialize Memory & Attention Module.
        
        Args:
            input_dim: Input feature dimension
            attention_dim: Attention/hidden dimension
            memory_size: Size of memory cells
            num_memory_cells: Number of memory cells
            num_heads: Number of attention heads
            dropout: Dropout rate
            decay_rate: Memory decay rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = attention_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, attention_dim)
        
        # Rote learning memory
        self.rote_memory = RoteLearningMemory(
            input_dim=attention_dim,
            memory_size=memory_size,
            num_memory_cells=num_memory_cells,
            num_heads=num_heads // 2,
            dropout=dropout,
            decay_rate=decay_rate
        )
        
        # Repetition attention
        self.repetition_attn = RepetitionAttention(
            input_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # LSTM for sequential memory (temporal patterns)
        self.memory_lstm = nn.LSTM(
            input_size=attention_dim,
            hidden_size=attention_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Combine all memory sources
        self.combiner = nn.Sequential(
            nn.Linear(attention_dim * 3, attention_dim),
            nn.LayerNorm(attention_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Context-aware projection
        self.context_proj = nn.Linear(attention_dim, attention_dim)
        
        self.layer_norm = nn.LayerNorm(attention_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through complete memory & attention module.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            mask: Attention mask [batch_size, seq_len]
            return_stats: Whether to return memory statistics
            
        Returns:
            Tuple of:
            - Context-aware features [batch_size, seq_len, output_dim]
            - Context vector [batch_size, output_dim]
        """
        # Project input
        x = self.input_proj(x)
        
        # 1. Rote learning memory
        memory_output, memory_stats = self.rote_memory(x)
        
        # 2. Repetition attention
        attn_output, repetition_scores = self.repetition_attn(x, mask)
        
        # 3. LSTM sequential memory
        lstm_output, _ = self.memory_lstm(x)
        
        # Combine all memory sources
        combined = torch.cat([memory_output, attn_output, lstm_output], dim=-1)
        output = self.combiner(combined)
        
        # Residual connection
        output = output + x
        output = self.layer_norm(output)
        
        # Create context vector (weighted by repetition importance)
        if mask is not None:
            weights = repetition_scores * mask
        else:
            weights = repetition_scores
        
        weights = F.softmax(weights, dim=-1)
        context = (output * weights.unsqueeze(-1)).sum(dim=1)
        context = self.context_proj(context)
        
        if return_stats:
            return output, context, {
                'memory_stats': memory_stats,
                'repetition_scores': repetition_scores,
                'retention_score': self.rote_memory.get_retention_score()
            }
        
        return output, context
    
    def get_output_dim(self) -> int:
        return self.output_dim
    
    def get_retention_score(self) -> float:
        """Get the current memory retention score."""
        return self.rote_memory.get_retention_score()
    
    def reset_memory(self):
        """Reset memory to initial state."""
        self.rote_memory.reset_memory_strength()


if __name__ == "__main__":
    # Test Memory & Attention Module
    batch_size = 4
    seq_len = 50
    input_dim = 256
    
    # Create module
    module = MemoryAttentionModule(
        input_dim=input_dim,
        attention_dim=512,
        memory_size=256,
        num_memory_cells=16
    )
    
    # Random input
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    output, context, stats = module(x, mask, return_stats=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Output dimension: {module.get_output_dim()}")
    print(f"\nMemory Statistics:")
    print(f"  Retention score: {stats['retention_score']:.4f}")
    print(f"  Memory strength: {stats['memory_stats']['memory_strength']}")
    print(f"  Gate values: {stats['memory_stats']['gate_values']:.4f}")
    
    # Simulate training (memory strength should change)
    module.train()
    for i in range(5):
        output, context = module(x, mask)
        print(f"\nIteration {i+1} - Retention score: {module.get_retention_score():.4f}")
