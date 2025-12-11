"""
Hierarchical RNN Module
Multi-level RNN for understanding poetry at different granularities:
- Character-level: Syllables, phonemes, rhymes
- Line-level: Semantic meaning, poetic flow
- Poem-level: Overall structure and theme
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class CharacterLevelRNN(nn.Module):
    """
    Character-level RNN for capturing fine-grained patterns:
    - Syllable structures
    - Phonetic patterns
    - Rhyme schemes
    - Alliteration and assonance
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        rnn_type: str = "lstm"
    ):
        """
        Initialize Character-level RNN.
        
        Args:
            input_dim: Dimension of character embeddings
            hidden_size: Hidden state size
            num_layers: Number of RNN layers
            bidirectional: Use bidirectional RNN
            dropout: Dropout rate
            rnn_type: Type of RNN ("lstm", "gru")
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # RNN layer
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        
        self.is_lstm = rnn_type == "lstm"
        
        # Output dimension
        self.output_dim = hidden_size * self.num_directions
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through character-level RNN.
        
        Args:
            x: Character embeddings [batch_size, seq_len, input_dim]
            lengths: Sequence lengths for packing (optional)
            
        Returns:
            Tuple of:
            - Sequence outputs [batch_size, seq_len, output_dim]
            - Final hidden state [batch_size, output_dim]
        """
        if lengths is not None:
            # Pack padded sequence for efficiency
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, hidden = self.rnn(x_packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True
            )
        else:
            outputs, hidden = self.rnn(x)
        
        # Get final hidden state
        if self.is_lstm:
            hidden = hidden[0]  # Take h, not c
        
        # Combine bidirectional hidden states
        if self.bidirectional:
            # [num_layers * 2, batch, hidden] -> [batch, hidden * 2]
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]
        
        # Apply layer norm and dropout
        outputs = self.layer_norm(outputs)
        outputs = self.dropout(outputs)
        
        return outputs, hidden
    
    def get_output_dim(self) -> int:
        return self.output_dim


class LineLevelRNN(nn.Module):
    """
    Line-level RNN for understanding semantic flow between lines:
    - Line-to-line coherence
    - Thematic progression
    - Enjambment and flow
    - Stanza structure
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        """
        Initialize Line-level RNN.
        
        Args:
            input_dim: Dimension of line representations
            hidden_size: Hidden state size
            num_layers: Number of RNN layers
            bidirectional: Use bidirectional RNN
            dropout: Dropout rate
            use_attention: Use self-attention over lines
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_attention = use_attention
        
        # LSTM for processing lines
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.output_dim = hidden_size * self.num_directions
        
        # Self-attention for line relationships
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.output_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(self.output_dim)
        
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        line_representations: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through line-level RNN.
        
        Args:
            line_representations: Line embeddings [batch_size, num_lines, input_dim]
            mask: Line mask [batch_size, num_lines] (optional)
            
        Returns:
            Tuple of:
            - Line outputs [batch_size, num_lines, output_dim]
            - Poem representation [batch_size, output_dim]
        """
        # RNN processing
        outputs, (hidden, _) = self.rnn(line_representations)
        
        # Apply self-attention
        if self.use_attention:
            # Create attention mask
            attn_mask = None
            if mask is not None:
                attn_mask = ~mask.bool()  # Invert for attention
            
            attn_out, _ = self.attention(
                outputs, outputs, outputs,
                key_padding_mask=attn_mask
            )
            outputs = self.attention_norm(outputs + attn_out)
        
        # Layer norm and dropout
        outputs = self.layer_norm(outputs)
        outputs = self.dropout(outputs)
        
        # Combine bidirectional hidden states for poem representation
        if self.bidirectional:
            poem_repr = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            poem_repr = hidden[-1]
        
        return outputs, poem_repr
    
    def get_output_dim(self) -> int:
        return self.output_dim


class HierarchicalRNN(nn.Module):
    """
    Complete Hierarchical RNN combining character and line levels.
    
    Architecture:
    1. Character-level RNN processes each line's characters
    2. Line representations are formed from character outputs
    3. Line-level RNN processes line sequence
    4. Final poem representation combines both levels
    
    This hierarchical approach is inspired by how humans understand
    poetry at multiple levels simultaneously.
    """
    
    def __init__(
        self,
        char_input_dim: int,
        word_input_dim: int,
        char_hidden_size: int = 256,
        line_hidden_size: int = 512,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        """
        Initialize Hierarchical RNN.
        
        Args:
            char_input_dim: Dimension of character embeddings
            word_input_dim: Dimension of word embeddings
            char_hidden_size: Hidden size for character RNN
            line_hidden_size: Hidden size for line RNN
            num_layers: Number of RNN layers at each level
            bidirectional: Use bidirectional RNNs
            dropout: Dropout rate
            use_attention: Use attention mechanisms
        """
        super().__init__()
        
        # Character-level processing
        self.char_rnn = CharacterLevelRNN(
            input_dim=char_input_dim,
            hidden_size=char_hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )
        
        # Word-level processing (for semantic understanding within lines)
        self.word_rnn = nn.LSTM(
            input_size=word_input_dim,
            hidden_size=char_hidden_size,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Combine char and word representations
        char_out_dim = self.char_rnn.get_output_dim()
        word_out_dim = char_hidden_size * (2 if bidirectional else 1)
        
        self.line_combiner = nn.Linear(
            char_out_dim + word_out_dim,
            line_hidden_size
        )
        
        # Line-level processing
        self.line_rnn = LineLevelRNN(
            input_dim=line_hidden_size,
            hidden_size=line_hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            use_attention=use_attention
        )
        
        # Final output dimension
        self.output_dim = self.line_rnn.get_output_dim()
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(
        self,
        char_embeddings: torch.Tensor,
        word_embeddings: torch.Tensor,
        line_lengths: Optional[torch.Tensor] = None,
        num_lines: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through hierarchical RNN.
        
        Args:
            char_embeddings: Character embeddings 
                [batch * num_lines, max_chars_per_line, char_dim]
            word_embeddings: Word embeddings
                [batch * num_lines, max_words_per_line, word_dim]
            line_lengths: Length of each line (optional)
            num_lines: Number of lines per poem (optional)
            
        Returns:
            Tuple of:
            - Character-level features [batch, num_lines, max_chars, char_out_dim]
            - Line-level features [batch, num_lines, line_out_dim]
            - Poem representation [batch, output_dim]
        """
        batch_times_lines = char_embeddings.size(0)
        
        # Process characters
        char_outputs, char_hidden = self.char_rnn(char_embeddings, line_lengths)
        
        # Process words
        word_outputs, (word_hidden, _) = self.word_rnn(word_embeddings)
        
        # Get word representation (last hidden state)
        if self.char_rnn.bidirectional:
            word_repr = torch.cat([word_hidden[-2], word_hidden[-1]], dim=-1)
        else:
            word_repr = word_hidden[-1]
        
        # Combine character and word representations for each line
        line_repr = torch.cat([char_hidden, word_repr], dim=-1)
        line_repr = self.line_combiner(line_repr)  # [batch * num_lines, line_hidden]
        
        # Reshape for line-level processing
        # Assuming equal number of lines per poem for simplicity
        if num_lines is not None:
            batch_size = batch_times_lines // num_lines.max().item()
            max_lines = num_lines.max().item()
        else:
            # Assume fixed number of lines
            batch_size = batch_times_lines // 8  # Default 8 lines
            max_lines = 8
        
        line_repr = line_repr.view(batch_size, max_lines, -1)
        
        # Process lines
        line_outputs, poem_repr = self.line_rnn(line_repr)
        
        # Create hierarchical representation
        # Combine line-level and poem-level information
        poem_expanded = poem_repr.unsqueeze(1).expand_as(line_outputs)
        combined = torch.cat([line_outputs, poem_expanded], dim=-1)
        hierarchical_repr = self.fusion(combined)
        hierarchical_repr = self.layer_norm(hierarchical_repr)
        
        return char_outputs, hierarchical_repr, poem_repr
    
    def get_output_dim(self) -> int:
        return self.output_dim


class SimpleHierarchicalRNN(nn.Module):
    """
    Simplified hierarchical RNN that works with pre-computed line embeddings.
    More practical for initial experiments.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize Simplified Hierarchical RNN.
        
        Args:
            input_dim: Input embedding dimension
            hidden_size: Hidden state size
            num_layers: Number of RNN layers
            bidirectional: Use bidirectional RNN
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = hidden_size * self.num_directions
        
        # Word-level LSTM (within lines)
        self.word_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Line-level LSTM (across lines)
        self.line_lstm = nn.LSTM(
            input_size=self.output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention for line importance
        self.line_attention = nn.Sequential(
            nn.Linear(self.output_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        line_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input embeddings [batch_size, num_lines, max_words, input_dim]
               or [batch_size, seq_len, input_dim] for flat input
            line_mask: Mask for valid lines [batch_size, num_lines]
            
        Returns:
            Tuple of:
            - Hierarchical features [batch_size, num_lines, output_dim]
            - Poem representation [batch_size, output_dim]
        """
        if x.dim() == 4:
            # Hierarchical input
            batch_size, num_lines, max_words, input_dim = x.shape
            
            # Process each line
            x_flat = x.view(batch_size * num_lines, max_words, input_dim)
            word_outputs, (word_hidden, _) = self.word_lstm(x_flat)
            
            # Get line representations from hidden states
            if self.num_directions == 2:
                line_repr = torch.cat([word_hidden[-2], word_hidden[-1]], dim=-1)
            else:
                line_repr = word_hidden[-1]
            
            line_repr = line_repr.view(batch_size, num_lines, -1)
        else:
            # Flat input - treat entire sequence as line representations
            line_repr, _ = self.word_lstm(x)
        
        # Process lines
        line_outputs, (line_hidden, _) = self.line_lstm(line_repr)
        
        # Apply attention for poem representation
        attn_weights = self.line_attention(line_outputs)  # [batch, lines, 1]
        
        if line_mask is not None:
            attn_weights = attn_weights.masked_fill(
                ~line_mask.unsqueeze(-1).bool(), float('-inf')
            )
        
        attn_weights = F.softmax(attn_weights, dim=1)
        poem_repr = (line_outputs * attn_weights).sum(dim=1)
        
        # Final processing
        line_outputs = self.layer_norm(line_outputs)
        line_outputs = self.dropout(line_outputs)
        
        return line_outputs, poem_repr
    
    def get_output_dim(self) -> int:
        return self.output_dim


if __name__ == "__main__":
    # Test Hierarchical RNN
    batch_size = 4
    num_lines = 8
    max_chars = 50
    max_words = 15
    char_dim = 64
    word_dim = 256
    
    # Create model
    model = HierarchicalRNN(
        char_input_dim=char_dim,
        word_input_dim=word_dim,
        char_hidden_size=128,
        line_hidden_size=256
    )
    
    # Random inputs
    char_emb = torch.randn(batch_size * num_lines, max_chars, char_dim)
    word_emb = torch.randn(batch_size * num_lines, max_words, word_dim)
    num_lines_tensor = torch.full((batch_size,), num_lines)
    
    # Forward pass
    char_out, line_out, poem_repr = model(char_emb, word_emb, num_lines=num_lines_tensor)
    
    print(f"Character outputs shape: {char_out.shape}")
    print(f"Line outputs shape: {line_out.shape}")
    print(f"Poem representation shape: {poem_repr.shape}")
    print(f"Output dimension: {model.get_output_dim()}")
    
    # Test Simple Hierarchical RNN
    simple_model = SimpleHierarchicalRNN(input_dim=256)
    
    x = torch.randn(batch_size, num_lines, max_words, 256)
    line_out, poem_repr = simple_model(x)
    
    print(f"\nSimple model:")
    print(f"Line outputs shape: {line_out.shape}")
    print(f"Poem representation shape: {poem_repr.shape}")
