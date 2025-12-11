"""
CNN Feature Extraction Module
1D Convolutional Neural Network for capturing local patterns in poetry:
- Rhyme patterns
- Rhythm and meter
- Alliteration and assonance
- Local phonetic structures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class CNNFeatureExtractor(nn.Module):
    """
    1D CNN for extracting local pattern features from poem embeddings.
    
    Uses multiple kernel sizes to capture patterns at different scales:
    - Small kernels (2-3): Characters, phonemes, rhymes
    - Medium kernels (4-5): Words, short phrases
    - Large kernels (6-7): Phrases, rhythm patterns
    
    This is a key innovation for detecting rhythm, rhyme, and local
    poetic structures that traditional RNNs might miss.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_filters: int = 256,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True,
        use_residual: bool = True
    ):
        """
        Initialize CNN Feature Extractor.
        
        Args:
            input_dim: Dimension of input embeddings
            num_filters: Number of filters per kernel size
            kernel_sizes: List of kernel sizes to use
            dropout: Dropout rate
            activation: Activation function ("relu", "gelu", "leaky_relu")
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Convolutional layers for each kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2  # Same padding to preserve sequence length
            )
            for k in kernel_sizes
        ])
        
        # Batch normalization layers
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(num_filters)
                for _ in kernel_sizes
            ])
        
        # Second layer of convolutions for deeper feature extraction
        self.conv2 = nn.ModuleList([
            nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2
            )
            for k in kernel_sizes
        ])
        
        if use_batch_norm:
            self.batch_norms2 = nn.ModuleList([
                nn.BatchNorm1d(num_filters)
                for _ in kernel_sizes
            ])
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.ReLU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        self.output_dim = num_filters * len(kernel_sizes)
        
        # Projection layer for combining features
        self.projection = nn.Linear(self.output_dim, self.output_dim)
        
        # Residual projection if dimensions don't match
        if use_residual:
            self.residual_proj = nn.Linear(input_dim, self.output_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract CNN features from input embeddings.
        
        Args:
            x: Input embeddings [batch_size, seq_len, input_dim]
            mask: Attention mask [batch_size, seq_len] (optional)
            
        Returns:
            Tuple of:
            - Sequence features [batch_size, seq_len, output_dim]
            - Pooled features [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Save input for residual connection
        residual = x
        
        # Transpose for Conv1d: [batch, channels, length]
        x = x.transpose(1, 2)
        
        # Apply convolutions with different kernel sizes
        conv_outputs = []
        for i, conv in enumerate(self.convs):
            # First conv layer
            h = conv(x)
            if self.use_batch_norm:
                h = self.batch_norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            
            # Second conv layer
            h = self.conv2[i](h)
            if self.use_batch_norm:
                h = self.batch_norms2[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            
            conv_outputs.append(h)
        
        # Concatenate outputs from different kernel sizes
        # [batch, num_filters * num_kernels, seq_len]
        x = torch.cat(conv_outputs, dim=1)
        
        # Transpose back: [batch, seq_len, output_dim]
        x = x.transpose(1, 2)
        
        # Apply mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        # Project combined features
        x = self.projection(x)
        
        # Residual connection
        if self.use_residual:
            residual = self.residual_proj(residual)
            x = x + residual
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Pooling for fixed-size representation
        if mask is not None:
            # Masked average pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x_masked = x * mask_expanded
            pooled = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            # Max pooling and average pooling combined
            max_pooled, _ = x.max(dim=1)
            avg_pooled = x.mean(dim=1)
            pooled = (max_pooled + avg_pooled) / 2
        
        return x, pooled
    
    def get_output_dim(self) -> int:
        """Get the output feature dimension."""
        return self.output_dim


class MultiScaleCNN(nn.Module):
    """
    Multi-scale CNN that processes input at different granularities.
    Useful for capturing patterns at character, word, and phrase levels.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_scales: int = 3,
        dropout: float = 0.3
    ):
        """
        Initialize Multi-scale CNN.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension for each scale
            num_scales: Number of scales to process
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_scales = num_scales
        
        # CNN for each scale
        self.scale_cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=2**(i+1)+1, padding=2**i),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for i in range(num_scales)
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * num_scales, hidden_dim)
        self.output_dim = hidden_dim
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale CNN.
        
        Args:
            x: Input [batch_size, seq_len, input_dim]
            
        Returns:
            Features [batch_size, seq_len, hidden_dim]
        """
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        
        scale_outputs = []
        for cnn in self.scale_cnns:
            out = cnn(x)
            scale_outputs.append(out)
        
        # Concatenate scales
        x = torch.cat(scale_outputs, dim=1)  # [batch, hidden_dim * scales, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, hidden_dim * scales]
        
        # Fuse scales
        x = self.fusion(x)
        x = self.layer_norm(x)
        
        return x


class DilatedCNN(nn.Module):
    """
    Dilated CNN for capturing long-range patterns without losing resolution.
    Useful for detecting repeated patterns across lines in poetry.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.3
    ):
        """
        Initialize Dilated CNN.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of dilated conv layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            
            in_channels = input_dim if i == 0 else hidden_dim
            
            self.layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels, hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
        
        self.output_dim = hidden_dim
        self.residual_proj = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dilated CNN.
        
        Args:
            x: Input [batch_size, seq_len, input_dim]
            
        Returns:
            Features [batch_size, seq_len, hidden_dim]
        """
        residual = self.residual_proj(x)
        
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        
        for layer in self.layers:
            x = layer(x)
        
        x = x.transpose(1, 2)  # [batch, seq_len, hidden_dim]
        x = x + residual
        
        return x


if __name__ == "__main__":
    # Test CNN Feature Extractor
    batch_size = 4
    seq_len = 50
    input_dim = 256
    
    # Create model
    cnn = CNNFeatureExtractor(
        input_dim=input_dim,
        num_filters=128,
        kernel_sizes=[3, 5, 7]
    )
    
    # Random input (simulating embedded poem)
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    seq_features, pooled_features = cnn(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Sequence features shape: {seq_features.shape}")
    print(f"Pooled features shape: {pooled_features.shape}")
    print(f"Output dimension: {cnn.get_output_dim()}")
    
    # Test Multi-scale CNN
    multi_cnn = MultiScaleCNN(input_dim=input_dim)
    multi_out = multi_cnn(x)
    print(f"\nMulti-scale CNN output shape: {multi_out.shape}")
    
    # Test Dilated CNN
    dilated_cnn = DilatedCNN(input_dim=input_dim)
    dilated_out = dilated_cnn(x)
    print(f"Dilated CNN output shape: {dilated_out.shape}")
