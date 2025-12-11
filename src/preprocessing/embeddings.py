"""
Embedding Layer Module
Handles word embeddings using BERT or Word2Vec for poem representation.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
import numpy as np


class EmbeddingLayer(nn.Module):
    """
    Flexible embedding layer supporting:
    - Pre-trained BERT embeddings
    - Word2Vec embeddings
    - Learnable embeddings from scratch
    
    Provides both word-level and character-level embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 768,
        embedding_type: Literal["bert", "word2vec", "learnable"] = "learnable",
        pretrained_path: Optional[str] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0,
        dropout: float = 0.1,
        char_vocab_size: Optional[int] = None,
        char_embedding_dim: int = 64
    ):
        """
        Initialize embedding layer.
        
        Args:
            vocab_size: Size of word vocabulary
            embedding_dim: Dimension of word embeddings
            embedding_type: Type of embeddings ("bert", "word2vec", "learnable")
            pretrained_path: Path to pretrained embeddings (for word2vec)
            freeze_embeddings: Whether to freeze pretrained embeddings
            padding_idx: Index used for padding token
            dropout: Dropout rate
            char_vocab_size: Size of character vocabulary (if using char embeddings)
            char_embedding_dim: Dimension of character embeddings
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_type = embedding_type
        self.freeze_embeddings = freeze_embeddings
        self.padding_idx = padding_idx
        
        # Word embeddings
        if embedding_type == "bert":
            self._init_bert_embeddings()
        elif embedding_type == "word2vec":
            self._init_word2vec_embeddings(pretrained_path)
        else:
            self._init_learnable_embeddings()
        
        # Character embeddings (always learnable)
        self.char_embeddings = None
        if char_vocab_size is not None:
            self.char_embeddings = nn.Embedding(
                num_embeddings=char_vocab_size,
                embedding_dim=char_embedding_dim,
                padding_idx=padding_idx
            )
            self.char_embedding_dim = char_embedding_dim
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def _init_bert_embeddings(self):
        """Initialize BERT-based embeddings."""
        try:
            from transformers import BertModel, BertTokenizer
            
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            if self.freeze_embeddings:
                for param in self.bert_model.parameters():
                    param.requires_grad = False
            
            # BERT embedding dimension is 768
            self.embedding_dim = 768
            
            # Projection layer if needed
            self.projection = None
            
        except ImportError:
            print("Warning: transformers not installed. Falling back to learnable embeddings.")
            self._init_learnable_embeddings()
            self.embedding_type = "learnable"
    
    def _init_word2vec_embeddings(self, pretrained_path: Optional[str]):
        """Initialize Word2Vec embeddings."""
        if pretrained_path:
            try:
                # Load pretrained vectors
                vectors = self._load_word2vec(pretrained_path)
                self.word_embeddings = nn.Embedding.from_pretrained(
                    vectors,
                    freeze=self.freeze_embeddings,
                    padding_idx=self.padding_idx
                )
                self.embedding_dim = vectors.shape[1]
            except Exception as e:
                print(f"Warning: Could not load Word2Vec from {pretrained_path}: {e}")
                print("Falling back to learnable embeddings.")
                self._init_learnable_embeddings()
        else:
            # Use gensim's pretrained if available
            self._init_learnable_embeddings()
    
    def _load_word2vec(self, path: str) -> torch.Tensor:
        """Load Word2Vec vectors from file."""
        # This is a placeholder - implement based on your vector format
        # Could be .bin, .txt, or .npy format
        raise NotImplementedError("Implement based on your Word2Vec format")
    
    def _init_learnable_embeddings(self):
        """Initialize learnable embeddings from scratch."""
        self.word_embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx
        )
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.word_embeddings.weight)
        
        # Zero out padding embedding
        with torch.no_grad():
            self.word_embeddings.weight[self.padding_idx].fill_(0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for embeddings.
        
        Args:
            input_ids: Word token IDs [batch_size, seq_len]
            char_ids: Character IDs [batch_size, seq_len, char_len] (optional)
            attention_mask: Attention mask for BERT (optional)
            
        Returns:
            Embedded representations [batch_size, seq_len, embedding_dim]
        """
        if self.embedding_type == "bert" and hasattr(self, 'bert_model'):
            # Use BERT embeddings
            outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            embeddings = outputs.last_hidden_state
        else:
            # Use standard embeddings
            embeddings = self.word_embeddings(input_ids)
        
        # Apply layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def embed_chars(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Get character-level embeddings.
        
        Args:
            char_ids: Character IDs [batch_size, seq_len] or [batch_size, num_words, char_len]
            
        Returns:
            Character embeddings
        """
        if self.char_embeddings is None:
            raise ValueError("Character embeddings not initialized. "
                           "Set char_vocab_size in constructor.")
        
        return self.char_embeddings(char_ids)
    
    def get_output_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.embedding_dim
    
    def get_char_output_dim(self) -> int:
        """Get the character embedding dimension."""
        if self.char_embeddings is None:
            return 0
        return self.char_embedding_dim


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-style models.
    Adds position information to embeddings.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        max_seq_length: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize positional encoding.
        
        Args:
            embedding_dim: Dimension of embeddings
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-np.log(10000.0) / embedding_dim)
        )
        
        pe = torch.zeros(max_seq_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input embeddings [batch_size, seq_len, embedding_dim]
            
        Returns:
            Embeddings with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CombinedEmbedding(nn.Module):
    """
    Combined word and character embeddings for richer representation.
    Concatenates or adds word and character-derived embeddings.
    """
    
    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        word_embedding_dim: int = 300,
        char_embedding_dim: int = 64,
        char_hidden_dim: int = 128,
        combine_method: Literal["concat", "add"] = "concat",
        dropout: float = 0.1
    ):
        """
        Initialize combined embeddings.
        
        Args:
            word_vocab_size: Size of word vocabulary
            char_vocab_size: Size of character vocabulary
            word_embedding_dim: Dimension of word embeddings
            char_embedding_dim: Dimension of character embeddings
            char_hidden_dim: Hidden dimension for char CNN/RNN
            combine_method: How to combine word and char embeddings
            dropout: Dropout rate
        """
        super().__init__()
        
        self.combine_method = combine_method
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(
            word_vocab_size, 
            word_embedding_dim,
            padding_idx=0
        )
        
        # Character embeddings with CNN
        self.char_embeddings = nn.Embedding(
            char_vocab_size,
            char_embedding_dim,
            padding_idx=0
        )
        
        # CNN to process character embeddings
        self.char_cnn = nn.Conv1d(
            in_channels=char_embedding_dim,
            out_channels=char_hidden_dim,
            kernel_size=3,
            padding=1
        )
        
        # Compute output dimension
        if combine_method == "concat":
            self.output_dim = word_embedding_dim + char_hidden_dim
        else:  # add
            assert word_embedding_dim == char_hidden_dim, \
                "For 'add' method, word and char dimensions must match"
            self.output_dim = word_embedding_dim
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass combining word and character embeddings.
        
        Args:
            word_ids: Word token IDs [batch_size, seq_len]
            char_ids: Character IDs [batch_size, seq_len, max_char_len]
            
        Returns:
            Combined embeddings [batch_size, seq_len, output_dim]
        """
        # Word embeddings
        word_emb = self.word_embeddings(word_ids)  # [B, S, W_dim]
        
        # Character embeddings
        batch_size, seq_len, max_char_len = char_ids.shape
        
        # Reshape for processing
        char_ids_flat = char_ids.view(-1, max_char_len)  # [B*S, C]
        char_emb = self.char_embeddings(char_ids_flat)   # [B*S, C, C_dim]
        
        # Apply CNN (transpose for conv1d)
        char_emb = char_emb.transpose(1, 2)  # [B*S, C_dim, C]
        char_emb = self.char_cnn(char_emb)    # [B*S, H_dim, C]
        
        # Max pool over characters
        char_emb, _ = char_emb.max(dim=2)  # [B*S, H_dim]
        
        # Reshape back
        char_emb = char_emb.view(batch_size, seq_len, -1)  # [B, S, H_dim]
        
        # Combine
        if self.combine_method == "concat":
            combined = torch.cat([word_emb, char_emb], dim=-1)
        else:
            combined = word_emb + char_emb
        
        # Normalize and dropout
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        
        return combined
    
    def get_output_dim(self) -> int:
        """Get output embedding dimension."""
        return self.output_dim


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    seq_len = 20
    vocab_size = 1000
    char_vocab_size = 100
    
    # Test learnable embeddings
    embedding_layer = EmbeddingLayer(
        vocab_size=vocab_size,
        embedding_dim=256,
        embedding_type="learnable",
        char_vocab_size=char_vocab_size,
        char_embedding_dim=64
    )
    
    # Random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    embeddings = embedding_layer(input_ids)
    print(f"Word embeddings shape: {embeddings.shape}")
    
    # Test character embeddings
    char_ids = torch.randint(0, char_vocab_size, (batch_size, seq_len))
    char_emb = embedding_layer.embed_chars(char_ids)
    print(f"Char embeddings shape: {char_emb.shape}")
    
    # Test combined embeddings
    combined = CombinedEmbedding(
        word_vocab_size=vocab_size,
        char_vocab_size=char_vocab_size
    )
    
    char_ids_3d = torch.randint(0, char_vocab_size, (batch_size, seq_len, 15))
    combined_emb = combined(input_ids, char_ids_3d)
    print(f"Combined embeddings shape: {combined_emb.shape}")
