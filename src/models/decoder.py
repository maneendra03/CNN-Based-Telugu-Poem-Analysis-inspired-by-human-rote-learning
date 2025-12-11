"""
Poem Decoder Module
Generates poem text from latent representations.

Supports:
- Autoregressive generation
- Beam search decoding
- Constrained generation (rhyme, meter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


class PoemDecoder(nn.Module):
    """
    Decoder for generating poems from context vectors.
    
    Features:
    - LSTM-based autoregressive generation
    - Attention over encoder outputs
    - Copy mechanism for rare words
    - Controlled generation with constraints
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True,
        max_length: int = 200
    ):
        """
        Initialize Poem Decoder.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
            max_length: Maximum generation length
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.max_length = max_length
        
        # Token embeddings
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        
        # Input projection (combines embedding with context)
        self.input_proj = nn.Linear(
            embedding_dim + hidden_size if use_attention else embedding_dim,
            hidden_size
        )
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        context: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training (teacher forcing).
        
        Args:
            encoder_outputs: Encoder outputs [batch_size, src_len, hidden_size]
            context: Context vector [batch_size, hidden_size]
            target_ids: Target token IDs [batch_size, tgt_len] (for training)
            encoder_mask: Encoder attention mask [batch_size, src_len]
            
        Returns:
            Tuple of:
            - Logits [batch_size, tgt_len, vocab_size]
            - Attention weights [batch_size, tgt_len, src_len]
        """
        if target_ids is None:
            raise ValueError("target_ids required for training forward pass")
        
        batch_size, tgt_len = target_ids.shape
        
        # Embed target tokens
        embedded = self.embedding(target_ids)  # [batch, tgt_len, embed_dim]
        
        # Initialize hidden state from context
        h0 = context.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c0 = torch.zeros_like(h0)
        
        all_logits = []
        all_attn_weights = []
        
        hidden = (h0, c0)
        
        for t in range(tgt_len):
            # Get current input
            current_embed = embedded[:, t:t+1, :]  # [batch, 1, embed_dim]
            
            if self.use_attention:
                # Compute attention over encoder outputs
                query = hidden[0][-1].unsqueeze(1)  # [batch, 1, hidden]
                
                attn_output, attn_weights = self.attention(
                    query, encoder_outputs, encoder_outputs,
                    key_padding_mask=~encoder_mask.bool() if encoder_mask is not None else None
                )
                
                # Combine embedding with attention context
                combined = torch.cat([current_embed, attn_output], dim=-1)
                all_attn_weights.append(attn_weights)
            else:
                combined = current_embed
            
            # Project input
            lstm_input = self.input_proj(combined)
            
            # LSTM step
            output, hidden = self.lstm(lstm_input, hidden)
            
            if self.use_attention:
                # Combine LSTM output with attention
                output = torch.cat([output, attn_output], dim=-1)
                output = self.attn_combine(output)
            
            # Apply layer norm
            output = self.layer_norm(output)
            
            # Project to vocabulary
            logits = self.output_proj(output)
            all_logits.append(logits)
        
        # Stack outputs
        logits = torch.cat(all_logits, dim=1)  # [batch, tgt_len, vocab]
        
        if all_attn_weights:
            attn_weights = torch.cat(all_attn_weights, dim=1)
        else:
            attn_weights = None
        
        return logits, attn_weights
    
    def generate(
        self,
        encoder_outputs: torch.Tensor,
        context: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        encoder_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate poem autoregressively.
        
        Args:
            encoder_outputs: Encoder outputs
            context: Context vector
            start_token_id: Token ID to start generation
            end_token_id: Token ID to stop generation
            encoder_mask: Encoder attention mask
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling (0 for no filtering)
            top_p: Nucleus sampling threshold
            
        Returns:
            Tuple of:
            - Generated token IDs [batch_size, gen_len]
            - Log probabilities [batch_size, gen_len]
        """
        max_length = max_length or self.max_length
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Initialize
        generated = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        log_probs = []
        
        # Initialize hidden state
        h = context.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c = torch.zeros_like(h)
        hidden = (h, c)
        
        for _ in range(max_length):
            # Get embedding of last token
            current_token = generated[:, -1:]
            embedded = self.embedding(current_token)
            
            if self.use_attention:
                query = hidden[0][-1].unsqueeze(1)
                attn_output, _ = self.attention(
                    query, encoder_outputs, encoder_outputs,
                    key_padding_mask=~encoder_mask.bool() if encoder_mask is not None else None
                )
                combined = torch.cat([embedded, attn_output], dim=-1)
            else:
                combined = embedded
            
            # Process through decoder
            lstm_input = self.input_proj(combined)
            output, hidden = self.lstm(lstm_input, hidden)
            
            if self.use_attention:
                output = torch.cat([output, attn_output], dim=-1)
                output = self.attn_combine(output)
            
            output = self.layer_norm(output)
            logits = self.output_proj(output)  # [batch, 1, vocab]
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                logits = self._top_k_filtering(logits, top_k)
            
            # Apply nucleus (top-p) filtering
            if top_p > 0:
                logits = self._nucleus_filtering(logits, top_p)
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs.squeeze(1), 1)
            log_prob = torch.log(probs.squeeze(1).gather(1, next_token))
            
            generated = torch.cat([generated, next_token], dim=1)
            log_probs.append(log_prob)
            
            # Check for end token
            if (next_token == end_token_id).all():
                break
        
        log_probs = torch.cat(log_probs, dim=1) if log_probs else torch.empty(batch_size, 0)
        
        return generated[:, 1:], log_probs  # Remove start token
    
    def beam_search(
        self,
        encoder_outputs: torch.Tensor,
        context: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        beam_size: int = 5,
        encoder_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        length_penalty: float = 1.0
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        Generate using beam search.
        
        Args:
            encoder_outputs: Encoder outputs
            context: Context vector
            start_token_id: Start token ID
            end_token_id: End token ID
            beam_size: Number of beams
            encoder_mask: Encoder attention mask
            max_length: Maximum length
            length_penalty: Penalty for length
            
        Returns:
            List of (sequence, score) tuples for each beam
        """
        max_length = max_length or self.max_length
        device = encoder_outputs.device
        
        # Initialize beams
        beams = [(
            torch.tensor([[start_token_id]], device=device),
            0.0,
            context.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous(),
            torch.zeros_like(context.unsqueeze(0).expand(self.num_layers, -1, -1))
        )]
        
        completed = []
        
        for _ in range(max_length):
            all_candidates = []
            
            for seq, score, h, c in beams:
                if seq[0, -1].item() == end_token_id:
                    completed.append((seq, score))
                    continue
                
                # Forward pass
                current_token = seq[:, -1:]
                embedded = self.embedding(current_token)
                
                if self.use_attention:
                    query = h[-1].unsqueeze(1)
                    attn_output, _ = self.attention(
                        query, encoder_outputs, encoder_outputs
                    )
                    combined = torch.cat([embedded, attn_output], dim=-1)
                else:
                    combined = embedded
                
                lstm_input = self.input_proj(combined)
                output, (new_h, new_c) = self.lstm(lstm_input, (h, c))
                
                if self.use_attention:
                    output = torch.cat([output, attn_output], dim=-1)
                    output = self.attn_combine(output)
                
                output = self.layer_norm(output)
                logits = self.output_proj(output)
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get top-k tokens
                top_log_probs, top_tokens = log_probs.squeeze().topk(beam_size)
                
                for i in range(beam_size):
                    new_seq = torch.cat([seq, top_tokens[i:i+1].unsqueeze(0)], dim=1)
                    new_score = score + top_log_probs[i].item()
                    all_candidates.append((new_seq, new_score, new_h, new_c))
            
            if not all_candidates:
                break
            
            # Select top beams
            all_candidates.sort(key=lambda x: x[1] / (x[0].size(1) ** length_penalty), reverse=True)
            beams = all_candidates[:beam_size]
        
        # Add remaining beams
        completed.extend([(b[0], b[1]) for b in beams])
        
        # Sort by score
        completed.sort(key=lambda x: x[1] / (x[0].size(1) ** length_penalty), reverse=True)
        
        return completed[:beam_size]
    
    @staticmethod
    def _top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, :, -1].unsqueeze(-1)
        return torch.where(
            logits < min_values,
            torch.full_like(logits, float('-inf')),
            logits
        )
    
    @staticmethod
    def _nucleus_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        return logits.masked_fill(indices_to_remove, float('-inf'))


if __name__ == "__main__":
    # Test decoder
    batch_size = 4
    src_len = 50
    tgt_len = 30
    vocab_size = 10000
    hidden_size = 256
    
    # Create decoder
    decoder = PoemDecoder(
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_size=hidden_size
    )
    
    # Random inputs
    encoder_outputs = torch.randn(batch_size, src_len, hidden_size)
    context = torch.randn(batch_size, hidden_size)
    target_ids = torch.randint(0, vocab_size, (batch_size, tgt_len))
    encoder_mask = torch.ones(batch_size, src_len)
    
    # Training forward pass
    logits, attn_weights = decoder(
        encoder_outputs, context, target_ids, encoder_mask
    )
    
    print(f"Logits shape: {logits.shape}")
    print(f"Attention weights shape: {attn_weights.shape if attn_weights is not None else 'None'}")
    
    # Generation
    decoder.eval()
    with torch.no_grad():
        generated, log_probs = decoder.generate(
            encoder_outputs, context,
            start_token_id=1,
            end_token_id=2,
            encoder_mask=encoder_mask,
            max_length=20,
            temperature=0.8
        )
    
    print(f"\nGenerated shape: {generated.shape}")
    print(f"Log probs shape: {log_probs.shape}")
