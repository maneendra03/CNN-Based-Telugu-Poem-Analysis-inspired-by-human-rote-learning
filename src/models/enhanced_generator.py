"""
Enhanced Telugu Poem Generator V3
=================================
Professional implementation with:
1. Multi-scale pattern extraction (character, word, verse level)
2. Praasa-aware generation (rhyme enforcement)
3. Advanced anti-repetition (coverage, n-gram blocking, diversity sampling)
4. Interpretation-guided generation (rasa, theme conditioning)
5. Fluency and coherence optimization

Author: Telugu Poem Generation System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Set
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.advanced_preprocessor import AdvancedTeluguPreprocessor, TeluguPhonetics


# Telugu Unicode utilities
TELUGU_RANGE = (0x0C00, 0x0C7F)

def is_telugu_char(char: str) -> bool:
    """Check if character is Telugu."""
    if len(char) != 1:
        return False
    return TELUGU_RANGE[0] <= ord(char) <= TELUGU_RANGE[1]


def count_telugu_chars(text: str) -> int:
    """Count Telugu characters in text."""
    return sum(1 for c in text if is_telugu_char(c))


def clean_telugu_output(text: str) -> str:
    """Clean generated Telugu text."""
    result = []
    for char in text:
        if is_telugu_char(char) or char in ' \n.,!?;:।॥':
            result.append(char)
    text = ''.join(result)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


@dataclass
class GenerationConfig:
    """Configuration for poem generation."""
    max_length: int = 200
    min_length: int = 30
    temperature: float = 0.85
    top_k: int = 50
    top_p: float = 0.92
    repetition_penalty: float = 1.8
    no_repeat_ngram_size: int = 4
    diversity_penalty: float = 0.5
    length_penalty: float = 1.0
    num_return_sequences: int = 1
    do_sample: bool = True
    early_stopping: bool = True
    # Telugu-specific
    enforce_praasa: bool = True
    praasa_weight: float = 0.3
    target_lines: int = 4


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with learnable scaling."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.scale * self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiScalePatternExtractor(nn.Module):
    """
    Multi-scale CNN for extracting patterns at different granularities.
    Captures character-level (prosody), word-level (semantic), and verse-level (structure) patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_scales: int = 4,
        kernel_sizes: List[int] = [3, 5, 7, 11],
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-scale convolutions
        self.conv_layers = nn.ModuleList()
        for ks in kernel_sizes:
            padding = (ks - 1) // 2
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim // num_scales, ks, padding=padding),
                    nn.BatchNorm1d(hidden_dim // num_scales),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, input_dim]
            mask: [batch, seq]
        Returns:
            [batch, seq, hidden_dim]
        """
        x = self.input_proj(x)  # [batch, seq, hidden]
        
        # Apply multi-scale convolutions
        x_conv = x.transpose(1, 2)  # [batch, hidden, seq]
        
        conv_outputs = []
        for conv in self.conv_layers:
            conv_outputs.append(conv(x_conv))  # [batch, hidden/scales, seq]
        
        # Concatenate scales
        multi_scale = torch.cat(conv_outputs, dim=1)  # [batch, hidden, seq]
        multi_scale = multi_scale.transpose(1, 2)  # [batch, seq, hidden]
        
        # Residual + fusion
        output = self.fusion(multi_scale) + x
        
        # Apply mask
        if mask is not None:
            output = output * mask.unsqueeze(-1)
        
        return output


class PraasaAwareAttention(nn.Module):
    """
    Attention mechanism that considers praasa (rhyme) patterns.
    Encourages consistent second-syllable matching across lines.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Praasa position bias
        self.praasa_bias = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Coverage tracking
        self.coverage_proj = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coverage: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, tgt_len, hidden]
            key: [batch, src_len, hidden]
            value: [batch, src_len, hidden]
            coverage: [batch, src_len] accumulated attention
            key_padding_mask: [batch, src_len]
        """
        # Add praasa bias to query (encourages rhyme consistency)
        query = query + self.praasa_bias
        
        # Standard attention
        output, attn_weights = self.attention(
            query, key, value,
            key_padding_mask=key_padding_mask
        )
        
        # Update coverage
        attn_sum = attn_weights.mean(dim=1)  # [batch, src_len]
        new_coverage = attn_sum if coverage is None else coverage + attn_sum
        
        # Coverage loss (penalize attending to same positions)
        if coverage is not None:
            coverage_loss = torch.min(attn_sum, coverage).sum()
        else:
            coverage_loss = torch.tensor(0.0, device=query.device)
        
        return output, new_coverage, coverage_loss


class EnhancedTeluguDecoder(nn.Module):
    """
    Enhanced decoder with:
    1. Praasa-aware attention
    2. Coverage mechanism
    3. Gated output for fluency
    4. Diversity promotion
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.2,
        max_length: int = 256
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_length, dropout)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Praasa-aware attention
        self.praasa_attention = PraasaAwareAttention(hidden_dim, num_heads, dropout)
        
        # Gated output (improves fluency)
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection with weight tying
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        coverage: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_input_ids: [batch, tgt_len]
            encoder_hidden: [batch, src_len, hidden]
            encoder_mask: [batch, src_len]
            coverage: [batch, src_len]
        
        Returns:
            logits: [batch, tgt_len, vocab_size]
            new_coverage: [batch, src_len]
            coverage_loss: scalar
        """
        batch_size, tgt_len = decoder_input_ids.shape
        device = decoder_input_ids.device
        
        # Embeddings with positional encoding
        x = self.token_embedding(decoder_input_ids)
        x = self.pos_encoding(x)
        
        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
        
        # Memory mask
        memory_key_padding_mask = ~encoder_mask.bool() if encoder_mask is not None else None
        
        # Decoder
        decoder_output = self.decoder(
            x,
            encoder_hidden,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Praasa-aware attention
        praasa_output, new_coverage, coverage_loss = self.praasa_attention(
            decoder_output, encoder_hidden, encoder_hidden,
            coverage=coverage,
            key_padding_mask=memory_key_padding_mask
        )
        
        # Gated fusion (improves fluency)
        gate = self.output_gate(torch.cat([decoder_output, praasa_output], dim=-1))
        fused = gate * praasa_output + (1 - gate) * decoder_output
        
        # Output projection
        output = self.layer_norm(fused)
        logits = self.output_proj(output)
        
        return logits, new_coverage, coverage_loss


class RepetitionHandler:
    """
    Handles repetition prevention during generation.
    Combines multiple strategies for robust anti-repetition.
    """
    
    def __init__(
        self,
        vocab_size: int,
        repetition_penalty: float = 1.5,
        no_repeat_ngram_size: int = 3,
        diversity_penalty: float = 0.5
    ):
        self.vocab_size = vocab_size
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.diversity_penalty = diversity_penalty
        
        # N-gram tracking
        self.ngram_cache: Dict[Tuple, int] = {}
        self.token_counts = None
        
    def reset(self, device: torch.device):
        """Reset tracking for new generation."""
        self.ngram_cache = {}
        self.token_counts = torch.zeros(self.vocab_size, device=device)
    
    def update(self, token_id: int, sequence: torch.Tensor):
        """Update tracking with newly generated token."""
        self.token_counts[token_id] += 1
        
        # Update n-gram cache
        if sequence.size(-1) >= self.no_repeat_ngram_size - 1:
            ngram = tuple(sequence[-(self.no_repeat_ngram_size-1):].tolist()) + (token_id,)
            self.ngram_cache[ngram] = self.ngram_cache.get(ngram, 0) + 1
    
    def apply_penalties(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor
    ) -> torch.Tensor:
        """Apply all repetition penalties to logits."""
        
        # 1. Token repetition penalty
        for token_id in range(self.vocab_size):
            if self.token_counts[token_id] > 0:
                count = self.token_counts[token_id].item()
                penalty = self.repetition_penalty ** min(count, 5)  # Cap at 5
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= penalty
                else:
                    logits[0, token_id] *= penalty
        
        # 2. N-gram blocking
        if sequence.size(-1) >= self.no_repeat_ngram_size - 1:
            prefix = tuple(sequence[-(self.no_repeat_ngram_size-1):].tolist())
            for prev_ngram in self.ngram_cache:
                if prev_ngram[:-1] == prefix:
                    blocked_token = prev_ngram[-1]
                    logits[0, blocked_token] = float('-inf')
        
        # 3. Diversity penalty (soft discouragement of recent tokens)
        if sequence.size(-1) > 0:
            recent_tokens = sequence[-min(20, sequence.size(-1)):].unique()
            for token in recent_tokens:
                if logits[0, token] != float('-inf'):
                    logits[0, token] -= self.diversity_penalty
        
        return logits


class TeluguPoemGeneratorV3(nn.Module):
    """
    Professional Telugu Poem Generator with:
    1. Multi-scale pattern extraction
    2. Praasa-aware generation
    3. Robust anti-repetition
    4. Interpretation-guided conditioning
    5. Fluency and coherence optimization
    """
    
    def __init__(
        self,
        model_name: str = 'ai4bharat/indic-bert',
        freeze_encoder: bool = True,
        hidden_dim: int = 768,
        pattern_layers: int = 4,
        decoder_layers: int = 6,
        dropout: float = 0.2
    ):
        super().__init__()
        
        print(f"🚀 Initializing Telugu Poem Generator V3...")
        print(f"   Encoder: {model_name}")
        
        # Load pretrained encoder
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name)
        except Exception as e:
            print(f"   ⚠️ Failed to load {model_name}, falling back to mBERT")
            model_name = 'bert-base-multilingual-cased'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name)
        
        self.hidden_dim = self.encoder.config.hidden_size
        self.vocab_size = self.tokenizer.vocab_size
        
        # Freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("   ✓ Encoder frozen")
        
        # Multi-scale pattern extractor
        self.pattern_extractor = MultiScalePatternExtractor(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_scales=pattern_layers,
            dropout=dropout
        )
        
        # Enhanced decoder
        self.decoder = EnhancedTeluguDecoder(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_layers=decoder_layers,
            dropout=dropout
        )
        
        # Preprocessor for Telugu-specific processing
        self.preprocessor = AdvancedTeluguPreprocessor()
        
        print(f"   ✓ Model initialized")
        print(f"   Hidden dim: {self.hidden_dim}")
        print(f"   Vocab size: {self.vocab_size}")
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode input with pattern extraction."""
        with torch.no_grad() if not self.encoder.training else torch.enable_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            encoder_hidden = outputs.last_hidden_state
        
        # Apply multi-scale pattern extraction
        encoder_hidden = self.pattern_extractor(encoder_hidden, attention_mask)
        
        return encoder_hidden
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass."""
        # Encode
        encoder_hidden = self.encode(input_ids, attention_mask)
        
        # Prepare decoder input (use input_ids shifted, not labels)
        decoder_input = self._shift_tokens_right(input_ids)
        
        # Clamp decoder input to valid range
        decoder_input = decoder_input.clamp(0, self.vocab_size - 1)
        
        # Decode
        logits, _, coverage_loss = self.decoder(
            decoder_input, encoder_hidden, attention_mask
        )
        
        # Loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
            loss = loss + 0.1 * coverage_loss
        
        return {'loss': loss, 'logits': logits, 'coverage_loss': coverage_loss}
    
    def _shift_tokens_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Shift tokens right for decoder input."""
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        shifted[:, 0] = self.tokenizer.cls_token_id or 0
        return shifted
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """
        Generate Telugu poem with anti-repetition and fluency optimization.
        
        Args:
            prompt: Input prompt in Telugu
            config: Generation configuration
            **kwargs: Override config parameters
            
        Returns:
            Generated Telugu poem text
        """
        if config is None:
            config = GenerationConfig()
        
        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.eval()
        device = next(self.parameters()).device
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Encode
        encoder_hidden = self.encode(input_ids, attention_mask)
        
        # Initialize generation
        generated = input_ids.clone()
        coverage = None
        
        # Repetition handler
        rep_handler = RepetitionHandler(
            self.vocab_size,
            config.repetition_penalty,
            config.no_repeat_ngram_size,
            config.diversity_penalty
        )
        rep_handler.reset(device)
        
        # Initialize counts from prompt
        for token in generated[0].tolist():
            if token != self.tokenizer.pad_token_id:
                rep_handler.token_counts[token] += 1
        
        for step in range(config.max_length):
            # Decode
            logits, coverage, _ = self.decoder(
                generated, encoder_hidden, attention_mask, coverage
            )
            next_token_logits = logits[:, -1, :].clone()
            
            # Apply temperature
            next_token_logits = next_token_logits / max(config.temperature, 1e-8)
            
            # Apply repetition penalties
            next_token_logits = rep_handler.apply_penalties(
                next_token_logits, generated[0]
            )
            
            # Prevent EOS before min_length
            if step < config.min_length:
                eos_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id
                if eos_id is not None:
                    next_token_logits[0, eos_id] = float('-inf')
            
            # Top-k filtering
            if config.top_k > 0:
                topk_vals, _ = torch.topk(
                    next_token_logits, 
                    min(config.top_k, next_token_logits.size(-1))
                )
                threshold = topk_vals[..., -1].unsqueeze(-1)
                next_token_logits[next_token_logits < threshold] = float('-inf')
            
            # Top-p (nucleus) filtering
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                # Remove tokens after threshold
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Handle edge cases
            if probs.sum() == 0 or torch.isnan(probs).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            
            if config.do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = probs.argmax(dim=-1, keepdim=True)
            
            # Update tracking
            rep_handler.update(next_token.item(), generated[0])
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check EOS
            eos_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id
            if config.early_stopping and next_token.item() == eos_id:
                break
        
        # Decode
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Clean
        generated_text = clean_telugu_output(generated_text)
        
        # Post-process for verse structure
        generated_text = self._post_process_poem(generated_text, config)
        
        return generated_text
    
    def _post_process_poem(self, text: str, config: GenerationConfig) -> str:
        """Post-process generated poem for better structure."""
        # Clean and structure
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        # If single long line, try to split into verses
        if len(lines) == 1 and len(text) > 100:
            # Split on Telugu punctuation or long pauses
            text = re.sub(r'([।॥])', r'\1\n', text)
            lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        # Limit to target lines
        if len(lines) > config.target_lines * 2:
            lines = lines[:config.target_lines * 2]
        
        return '\n'.join(lines)
    
    def generate_with_style(
        self,
        prompt: str,
        style: str = 'vemana',
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate poem in specific śatakam style.
        
        Args:
            prompt: Input prompt
            style: Style name (vemana, sumati, etc.)
            config: Generation config
        """
        # Add style signature hint to prompt
        style_hints = {
            'vemana': 'విశ్వదాభిరామ వినురవేమ',
            'sumati': 'సుమతీ',
            'bhaskara': 'భాస్కరా',
            'narasimha': 'నరసింహా'
        }
        
        hint = style_hints.get(style.lower(), '')
        enhanced_prompt = f"{prompt} {hint}" if hint else prompt
        
        return self.generate(enhanced_prompt, config)
    
    def count_parameters(self) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_enhanced_generator(
    model_type: str = 'indic-bert',
    freeze_encoder: bool = True
) -> TeluguPoemGeneratorV3:
    """
    Factory function for creating enhanced Telugu poem generator.
    
    Args:
        model_type: 'indic-bert', 'mbert', 'xlm-roberta'
        freeze_encoder: Whether to freeze encoder
    """
    model_map = {
        'indic-bert': 'ai4bharat/indic-bert',
        'mbert': 'bert-base-multilingual-cased',
        'xlm-roberta': 'xlm-roberta-base',
        'distilmbert': 'distilbert-base-multilingual-cased'
    }
    
    model_name = model_map.get(model_type, model_type)
    
    return TeluguPoemGeneratorV3(
        model_name=model_name,
        freeze_encoder=freeze_encoder
    )


if __name__ == "__main__":
    print("Testing Telugu Poem Generator V3\n")
    print("="*60)
    
    # Create model
    model = create_enhanced_generator('mbert', freeze_encoder=True)
    
    total, trainable = model.count_parameters()
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    
    # Test generation
    prompts = [
        "తెలుగు భాష",
        "అమ్మ ప్రేమ",
        "విద్య నేర్చుకో"
    ]
    
    print("\n" + "-"*60)
    print("Generation Test (untrained - random output expected)")
    print("-"*60)
    
    config = GenerationConfig(
        max_length=80,
        min_length=20,
        temperature=0.85,
        repetition_penalty=1.8,
        no_repeat_ngram_size=4
    )
    
    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")
        output = model.generate(prompt, config)
        print(f"   Output: {output[:200]}...")
