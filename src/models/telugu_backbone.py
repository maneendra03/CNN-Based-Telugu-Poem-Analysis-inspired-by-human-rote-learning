"""
Telugu Pre-trained Model Backbone
Using IndicBERT or MuRIL for Telugu language poem generation.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Config
)


class TeluguPoemGenerator(nn.Module):
    """
    Telugu poem generator using IndicBERT/MuRIL as backbone.
    Since these are encoder-only models, we add a decoder head.
    """
    
    def __init__(
        self,
        model_name: str = 'ai4bharat/indic-bert',
        hidden_size: int = 768,
        vocab_size: int = 50000,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        
        print(f"Loading Telugu model ({model_name})...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load encoder model
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get actual hidden size from model
        self.hidden_size = self.encoder.config.hidden_size
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Backbone frozen")
        
        # Add decoder layers for generation (reduced for memory)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=self.hidden_size * 2,  # Reduced for memory
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2  # Reduced from 4 to 2 for memory
        )
        
        # Output projection to vocabulary
        vocab_size = self.tokenizer.vocab_size
        self.output_proj = nn.Linear(self.hidden_size, vocab_size)
        
        # Embedding for decoder input
        self.decoder_embedding = nn.Embedding(vocab_size, self.hidden_size)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, self.hidden_size) * 0.02)
        
        print(f"Telugu model loaded! Hidden size: {self.hidden_size}")
    
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Encode input text using IndicBERT."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state
    
    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor = None
    ):
        """Decode using transformer decoder."""
        # Get decoder embeddings
        decoder_embeds = self.decoder_embedding(decoder_input_ids)
        
        # Add positional encoding
        seq_len = decoder_embeds.size(1)
        decoder_embeds = decoder_embeds + self.pos_encoding[:, :seq_len, :]
        
        # Create causal mask for autoregressive generation
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        tgt_mask = tgt_mask.to(decoder_embeds.device)
        
        # Decode
        decoder_output = self.decoder(
            tgt=decoder_embeds,
            memory=encoder_hidden_states,
            tgt_mask=tgt_mask
        )
        
        # Project to vocabulary
        logits = self.output_proj(decoder_output)
        
        return logits
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
        labels: torch.Tensor = None
    ):
        """Forward pass for training."""
        # Encode
        encoder_hidden = self.encode(input_ids, attention_mask)
        
        # If no decoder input, use shifted input
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
        
        # Decode
        logits = self.decode(decoder_input_ids, encoder_hidden, attention_mask)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        
        return {'loss': loss, 'logits': logits}
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """Generate Telugu poem from prompt."""
        self.eval()
        device = next(self.parameters()).device
        
        # Encode prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Get encoder hidden states
        with torch.no_grad():
            encoder_hidden = self.encode(input_ids, attention_mask)
        
        # Start with prompt tokens as decoder input
        generated = input_ids.clone()
        
        # Generate tokens autoregressively
        for _ in range(max_length):
            with torch.no_grad():
                logits = self.decode(generated, encoder_hidden, attention_mask)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        return generated_text


class TeluguMuRILGenerator(nn.Module):
    """
    Telugu poem generator using Google's MuRIL model.
    MuRIL is specifically trained on Indian languages including Telugu.
    """
    
    def __init__(
        self,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        model_name = 'google/muril-base-cased'
        print(f"Loading MuRIL model for Telugu...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        self.hidden_size = self.encoder.config.hidden_size
        
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Decoder layers
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=12,
                dim_feedforward=self.hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )
        
        vocab_size = self.tokenizer.vocab_size
        self.output_proj = nn.Linear(self.hidden_size, vocab_size)
        self.decoder_embedding = nn.Embedding(vocab_size, self.hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, self.hidden_size) * 0.02)
        
        print(f"MuRIL Telugu model loaded!")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass."""
        encoder_output = self.encoder(input_ids, attention_mask).last_hidden_state
        
        decoder_embeds = self.decoder_embedding(input_ids)
        seq_len = decoder_embeds.size(1)
        decoder_embeds = decoder_embeds + self.pos_encoding[:, :seq_len, :]
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(decoder_embeds.device)
        
        decoder_output = self.decoder(decoder_embeds, encoder_output, tgt_mask=tgt_mask)
        logits = self.output_proj(decoder_output)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        
        return {'loss': loss, 'logits': logits}
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.8) -> str:
        """Generate Telugu text."""
        self.eval()
        device = next(self.parameters()).device
        
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            encoder_hidden = self.encoder(input_ids, attention_mask).last_hidden_state
        
        generated = input_ids.clone()
        
        for _ in range(max_length):
            with torch.no_grad():
                decoder_embeds = self.decoder_embedding(generated)
                seq_len = decoder_embeds.size(1)
                decoder_embeds = decoder_embeds + self.pos_encoding[:, :seq_len, :]
                
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                decoder_output = self.decoder(decoder_embeds, encoder_hidden, tgt_mask=tgt_mask)
                logits = self.output_proj(decoder_output)
                
                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == self.tokenizer.sep_token_id:
                    break
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)


def create_telugu_generator(model_type: str = 'distilmbert', freeze_backbone: bool = True):
    """
    Factory function to create Telugu poem generator.
    
    Args:
        model_type: 'distilmbert', 'mbert', 'xlm-roberta', or 'muril'
        freeze_backbone: Whether to freeze pre-trained weights
    
    Returns:
        Telugu poem generator model
        
    Available Models (PUBLIC - no login required):
        - distilmbert: DistilBERT Multilingual (Recommended for 8GB RAM)
        - mbert: Multilingual BERT (110M params - needs 8GB+)
        - xlm-roberta: XLM-RoBERTa (700M params - needs 16GB+)
    
    Gated Models (require HuggingFace login):
        - indicbert: IndicBERT (requires login at huggingface.co)
        - muril: MuRIL (requires login)
    """
    if model_type == 'distilmbert':
        # PUBLIC - No login required, SMALL (134M params), Telugu support
        return TeluguPoemGenerator(
            model_name='distilbert-base-multilingual-cased',
            freeze_backbone=freeze_backbone
        )
    elif model_type == 'mbert':
        # PUBLIC - No login required, 110M params
        return TeluguPoemGenerator(
            model_name='bert-base-multilingual-cased',
            freeze_backbone=freeze_backbone
        )
    elif model_type == 'xlm-roberta':
        # PUBLIC - 700M params, needs 16GB+ RAM
        return TeluguPoemGenerator(
            model_name='xlm-roberta-base',
            freeze_backbone=freeze_backbone
        )
    elif model_type == 'indicbert':
        # GATED - Requires HuggingFace login
        print("⚠️ IndicBERT requires HuggingFace login!")
        print("   Run: huggingface-cli login")
        print("   Or use: model_type='xlm-roberta' (public)")
        return TeluguPoemGenerator(
            model_name='ai4bharat/indic-bert',
            freeze_backbone=freeze_backbone
        )
    elif model_type == 'muril':
        return TeluguMuRILGenerator(freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'xlm-roberta', 'mbert', 'indicbert', or 'muril'")


if __name__ == "__main__":
    # Test Telugu model
    print("Testing Telugu Poem Generator...")
    
    model = create_telugu_generator('indicbert', freeze_backbone=True)
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    
    # Test generation
    prompt = "చందమామ రావే"  # "Come, Moon" - famous Telugu poem
    print(f"\nPrompt: {prompt}")
    
    # Note: Generation will be random without training
    generated = model.generate(prompt, max_length=30)
    print(f"Generated: {generated}")
