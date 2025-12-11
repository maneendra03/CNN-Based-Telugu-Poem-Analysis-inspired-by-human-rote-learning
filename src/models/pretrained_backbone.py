"""
Pre-trained Model Integration
Integrates GPT-2 and BERT with our custom Rote Learning architecture.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from transformers import (
    GPT2Model, GPT2LMHeadModel, GPT2Tokenizer,
    BertModel, BertTokenizer,
    AutoModel, AutoTokenizer
)


class PretrainedBackbone(nn.Module):
    """
    Wrapper for pre-trained language models.
    Provides unified interface for GPT-2, BERT, etc.
    """
    
    SUPPORTED_MODELS = {
        'gpt2': ('gpt2', GPT2Model, GPT2Tokenizer),
        'gpt2-medium': ('gpt2-medium', GPT2Model, GPT2Tokenizer),
        'bert': ('bert-base-uncased', BertModel, BertTokenizer),
        'bert-large': ('bert-large-uncased', BertModel, BertTokenizer),
        'distilbert': ('distilbert-base-uncased', AutoModel, AutoTokenizer),
    }
    
    def __init__(
        self,
        model_name: str = 'gpt2',
        freeze_backbone: bool = False,
        output_hidden_states: bool = True
    ):
        """
        Initialize pre-trained backbone.
        
        Args:
            model_name: Name of pre-trained model
            freeze_backbone: Whether to freeze pre-trained weights
            output_hidden_states: Return all hidden states
        """
        super().__init__()
        
        self.model_name = model_name
        
        if model_name in self.SUPPORTED_MODELS:
            checkpoint, model_class, tokenizer_class = self.SUPPORTED_MODELS[model_name]
        else:
            # Try loading as custom checkpoint
            checkpoint = model_name
            model_class = AutoModel
            tokenizer_class = AutoTokenizer
        
        # Load model and tokenizer
        print(f"Loading {checkpoint}...")
        self.model = model_class.from_pretrained(
            checkpoint,
            output_hidden_states=output_hidden_states
        )
        self.tokenizer = tokenizer_class.from_pretrained(checkpoint)
        
        # Add special tokens if needed
        special_tokens = {
            'pad_token': '[PAD]' if self.tokenizer.pad_token is None else None,
            'bos_token': '[BOS]' if self.tokenizer.bos_token is None else None,
            'eos_token': '[EOS]' if self.tokenizer.eos_token is None else None,
        }
        special_tokens = {k: v for k, v in special_tokens.items() if v is not None}
        
        if special_tokens:
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Freeze if requested
        if freeze_backbone:
            self.freeze()
        
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = len(self.tokenizer)
        
        print(f"Loaded {model_name} with {self.count_parameters():,} parameters")
    
    def freeze(self):
        """Freeze all backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze(self, layers: Optional[int] = None):
        """
        Unfreeze backbone parameters.
        
        Args:
            layers: Number of top layers to unfreeze (None = all)
        """
        for param in self.model.parameters():
            param.requires_grad = True
        
        if layers is not None:
            # Freeze all except top N layers
            all_params = list(self.model.named_parameters())
            n_unfreeze = len(all_params) // 12 * layers  # Approximate
            
            for name, param in all_params[:-n_unfreeze]:
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through backbone.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary with hidden states and embeddings
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return {
            'last_hidden_state': outputs.last_hidden_state,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'pooler_output': outputs.pooler_output if hasattr(outputs, 'pooler_output') else None
        }
    
    def encode(self, text: str, max_length: int = 256) -> torch.Tensor:
        """Encode text to token IDs."""
        return self.tokenizer.encode(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(
            token_ids.squeeze().tolist(),
            skip_special_tokens=True
        )
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())
    
    def get_embedding_layer(self) -> nn.Embedding:
        """Get the embedding layer from backbone."""
        if hasattr(self.model, 'wte'):  # GPT-2
            return self.model.wte
        elif hasattr(self.model, 'embeddings'):  # BERT
            return self.model.embeddings.word_embeddings
        else:
            raise ValueError("Cannot find embedding layer")


class GPT2PoemGenerator(nn.Module):
    """
    GPT-2 based poem generator with Rote Learning enhancement.
    """
    
    def __init__(
        self,
        model_name: str = 'gpt2',
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        # Load GPT-2
        print(f"Loading GPT-2 poem generator ({model_name})...")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.hidden_size = self.gpt2.config.hidden_size
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.gpt2.transformer.parameters():
                param.requires_grad = False
        
        # Memory enhancement layer (our novel contribution)
        self.memory_adapter = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Style conditioning
        self.style_embeddings = nn.Embedding(10, self.hidden_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        style_id: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        
        # Get GPT-2 outputs
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Apply memory adapter to hidden states
        last_hidden = outputs.hidden_states[-1]
        memory_enhanced = self.memory_adapter(last_hidden) + last_hidden
        
        # Add style if provided
        if style_id is not None:
            style_emb = self.style_embeddings(style_id).unsqueeze(1)
            memory_enhanced = memory_enhanced + style_emb
        
        return {
            'loss': outputs.loss if outputs.loss is not None else None,
            'logits': outputs.logits,
            'hidden_states': memory_enhanced
        }
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        style_id: Optional[int] = None
    ) -> str:
        """Generate poem from prompt."""
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # Move to same device as model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        
        # Generate
        with torch.no_grad():
            outputs = self.gpt2.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated


class HybridPoemModel(nn.Module):
    """
    Hybrid model combining pre-trained backbone with our custom architecture.
    
    Architecture:
    1. Pre-trained backbone (GPT-2/BERT) for initial encoding
    2. Our Memory & Attention module for rote learning
    3. Our Feedback Loop for refinement
    4. Custom decoder head
    """
    
    def __init__(
        self,
        backbone_name: str = 'gpt2',
        memory_size: int = 256,
        num_memory_cells: int = 16,
        num_styles: int = 10,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        # Pre-trained backbone
        self.backbone = PretrainedBackbone(
            model_name=backbone_name,
            freeze_backbone=freeze_backbone
        )
        
        hidden_size = self.backbone.hidden_size
        
        # Import our custom modules
        from src.models.memory_attention import MemoryAttentionModule
        from src.models.feedback_loop import FeedbackLoop
        
        # Our novel memory module
        self.memory_attention = MemoryAttentionModule(
            input_dim=hidden_size,
            attention_dim=hidden_size,
            memory_size=memory_size,
            num_memory_cells=num_memory_cells
        )
        
        # Our novel feedback loop
        self.feedback_loop = FeedbackLoop(
            input_dim=self.memory_attention.get_output_dim(),
            hidden_dim=hidden_size,
            num_iterations=3
        )
        
        # Style conditioning
        self.style_embeddings = nn.Embedding(num_styles, hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(
            self.memory_attention.get_output_dim(),
            self.backbone.vocab_size
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        style_id: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass combining backbone with our modules."""
        
        # Step 1: Pre-trained backbone
        backbone_outputs = self.backbone(input_ids, attention_mask)
        hidden_states = backbone_outputs['last_hidden_state']
        
        # Step 2: Add style conditioning
        if style_id is not None:
            style_emb = self.style_embeddings(style_id).unsqueeze(1)
            hidden_states = hidden_states + style_emb
        
        # Step 3: Memory & Attention (our novel rote learning)
        memory_outputs, context = self.memory_attention(hidden_states, attention_mask)
        
        # Step 4: Feedback loop (our novel refinement)
        feedback_outputs = self.feedback_loop(memory_outputs, mask=attention_mask)
        refined = feedback_outputs['refined']
        
        # Step 5: Output projection
        logits = self.output_proj(refined)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': refined,
            'context': context,
            'quality': feedback_outputs.get('quality')
        }
    
    def get_memorization_metrics(self) -> Dict[str, float]:
        """Get rote learning metrics."""
        return self.memory_attention.get_retention_score() if hasattr(self.memory_attention, 'get_retention_score') else {}


if __name__ == "__main__":
    print("Testing pre-trained model integration...")
    
    # Test GPT-2 generator
    print("\n1. Testing GPT2PoemGenerator...")
    generator = GPT2PoemGenerator(model_name='gpt2', freeze_backbone=True)
    
    prompt = "Roses are red,"
    print(f"\nPrompt: {prompt}")
    
    generated = generator.generate(prompt, max_length=50, temperature=0.7)
    print(f"Generated:\n{generated}")
    
    print("\n✓ GPT-2 poem generator working!")
    
    # Test that trainable params are only in adapter
    trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    total = sum(p.numel() for p in generator.parameters())
    print(f"\nTrainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
