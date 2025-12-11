"""
Poem Learner - Complete Model
Integrates all modules into a unified architecture for poem learning and generation.

Architecture:
1. Input & Preprocessing (embeddings)
2. CNN Feature Extraction (local patterns)
3. Hierarchical RNN (multi-level understanding)
4. Memory & Attention (rote learning simulation)
5. Knowledge Integration (poetic rules)
6. Feedback Loop (iterative refinement)
7. Decoder (generation)

This is the core innovation: combining cognitive-inspired rote learning
with deep neural networks for poetry understanding and generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

from .cnn_module import CNNFeatureExtractor
from .hierarchical_rnn import SimpleHierarchicalRNN
from .memory_attention import MemoryAttentionModule
from .knowledge_base import KnowledgeBase
from .feedback_loop import FeedbackLoop
from .decoder import PoemDecoder


class PoemLearner(nn.Module):
    """
    Complete Poem Learning & Interpretation Model.
    
    Combines all architectural innovations:
    - CNN for local pattern detection (rhythm, rhyme)
    - Hierarchical RNN for multi-level understanding
    - Memory & Attention for rote learning simulation
    - Knowledge base for rule-grounded generation
    - Feedback loop for iterative refinement
    
    Supports:
    - Poem encoding/understanding
    - Poem generation
    - Style transfer
    - Poem completion
    - Rhyme and meter analysis
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_cnn_filters: int = 256,
        cnn_kernel_sizes: List[int] = [3, 5, 7],
        rnn_hidden_size: int = 512,
        rnn_num_layers: int = 2,
        memory_size: int = 256,
        num_memory_cells: int = 16,
        num_attention_heads: int = 8,
        num_feedback_iterations: int = 3,
        num_styles: int = 20,
        dropout: float = 0.3,
        use_knowledge_base: bool = True,
        use_feedback_loop: bool = True,
        max_seq_length: int = 512
    ):
        """
        Initialize Poem Learner.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_cnn_filters: Number of CNN filters
            cnn_kernel_sizes: CNN kernel sizes
            rnn_hidden_size: RNN hidden size
            rnn_num_layers: Number of RNN layers
            memory_size: Size of memory cells
            num_memory_cells: Number of memory cells
            num_attention_heads: Number of attention heads
            num_feedback_iterations: Number of refinement iterations
            num_styles: Number of supported styles
            dropout: Dropout rate
            use_knowledge_base: Whether to use knowledge base
            use_feedback_loop: Whether to use feedback loop
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_knowledge_base = use_knowledge_base
        self.use_feedback_loop = use_feedback_loop
        
        # ============================================
        # Module 1: Embedding Layer
        # ============================================
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.embed_norm = nn.LayerNorm(embedding_dim)
        
        # ============================================
        # Module 2: CNN Feature Extraction
        # ============================================
        self.cnn_extractor = CNNFeatureExtractor(
            input_dim=embedding_dim,
            num_filters=num_cnn_filters,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout,
            use_batch_norm=True,
            use_residual=True
        )
        cnn_output_dim = self.cnn_extractor.get_output_dim()
        
        # ============================================
        # Module 3: Hierarchical RNN
        # ============================================
        self.hierarchical_rnn = SimpleHierarchicalRNN(
            input_dim=cnn_output_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=True,
            dropout=dropout
        )
        rnn_output_dim = self.hierarchical_rnn.get_output_dim()
        
        # ============================================
        # Module 4: Memory & Attention
        # ============================================
        self.memory_attention = MemoryAttentionModule(
            input_dim=rnn_output_dim,
            attention_dim=hidden_dim,
            memory_size=memory_size,
            num_memory_cells=num_memory_cells,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        memory_output_dim = self.memory_attention.get_output_dim()
        
        # ============================================
        # Module 5: Knowledge Base
        # ============================================
        if use_knowledge_base:
            self.knowledge_base = KnowledgeBase(
                embed_dim=memory_output_dim,
                num_styles=num_styles,
                dropout=dropout
            )
        else:
            self.knowledge_base = None
        
        # ============================================
        # Module 6: Feedback Loop
        # ============================================
        if use_feedback_loop:
            self.feedback_loop = FeedbackLoop(
                input_dim=memory_output_dim,
                hidden_dim=hidden_dim,
                num_iterations=num_feedback_iterations,
                dropout=dropout
            )
        else:
            self.feedback_loop = None
        
        # ============================================
        # Module 7: Decoder
        # ============================================
        self.decoder = PoemDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=memory_output_dim,
            num_layers=2,
            dropout=dropout,
            use_attention=True
        )
        
        # ============================================
        # Additional Components
        # ============================================
        
        # Context projection for decoder
        self.context_proj = nn.Linear(memory_output_dim * 2, memory_output_dim)
        
        # Classification heads
        self.style_classifier = nn.Linear(memory_output_dim, num_styles)
        
        # Output projection for encoding task
        self.output_proj = nn.Linear(memory_output_dim, hidden_dim)
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(memory_output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        style_id: Optional[torch.Tensor] = None,
        return_all_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a poem into latent representation.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            style_id: Target style ID [batch_size] (optional)
            return_all_features: Whether to return intermediate features
            
        Returns:
            Dictionary with:
            - encoded: Final encoded representation
            - context: Context vector for generation
            - Additional features if return_all_features=True
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()
        
        # Step 1: Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.embedding(input_ids) + self.pos_embedding(positions)
        embeddings = self.embed_norm(embeddings)
        embeddings = self.embed_dropout(embeddings)
        
        # Step 2: CNN Feature Extraction
        cnn_features, cnn_pooled = self.cnn_extractor(embeddings, attention_mask)
        
        # Step 3: Hierarchical RNN
        rnn_features, poem_repr = self.hierarchical_rnn(cnn_features, attention_mask)
        
        # Step 4: Memory & Attention
        memory_features, context = self.memory_attention(rnn_features, attention_mask)
        
        # Step 5: Knowledge Integration
        if self.use_knowledge_base and self.knowledge_base is not None:
            kb_output = self.knowledge_base(memory_features, style_id)
            features = kb_output['knowledge_features']
        else:
            features = memory_features
        
        # Step 6: Feedback Loop (refinement)
        if self.use_feedback_loop and self.feedback_loop is not None:
            feedback_result = self.feedback_loop(features, mask=attention_mask)
            features = feedback_result['refined']
            quality = feedback_result['quality']
        else:
            quality = None
        
        # Final processing
        features = self.final_norm(features)
        
        # Create context for decoder
        pooled = features.mean(dim=1)
        decoder_context = self.context_proj(torch.cat([context, pooled], dim=-1))
        
        result = {
            'encoded': features,
            'context': decoder_context,
            'poem_representation': pooled
        }
        
        if quality is not None:
            result['quality'] = quality
        
        if return_all_features:
            result.update({
                'embeddings': embeddings,
                'cnn_features': cnn_features,
                'rnn_features': rnn_features,
                'memory_features': memory_features,
                'attention_mask': attention_mask
            })
        
        return result
    
    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        style_id: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            input_ids: Input token IDs [batch_size, src_len]
            target_ids: Target token IDs [batch_size, tgt_len] (for generation)
            attention_mask: Attention mask
            style_id: Style ID for conditioning
            
        Returns:
            Dictionary with loss components and outputs
        """
        # Encode input
        encoded = self.encode(
            input_ids, attention_mask, style_id,
            return_all_features=True
        )
        
        result = {
            'encoded': encoded['encoded'],
            'context': encoded['context'],
            'poem_representation': encoded['poem_representation']
        }
        
        # Decode if target provided (for generation task)
        if target_ids is not None:
            logits, attn_weights = self.decoder(
                encoder_outputs=encoded['encoded'],
                context=encoded['context'],
                target_ids=target_ids,
                encoder_mask=encoded['attention_mask']
            )
            result['logits'] = logits
            result['decoder_attention'] = attn_weights
        
        # Style classification
        style_logits = self.style_classifier(encoded['poem_representation'])
        result['style_logits'] = style_logits
        
        # Quality score if available
        if 'quality' in encoded:
            result['quality'] = encoded['quality']
        
        return result
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        max_length: int = 100,
        start_token_id: int = 1,
        end_token_id: int = 2,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        style_id: Optional[torch.Tensor] = None,
        beam_size: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a poem.
        
        Args:
            input_ids: Optional input tokens for conditioning
            prompt: Optional text prompt (requires tokenizer)
            context: Optional pre-computed context vector
            max_length: Maximum generation length
            start_token_id: Start of sequence token
            end_token_id: End of sequence token
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            style_id: Style to generate in
            beam_size: Beam search size (0 for sampling)
            
        Returns:
            Dictionary with generated tokens and metadata
        """
        # Get encoder outputs and context
        if context is None:
            if input_ids is None:
                raise ValueError("Either input_ids or context must be provided")
            
            encoded = self.encode(input_ids, style_id=style_id)
            encoder_outputs = encoded['encoded']
            context = encoded['context']
            encoder_mask = (input_ids != 0).float()
        else:
            # If only context provided, create dummy encoder outputs
            batch_size = context.size(0)
            encoder_outputs = context.unsqueeze(1)
            encoder_mask = torch.ones(batch_size, 1, device=context.device)
        
        # Generate
        if beam_size > 0:
            results = self.decoder.beam_search(
                encoder_outputs=encoder_outputs,
                context=context,
                start_token_id=start_token_id,
                end_token_id=end_token_id,
                beam_size=beam_size,
                encoder_mask=encoder_mask,
                max_length=max_length
            )
            generated = results[0][0]  # Best beam
            score = results[0][1]
            return {
                'generated_ids': generated,
                'score': score,
                'all_beams': results
            }
        else:
            generated_ids, log_probs = self.decoder.generate(
                encoder_outputs=encoder_outputs,
                context=context,
                start_token_id=start_token_id,
                end_token_id=end_token_id,
                encoder_mask=encoder_mask,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            return {
                'generated_ids': generated_ids,
                'log_probs': log_probs
            }
    
    def get_memorization_metrics(self) -> Dict[str, float]:
        """
        Get metrics related to rote learning/memorization.
        
        Returns:
            Dictionary with memorization metrics
        """
        metrics = {}
        
        # Memory retention score
        if hasattr(self.memory_attention, 'get_retention_score'):
            metrics['retention_score'] = self.memory_attention.get_retention_score()
        
        return metrics
    
    def reset_memory(self):
        """Reset memory modules to initial state."""
        if hasattr(self.memory_attention, 'reset_memory'):
            self.memory_attention.reset_memory()
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_poem_learner(config: dict) -> PoemLearner:
    """
    Create a PoemLearner model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized PoemLearner model
    """
    return PoemLearner(
        vocab_size=config.get('vocab_size', 50000),
        embedding_dim=config.get('embedding', {}).get('embedding_dim', 256),
        hidden_dim=config.get('hidden_dim', 512),
        num_cnn_filters=config.get('cnn', {}).get('num_filters', 256),
        cnn_kernel_sizes=config.get('cnn', {}).get('kernel_sizes', [3, 5, 7]),
        rnn_hidden_size=config.get('hierarchical_rnn', {}).get('line_level', {}).get('hidden_size', 512),
        rnn_num_layers=config.get('hierarchical_rnn', {}).get('line_level', {}).get('num_layers', 2),
        memory_size=config.get('memory_attention', {}).get('memory_size', 256),
        num_memory_cells=config.get('memory_attention', {}).get('num_memory_cells', 16),
        num_attention_heads=config.get('memory_attention', {}).get('attention_heads', 8),
        num_feedback_iterations=config.get('knowledge_base', {}).get('feedback_iterations', 3),
        dropout=config.get('cnn', {}).get('dropout', 0.3),
        use_knowledge_base=config.get('knowledge_base', {}).get('use_knowledge', True),
        use_feedback_loop=True,
        max_seq_length=config.get('data', {}).get('max_seq_length', 512)
    )


if __name__ == "__main__":
    # Test the complete model
    batch_size = 4
    seq_len = 50
    vocab_size = 10000
    
    # Create model
    model = PoemLearner(
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_dim=512
    )
    
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Random input
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(1, vocab_size, (batch_size, 30))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    model.train()
    output = model(input_ids, target_ids, attention_mask)
    
    print(f"\nEncoded shape: {output['encoded'].shape}")
    print(f"Context shape: {output['context'].shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Style logits shape: {output['style_logits'].shape}")
    
    # Test generation
    model.eval()
    with torch.no_grad():
        gen_output = model.generate(
            input_ids=input_ids[:1],
            max_length=20,
            temperature=0.8
        )
    
    print(f"\nGenerated shape: {gen_output['generated_ids'].shape}")
    print(f"Memorization metrics: {model.get_memorization_metrics()}")
