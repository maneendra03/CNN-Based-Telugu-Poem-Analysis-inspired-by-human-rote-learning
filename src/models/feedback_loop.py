"""
Feedback Loop Module
Implements the polishing loop for iterative refinement:
1. Generate initial output
2. Compare with ideal style/rules
3. Compute feedback signal
4. Adjust weights and regenerate
5. Repeat until convergence

This is key for the "learning through repetition" paradigm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class FeedbackComparator(nn.Module):
    """
    Compares generated output with target/ideal patterns.
    Provides feedback signal for refinement.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_criteria: int = 5,
        dropout: float = 0.2
    ):
        """
        Initialize Feedback Comparator.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_criteria: Number of evaluation criteria
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_criteria = num_criteria
        
        # Criteria encoders (one per criterion)
        self.criteria_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_criteria)
        ])
        
        # Comparison network
        self.comparator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Overall quality estimator
        self.quality_head = nn.Sequential(
            nn.Linear(num_criteria, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Feedback signal generator
        self.feedback_generator = nn.Sequential(
            nn.Linear(input_dim + num_criteria, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compare generated output with target.
        
        Args:
            generated: Generated features [batch_size, seq_len, input_dim]
            target: Target features [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of:
            - Criteria scores [batch_size, num_criteria]
            - Overall quality [batch_size, 1]
            - Feedback signal [batch_size, seq_len, input_dim]
        """
        # Pool to fixed size
        gen_pooled = generated.mean(dim=1)  # [batch, input_dim]
        tgt_pooled = target.mean(dim=1)
        
        # Evaluate each criterion
        criteria_scores = []
        for encoder in self.criteria_encoders:
            gen_encoded = encoder(gen_pooled)
            tgt_encoded = encoder(tgt_pooled)
            
            combined = torch.cat([gen_encoded, tgt_encoded], dim=-1)
            score = self.comparator(combined)
            criteria_scores.append(score)
        
        criteria_scores = torch.cat(criteria_scores, dim=-1)  # [batch, num_criteria]
        
        # Overall quality
        quality = self.quality_head(criteria_scores)
        
        # Generate feedback signal
        feedback_input = torch.cat(
            [generated, criteria_scores.unsqueeze(1).expand(-1, generated.size(1), -1)],
            dim=-1
        )
        feedback = self.feedback_generator(feedback_input)
        
        return criteria_scores, quality, feedback


class WeightAdjuster(nn.Module):
    """
    Adjusts model weights based on feedback.
    Implements gradient-free refinement.
    """
    
    def __init__(
        self,
        input_dim: int,
        adjustment_rate: float = 0.1,
        momentum: float = 0.9
    ):
        """
        Initialize Weight Adjuster.
        
        Args:
            input_dim: Feature dimension
            adjustment_rate: Learning rate for adjustments
            momentum: Momentum for smoothing adjustments
        """
        super().__init__()
        
        self.adjustment_rate = adjustment_rate
        self.momentum = momentum
        
        # Adjustment predictor
        self.adjustment_net = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Tanh(),  # Bounded adjustments
            nn.Linear(input_dim, input_dim)
        )
        
        # Running average of adjustments
        self.register_buffer('running_adjustment', torch.zeros(input_dim))
    
    def forward(
        self,
        features: torch.Tensor,
        feedback: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply adjustments based on feedback.
        
        Args:
            features: Current features [batch_size, seq_len, input_dim]
            feedback: Feedback signal [batch_size, seq_len, input_dim]
            
        Returns:
            Adjusted features [batch_size, seq_len, input_dim]
        """
        # Compute adjustment
        combined = torch.cat([features, feedback], dim=-1)
        adjustment = self.adjustment_net(combined)
        
        # Scale adjustment
        adjustment = adjustment * self.adjustment_rate
        
        # Apply adjustment
        adjusted = features + adjustment
        
        # Update running adjustment (for monitoring)
        if self.training:
            batch_adjustment = adjustment.mean(dim=(0, 1))
            self.running_adjustment = (
                self.momentum * self.running_adjustment +
                (1 - self.momentum) * batch_adjustment
            )
        
        return adjusted


class FeedbackLoop(nn.Module):
    """
    Complete Feedback Loop for iterative refinement.
    
    The polishing loop:
    1. Take initial features
    2. Compare with knowledge base expectations
    3. Generate feedback signal
    4. Adjust features
    5. Repeat for N iterations
    
    This simulates the human process of refining through practice.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_criteria: int = 5,
        num_iterations: int = 3,
        dropout: float = 0.2,
        convergence_threshold: float = 0.01
    ):
        """
        Initialize Feedback Loop.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_criteria: Number of quality criteria
            num_iterations: Maximum refinement iterations
            dropout: Dropout rate
            convergence_threshold: Stop if improvement below this
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_iterations = num_iterations
        self.convergence_threshold = convergence_threshold
        
        # Feedback comparator
        self.comparator = FeedbackComparator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_criteria=num_criteria,
            dropout=dropout
        )
        
        # Weight adjuster
        self.adjuster = WeightAdjuster(
            input_dim=input_dim,
            adjustment_rate=0.1
        )
        
        # Refinement transformer layer
        self.refine_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Quality predictor (for early stopping)
        self.quality_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Iteration embedding (to encode which iteration we're at)
        self.iteration_embeddings = nn.Embedding(num_iterations, input_dim)
        
        # Layer norms
        self.input_norm = nn.LayerNorm(input_dim)
        self.output_norm = nn.LayerNorm(input_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_history: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through feedback loop.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            target: Target features for comparison (optional)
            mask: Attention mask [batch_size, seq_len]
            return_history: Whether to return refinement history
            
        Returns:
            Dictionary with:
            - refined: Refined features
            - quality: Quality scores per iteration
            - criteria: Criteria scores per iteration
            - num_iterations: Actual iterations used
        """
        batch_size = x.size(0)
        
        # Normalize input
        current = self.input_norm(x)
        
        # If no target, use self as reference (self-improvement)
        if target is None:
            # Create a "cleaner" version through attention
            target, _ = self.refine_attn(current, current, current)
        
        # History tracking
        history = []
        quality_history = []
        criteria_history = []
        
        prev_quality = 0.0
        
        for i in range(self.num_iterations):
            # Get iteration embedding
            iter_idx = torch.full((batch_size,), i, dtype=torch.long, device=x.device)
            iter_emb = self.iteration_embeddings(iter_idx)  # [batch, input_dim]
            
            # Add iteration context
            current_with_iter = current + iter_emb.unsqueeze(1)
            
            # Compare with target
            criteria, quality, feedback = self.comparator(
                current_with_iter, target
            )
            
            # Store history
            if return_history:
                history.append(current.clone())
            quality_history.append(quality.mean().item())
            criteria_history.append(criteria.mean(dim=0).tolist())
            
            # Check for convergence
            current_quality = quality.mean().item()
            if i > 0 and abs(current_quality - prev_quality) < self.convergence_threshold:
                break
            prev_quality = current_quality
            
            # Apply adjustment
            adjusted = self.adjuster(current, feedback)
            
            # Refine through attention
            refined, _ = self.refine_attn(
                adjusted, adjusted, adjusted,
                key_padding_mask=~mask.bool() if mask is not None else None
            )
            
            # Residual connection
            current = current + refined
        
        # Final normalization
        output = self.output_norm(current)
        
        # Final quality assessment
        final_quality = self.quality_predictor(output.mean(dim=1))
        
        result = {
            'refined': output,
            'quality': final_quality,
            'quality_history': quality_history,
            'criteria_history': criteria_history,
            'num_iterations': i + 1
        }
        
        if return_history:
            result['history'] = history
        
        return result
    
    def get_improvement_curve(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> List[float]:
        """
        Get quality improvement curve over iterations.
        Useful for visualizing the polishing process.
        
        Args:
            x: Input features
            target: Target features
            
        Returns:
            List of quality scores per iteration
        """
        with torch.no_grad():
            result = self.forward(x, target, return_history=True)
        return result['quality_history']


class AdaptiveFeedbackLoop(FeedbackLoop):
    """
    Adaptive version that learns optimal number of iterations.
    Uses reinforcement learning-style approach.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Continuation predictor (should we continue refining?)
        self.continue_predictor = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.input_dim),
            nn.GELU(),
            nn.Linear(self.input_dim, 1),
            nn.Sigmoid()
        )
    
    def should_continue(
        self,
        current: torch.Tensor,
        previous: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict whether to continue refining.
        
        Args:
            current: Current features
            previous: Previous iteration features
            
        Returns:
            Continuation probability [batch_size, 1]
        """
        combined = torch.cat([
            current.mean(dim=1),
            previous.mean(dim=1)
        ], dim=-1)
        
        return self.continue_predictor(combined)
    
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        min_iterations: int = 1,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward with adaptive iteration count.
        """
        batch_size = x.size(0)
        current = self.input_norm(x)
        
        if target is None:
            target, _ = self.refine_attn(current, current, current)
        
        quality_history = []
        previous = current.clone()
        
        for i in range(self.num_iterations):
            # Get iteration embedding
            iter_idx = torch.full((batch_size,), i, device=x.device, dtype=torch.long)
            iter_emb = self.iteration_embeddings(iter_idx)
            
            # Compare and get feedback
            criteria, quality, feedback = self.comparator(
                current + iter_emb.unsqueeze(1), target
            )
            quality_history.append(quality.mean().item())
            
            # Check if we should continue
            if i >= min_iterations:
                continue_prob = self.should_continue(current, previous)
                if continue_prob.mean() < 0.5:
                    break
            
            # Save previous state
            previous = current.clone()
            
            # Refine
            adjusted = self.adjuster(current, feedback)
            refined, _ = self.refine_attn(adjusted, adjusted, adjusted)
            current = current + refined
        
        output = self.output_norm(current)
        final_quality = self.quality_predictor(output.mean(dim=1))
        
        return {
            'refined': output,
            'quality': final_quality,
            'quality_history': quality_history,
            'num_iterations': i + 1
        }


if __name__ == "__main__":
    # Test Feedback Loop
    batch_size = 4
    seq_len = 50
    input_dim = 256
    
    # Create module
    feedback_loop = FeedbackLoop(
        input_dim=input_dim,
        hidden_dim=128,
        num_iterations=5
    )
    
    # Random inputs
    x = torch.randn(batch_size, seq_len, input_dim)
    target = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    feedback_loop.eval()
    with torch.no_grad():
        result = feedback_loop(x, target, mask, return_history=True)
    
    print(f"Refined shape: {result['refined'].shape}")
    print(f"Final quality: {result['quality'].mean().item():.4f}")
    print(f"Iterations used: {result['num_iterations']}")
    print(f"Quality history: {result['quality_history']}")
    print(f"Criteria history: {result['criteria_history']}")
    
    # Test adaptive version
    adaptive_loop = AdaptiveFeedbackLoop(
        input_dim=input_dim,
        num_iterations=5
    )
    
    adaptive_loop.eval()
    with torch.no_grad():
        adaptive_result = adaptive_loop(x, target, mask)
    
    print(f"\nAdaptive loop iterations: {adaptive_result['num_iterations']}")
    print(f"Adaptive quality history: {adaptive_result['quality_history']}")
