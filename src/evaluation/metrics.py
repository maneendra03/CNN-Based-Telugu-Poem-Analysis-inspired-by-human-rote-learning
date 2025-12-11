"""
Evaluation Metrics for Poem Generation
Includes standard NLG metrics and poetry-specific metrics.
"""

import torch
from typing import List, Dict, Optional, Tuple
import re
from collections import Counter
import math


class PoemMetrics:
    """
    Collection of metrics for evaluating poem generation:
    - BLEU score
    - Perplexity
    - Rhyme accuracy
    - Meter consistency
    - Diversity metrics
    - Memorization metrics
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metric accumulators."""
        self.predictions = []
        self.references = []
        self.perplexity_sum = 0.0
        self.perplexity_count = 0
    
    def update(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        logits: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with new predictions.
        
        Args:
            predictions: List of generated poems
            references: List of reference poems (for BLEU)
            logits: Model logits (for perplexity)
            targets: Target token IDs (for perplexity)
        """
        self.predictions.extend(predictions)
        if references:
            self.references.extend(references)
        
        # Update perplexity if logits provided
        if logits is not None and targets is not None:
            ppl = self._compute_batch_perplexity(logits, targets)
            self.perplexity_sum += ppl
            self.perplexity_count += 1
    
    def compute_all(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        # BLEU score
        if self.references:
            metrics['bleu'] = self.compute_bleu()
        
        # Perplexity
        if self.perplexity_count > 0:
            metrics['perplexity'] = self.perplexity_sum / self.perplexity_count
        
        # Rhyme accuracy
        metrics['rhyme_accuracy'] = self.compute_rhyme_accuracy()
        
        # Meter consistency
        metrics['meter_consistency'] = self.compute_meter_consistency()
        
        # Diversity
        diversity = self.compute_diversity()
        metrics.update(diversity)
        
        return metrics
    
    def compute_bleu(self, n_grams: int = 4, smoothing: bool = True) -> float:
        """
        Compute BLEU score.
        
        Args:
            n_grams: Maximum n-gram size
            smoothing: Apply smoothing for short sequences
            
        Returns:
            BLEU score (0-1)
        """
        if not self.references or not self.predictions:
            return 0.0
        
        scores = []
        
        for pred, ref in zip(self.predictions, self.references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            if len(pred_tokens) == 0:
                scores.append(0.0)
                continue
            
            # Compute n-gram precisions
            precisions = []
            for n in range(1, n_grams + 1):
                pred_ngrams = self._get_ngrams(pred_tokens, n)
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                
                if not pred_ngrams:
                    if smoothing:
                        precisions.append(1.0 / len(pred_tokens))
                    else:
                        precisions.append(0.0)
                    continue
                
                matches = sum((pred_ngrams & ref_ngrams).values())
                total = sum(pred_ngrams.values())
                
                if smoothing and matches == 0:
                    precisions.append(1.0 / (total + 1))
                else:
                    precisions.append(matches / total if total > 0 else 0)
            
            # Brevity penalty
            bp = 1.0
            if len(pred_tokens) < len(ref_tokens):
                bp = math.exp(1 - len(ref_tokens) / len(pred_tokens))
            
            # Geometric mean of precisions
            if all(p > 0 for p in precisions):
                log_score = sum(math.log(p) for p in precisions) / len(precisions)
                bleu = bp * math.exp(log_score)
            else:
                bleu = 0.0
            
            scores.append(bleu)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def compute_rhyme_accuracy(self) -> float:
        """
        Compute rhyme accuracy for generated poems.
        Checks if rhyming lines actually rhyme.
        
        Returns:
            Rhyme accuracy (0-1)
        """
        if not self.predictions:
            return 0.0
        
        total_rhymes = 0
        correct_rhymes = 0
        
        for poem in self.predictions:
            lines = [l.strip() for l in poem.split('\n') if l.strip()]
            
            if len(lines) < 2:
                continue
            
            # Check adjacent pairs and alternating pairs
            for i in range(0, len(lines) - 1, 2):
                if i + 1 < len(lines):
                    # Adjacent rhyme (AA pattern)
                    total_rhymes += 1
                    if self._lines_rhyme(lines[i], lines[i + 1]):
                        correct_rhymes += 1
                
                if i + 2 < len(lines):
                    # Alternating rhyme (ABAB pattern)
                    total_rhymes += 1
                    if self._lines_rhyme(lines[i], lines[i + 2]):
                        correct_rhymes += 1
        
        return correct_rhymes / total_rhymes if total_rhymes > 0 else 0.0
    
    def compute_meter_consistency(self) -> float:
        """
        Compute meter/rhythm consistency.
        Measures variance in syllable counts per line.
        
        Returns:
            Meter consistency (0-1, higher is more consistent)
        """
        if not self.predictions:
            return 0.0
        
        consistencies = []
        
        for poem in self.predictions:
            lines = [l.strip() for l in poem.split('\n') if l.strip()]
            
            if len(lines) < 2:
                continue
            
            syllable_counts = [self._count_syllables(line) for line in lines]
            
            if not syllable_counts:
                continue
            
            # Compute coefficient of variation (lower = more consistent)
            mean_syllables = sum(syllable_counts) / len(syllable_counts)
            if mean_syllables == 0:
                continue
                
            variance = sum((s - mean_syllables) ** 2 for s in syllable_counts) / len(syllable_counts)
            std = math.sqrt(variance)
            cv = std / mean_syllables
            
            # Convert to consistency score (0-1)
            consistency = max(0, 1 - cv)
            consistencies.append(consistency)
        
        return sum(consistencies) / len(consistencies) if consistencies else 0.0
    
    def compute_diversity(self) -> Dict[str, float]:
        """
        Compute diversity metrics.
        
        Returns:
            Dictionary with diversity metrics
        """
        if not self.predictions:
            return {'distinct_1': 0.0, 'distinct_2': 0.0, 'vocab_diversity': 0.0}
        
        all_tokens = []
        all_bigrams = []
        
        for poem in self.predictions:
            tokens = poem.lower().split()
            all_tokens.extend(tokens)
            
            for i in range(len(tokens) - 1):
                all_bigrams.append((tokens[i], tokens[i + 1]))
        
        # Distinct-1 (unique unigrams / total unigrams)
        distinct_1 = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
        
        # Distinct-2 (unique bigrams / total bigrams)
        distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0
        
        # Vocabulary diversity (across poems)
        vocab_per_poem = [set(poem.lower().split()) for poem in self.predictions]
        all_vocab = set()
        for v in vocab_per_poem:
            all_vocab.update(v)
        
        avg_vocab_size = sum(len(v) for v in vocab_per_poem) / len(vocab_per_poem)
        vocab_diversity = len(all_vocab) / (avg_vocab_size * len(self.predictions)) if avg_vocab_size > 0 else 0.0
        
        return {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'vocab_diversity': min(1.0, vocab_diversity)
        }
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from token list."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
        return Counter(ngrams)
    
    def _lines_rhyme(self, line1: str, line2: str) -> bool:
        """Check if two lines rhyme."""
        # Extract last words
        words1 = re.findall(r'\b\w+\b', line1.lower())
        words2 = re.findall(r'\b\w+\b', line2.lower())
        
        if not words1 or not words2:
            return False
        
        last1 = words1[-1]
        last2 = words2[-1]
        
        # Check if endings match
        min_len = min(len(last1), len(last2), 3)
        return last1[-min_len:] == last2[-min_len:]
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text."""
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        
        total = 0
        for word in words:
            count = len(re.findall(r'[aeiouy]+', word))
            if word.endswith('e') and count > 1:
                count -= 1
            total += max(1, count)
        
        return total
    
    def _compute_batch_perplexity(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Compute perplexity for a batch."""
        # Flatten
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # Cross entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits_flat, targets_flat, ignore_index=0, reduction='mean'
        )
        
        return torch.exp(loss).item()


class MemorizationCurve:
    """
    Tracks the memorization curve during training.
    Measures how well the model "memorizes" patterns over epochs.
    
    This is a novel metric for the rote learning paradigm.
    """
    
    def __init__(self):
        self.history: List[Dict] = []
    
    def record(
        self,
        epoch: int,
        retention_score: float,
        pattern_recognition: float,
        recall_accuracy: float
    ):
        """
        Record memorization metrics for an epoch.
        
        Args:
            epoch: Current epoch
            retention_score: Memory retention score
            pattern_recognition: How well patterns are recognized
            recall_accuracy: Accuracy of recalling learned content
        """
        self.history.append({
            'epoch': epoch,
            'retention_score': retention_score,
            'pattern_recognition': pattern_recognition,
            'recall_accuracy': recall_accuracy
        })
    
    def get_curve(self) -> Dict[str, List[float]]:
        """Get the memorization curve data."""
        return {
            'epochs': [h['epoch'] for h in self.history],
            'retention': [h['retention_score'] for h in self.history],
            'recognition': [h['pattern_recognition'] for h in self.history],
            'recall': [h['recall_accuracy'] for h in self.history]
        }
    
    def get_learning_efficiency(self) -> float:
        """
        Compute learning efficiency.
        How quickly does the model reach good memorization?
        
        Returns:
            Learning efficiency score
        """
        if len(self.history) < 2:
            return 0.0
        
        # Find epoch where retention reaches 70%
        target_retention = 0.7
        for h in self.history:
            if h['retention_score'] >= target_retention:
                # Earlier epoch = higher efficiency
                return 1.0 - (h['epoch'] / len(self.history))
        
        # Never reached target
        final_retention = self.history[-1]['retention_score']
        return final_retention / target_retention


if __name__ == "__main__":
    # Test metrics
    metrics = PoemMetrics()
    
    # Sample predictions and references
    predictions = [
        "Roses are red\nViolets are blue\nSugar is sweet\nAnd so are you",
        "The sun sets low\nThe moon rises high\nStars start to glow\nAcross the sky"
    ]
    
    references = [
        "Roses are red\nViolets are blue\nHoney is sweet\nAnd so are you",
        "The sun goes down\nThe moon comes up\nStars shine around\nFill up my cup"
    ]
    
    metrics.update(predictions, references)
    
    results = metrics.compute_all()
    
    print("Metrics:")
    for name, value in results.items():
        print(f"  {name}: {value:.4f}")
    
    # Test memorization curve
    curve = MemorizationCurve()
    for epoch in range(10):
        # Simulate improving memorization
        retention = 0.3 + 0.07 * epoch
        recognition = 0.4 + 0.06 * epoch
        recall = 0.2 + 0.08 * epoch
        
        curve.record(epoch, retention, recognition, recall)
    
    print(f"\nMemoization Curve:")
    print(f"  Learning efficiency: {curve.get_learning_efficiency():.4f}")
    print(f"  Final retention: {curve.history[-1]['retention_score']:.4f}")
