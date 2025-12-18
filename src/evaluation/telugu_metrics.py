"""
Telugu Poem Evaluation Metrics
Metrics specifically designed for Telugu poetry evaluation.
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from collections import Counter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.telugu_cleaner import TeluguTextCleaner


class TeluguPoemMetrics:
    """
    Evaluation metrics for Telugu poetry.
    Includes praasa accuracy, meter consistency, and verse structure analysis.
    """
    
    def __init__(self, prosody_rules_path: str = None):
        """Initialize Telugu poem metrics."""
        self.cleaner = TeluguTextCleaner()
        
        # Load prosody rules if available
        if prosody_rules_path is None:
            prosody_rules_path = Path(__file__).parent.parent.parent / "data/knowledge_base/telugu_prosody.json"
        
        try:
            with open(prosody_rules_path, encoding='utf-8') as f:
                self.prosody_rules = json.load(f)
        except:
            self.prosody_rules = {}
        
        # Storage for batch evaluation
        self.predictions = []
        self.references = []
    
    def reset(self):
        """Reset stored predictions and references."""
        self.predictions = []
        self.references = []
    
    def update(self, predictions: List[str], references: List[str] = None):
        """Add predictions and references for batch evaluation."""
        self.predictions.extend(predictions)
        if references:
            self.references.extend(references)
    
    def compute_praasa_accuracy(self, poem: str) -> float:
        """
        Compute praasa (rhyme) accuracy.
        Telugu praasa = second akshara should match across lines.
        
        Args:
            poem: Telugu poem text
            
        Returns:
            Float between 0-1 indicating praasa accuracy
        """
        lines = self.cleaner.split_into_lines(poem)
        
        if len(lines) < 2:
            return 0.0
        
        second_aksharas = []
        for line in lines:
            second = self.cleaner.get_second_letter(line)
            if second:
                second_aksharas.append(second)
        
        if len(second_aksharas) < 2:
            return 0.0
        
        # Count matching pairs
        first_akshara = second_aksharas[0]
        matches = sum(1 for a in second_aksharas if a == first_akshara)
        
        return matches / len(second_aksharas)
    
    def compute_meter_consistency(self, poem: str) -> float:
        """
        Compute meter consistency based on aksharas per line.
        
        Args:
            poem: Telugu poem text
            
        Returns:
            Float between 0-1 indicating meter consistency
        """
        lines = self.cleaner.split_into_lines(poem)
        
        if len(lines) < 2:
            return 0.0
        
        aksharas_per_line = [self.cleaner.count_aksharas(line) for line in lines]
        
        if not aksharas_per_line:
            return 0.0
        
        avg = sum(aksharas_per_line) / len(aksharas_per_line)
        
        if avg == 0:
            return 0.0
        
        # Compute variance
        variance = sum((a - avg) ** 2 for a in aksharas_per_line) / len(aksharas_per_line)
        std_dev = variance ** 0.5
        
        # Consistency = 1 - normalized std dev
        consistency = max(0, 1 - (std_dev / avg)) if avg > 0 else 0
        
        return consistency
    
    def compute_telugu_ratio(self, text: str) -> float:
        """
        Compute ratio of Telugu characters.
        
        Args:
            text: Text to analyze
            
        Returns:
            Float between 0-1 indicating Telugu character ratio
        """
        if not text:
            return 0.0
        
        telugu_chars = sum(1 for c in text if self.cleaner.is_telugu(c))
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        
        return telugu_chars / total_chars if total_chars > 0 else 0.0
    
    def compute_vocabulary_diversity(self, poem: str) -> float:
        """
        Compute vocabulary diversity (unique aksharas / total aksharas).
        
        Args:
            poem: Telugu poem text
            
        Returns:
            Float between 0-1 indicating vocabulary diversity
        """
        aksharas = self.cleaner.extract_aksharas(poem)
        
        if not aksharas:
            return 0.0
        
        unique = len(set(aksharas))
        total = len(aksharas)
        
        return unique / total
    
    def compute_chandassu_match(self, poem: str, expected_chandassu: str = None) -> Dict:
        """
        Check if poem matches expected chandassu (meter type).
        
        Args:
            poem: Telugu poem text
            expected_chandassu: Expected meter type (optional)
            
        Returns:
            Dict with chandassu analysis results
        """
        lines = self.cleaner.split_into_lines(poem)
        aksharas_per_line = [self.cleaner.count_aksharas(line) for line in lines]
        
        avg = sum(aksharas_per_line) / len(aksharas_per_line) if aksharas_per_line else 0
        
        # Detect chandassu based on syllable count
        detected = []
        
        if 18 <= avg <= 22:
            detected.append('ఉత్పలమాల (Utpalamala)')
            detected.append('చంపకమాల (Champakamala)')
        elif 8 <= avg <= 12:
            detected.append('కందం (Kandham)')
            detected.append('ఆటవెలది (Aataveladi)')
        elif 6 <= avg <= 8:
            detected.append('తేటగీతి (Tetagiti)')
        else:
            detected.append('వచన కవిత (Free Verse)')
        
        match = False
        if expected_chandassu and expected_chandassu in detected:
            match = True
        
        return {
            'detected_chandassu': detected,
            'expected_chandassu': expected_chandassu,
            'match': match,
            'aksharas_per_line': aksharas_per_line,
            'avg_aksharas': avg
        }
    
    def compute_all(self) -> Dict:
        """
        Compute all metrics for stored predictions.
        
        Returns:
            Dict with all metric scores
        """
        if not self.predictions:
            return {}
        
        praasa_scores = []
        meter_scores = []
        telugu_ratios = []
        diversity_scores = []
        
        for poem in self.predictions:
            praasa_scores.append(self.compute_praasa_accuracy(poem))
            meter_scores.append(self.compute_meter_consistency(poem))
            telugu_ratios.append(self.compute_telugu_ratio(poem))
            diversity_scores.append(self.compute_vocabulary_diversity(poem))
        
        return {
            'praasa_accuracy': sum(praasa_scores) / len(praasa_scores) if praasa_scores else 0,
            'meter_consistency': sum(meter_scores) / len(meter_scores) if meter_scores else 0,
            'telugu_ratio': sum(telugu_ratios) / len(telugu_ratios) if telugu_ratios else 0,
            'vocabulary_diversity': sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0,
            'num_poems_evaluated': len(self.predictions)
        }
    
    def evaluate_poem(self, poem: str) -> Dict:
        """
        Comprehensive evaluation of a single Telugu poem.
        
        Args:
            poem: Telugu poem text
            
        Returns:
            Dict with all evaluation metrics
        """
        stats = self.cleaner.get_stats(poem)
        
        return {
            # Basic stats
            'num_lines': stats['num_lines'],
            'avg_aksharas_per_line': stats['avg_aksharas_per_line'],
            'telugu_ratio': stats['telugu_ratio'],
            
            # Prosody metrics
            'praasa_accuracy': self.compute_praasa_accuracy(poem),
            'meter_consistency': self.compute_meter_consistency(poem),
            'vocabulary_diversity': self.compute_vocabulary_diversity(poem),
            
            # Praasa details
            'praasa_details': stats['praasa'],
            
            # Chandassu
            'chandassu': self.compute_chandassu_match(poem)
        }


def evaluate_telugu_model(model, test_poems: List[Dict], num_samples: int = 50) -> Dict:
    """
    Evaluate Telugu poem generation model.
    
    Args:
        model: Telugu poem generator model
        test_poems: List of test poem dicts with 'text' field
        num_samples: Number of samples to evaluate
        
    Returns:
        Dict with evaluation results
    """
    metrics = TeluguPoemMetrics()
    
    predictions = []
    references = []
    
    print(f"Evaluating {min(num_samples, len(test_poems))} Telugu poems...")
    
    for i, poem_data in enumerate(test_poems[:num_samples]):
        ref_text = poem_data.get('text', '')
        
        # Get first line as prompt
        lines = ref_text.split('\n')
        if lines:
            prompt = lines[0][:50]
        else:
            continue
        
        # Generate poem
        try:
            generated = model.generate(prompt, max_length=100, temperature=0.8)
            predictions.append(generated)
            references.append(ref_text)
        except Exception as e:
            print(f"Error generating poem {i}: {e}")
            continue
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_samples} poems")
    
    metrics.update(predictions, references)
    results = metrics.compute_all()
    
    print("\n📊 Telugu Evaluation Results:")
    for name, value in results.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
    
    return results


if __name__ == "__main__":
    # Test Telugu metrics
    metrics = TeluguPoemMetrics()
    
    test_poem = """ఉప్పు కప్పురంబు నొక్కపోలికనుండు
చూడ చూడ రుచులు జాడ వేరు
పురుషులందు పుణ్య పురుషులు వేరయా
విశ్వదాభిరామ వినురవేమ"""
    
    print("Testing Telugu Poem Metrics")
    print("=" * 50)
    print(f"Poem:\n{test_poem}")
    print()
    
    results = metrics.evaluate_poem(test_poem)
    
    print("Evaluation Results:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
