"""
Telugu Poem Interpretation Module
=================================
Provides semantic understanding of Telugu poetry through:
1. Prosodic feature extraction (praasa, yati, chandas)
2. Emotional tone analysis (rasa)
3. Thematic classification
4. Structural pattern recognition
5. Meaning representation for generation feedback

Author: Telugu Poem Generation System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import re
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.advanced_preprocessor import (
    AdvancedTeluguPreprocessor,
    TeluguPhonetics,
    EMOTION_KEYWORDS,
    SATAKAM_SIGNATURES,
    GANA_PATTERNS
)


# Nine Rasas (emotional essences) in Sanskrit/Telugu aesthetics
NAVARASA = {
    'శృంగారం': {  # Shringara - Love/Romance
        'keywords': ['ప్రేమ', 'ప్రియ', 'కాంత', 'వల్లభ', 'మనసు', 'హృదయ', 'అనురాగ', 'ముద్దు'],
        'english': 'love'
    },
    'హాస్యం': {  # Hasya - Comedy/Humor
        'keywords': ['నవ్వు', 'హాస్య', 'చమత్కార', 'వినోద', 'సరస'],
        'english': 'humor'
    },
    'కరుణ': {  # Karuna - Compassion/Pathos
        'keywords': ['దుఃఖ', 'బాధ', 'కన్నీరు', 'శోక', 'వేదన', 'దయ', 'కరుణ'],
        'english': 'compassion'
    },
    'రౌద్రం': {  # Raudra - Anger/Fury
        'keywords': ['కోపం', 'క్రోధం', 'ఆగ్రహం', 'రౌద్ర', 'ఉగ్ర'],
        'english': 'anger'
    },
    'వీరం': {  # Veera - Heroism/Courage
        'keywords': ['వీర', 'శౌర్య', 'ధైర్య', 'పరాక్రమ', 'విజయ', 'యుద్ధ'],
        'english': 'heroism'
    },
    'భయానకం': {  # Bhayanaka - Fear/Terror
        'keywords': ['భయ', 'భీతి', 'తల్లడ', 'భయానక'],
        'english': 'fear'
    },
    'బీభత్సం': {  # Bibhatsa - Disgust/Aversion
        'keywords': ['జుగుప్స', 'అసహ్య', 'ద్వేష'],
        'english': 'disgust'
    },
    'అద్భుతం': {  # Adbhuta - Wonder/Amazement
        'keywords': ['అద్భుత', 'ఆశ్చర్య', 'విస్మయ', 'మహిమ'],
        'english': 'wonder'
    },
    'శాంతం': {  # Shanta - Peace/Tranquility
        'keywords': ['శాంతి', 'ప్రశాంత', 'నిర్మల', 'స్థిర', 'మోక్ష', 'ముక్తి'],
        'english': 'peace'
    }
}

# Thematic categories for Telugu poetry
THEMES = {
    'నీతి': {  # Moral/Ethics
        'keywords': ['నీతి', 'ధర్మ', 'సత్య', 'న్యాయ', 'మంచి', 'చెడు'],
        'english': 'morality'
    },
    'భక్తి': {  # Devotion
        'keywords': ['భక్తి', 'దేవ', 'ప్రార్థన', 'పూజ', 'స్తుతి', 'నమస్కార', 'శరణ'],
        'english': 'devotion'
    },
    'ప్రకృతి': {  # Nature
        'keywords': ['ప్రకృతి', 'వన', 'పూవు', 'నది', 'కొండ', 'చంద్ర', 'సూర్య', 'వాన'],
        'english': 'nature'
    },
    'జ్ఞానం': {  # Knowledge/Wisdom
        'keywords': ['జ్ఞాన', 'విద్య', 'బుద్ధి', 'తెలివి', 'పండిత', 'చదువు'],
        'english': 'wisdom'
    },
    'వైరాగ్యం': {  # Detachment
        'keywords': ['వైరాగ్య', 'త్యాగ', 'సన్యాస', 'మాయ', 'మోహ'],
        'english': 'detachment'
    },
    'సమాజం': {  # Society
        'keywords': ['సమాజ', 'లోక', 'ప్రజ', 'జన', 'మానవ'],
        'english': 'society'
    }
}


class ProsodyAnalyzer:
    """
    Analyzes prosodic features of Telugu poetry.
    """
    
    def __init__(self):
        self.preprocessor = AdvancedTeluguPreprocessor()
    
    def analyze_praasa(self, poem: str) -> Dict:
        """
        Analyze praasa (rhyme) - second akshara matching.
        """
        return self.preprocessor.analyze_praasa(poem)
    
    def analyze_yati(self, line: str) -> Dict:
        """
        Analyze yati (caesura/pause point) in a line.
        
        Yati divides the line into two parts with a pause.
        Common patterns: 4+4, 4+5, 5+5, etc.
        """
        aksharas = self.preprocessor.extract_aksharas(line)
        total = len(aksharas)
        
        if total < 6:
            return {'has_yati': False, 'position': None}
        
        # Find potential yati points (usually middle)
        mid = total // 2
        potential_positions = [mid - 1, mid, mid + 1]
        
        return {
            'has_yati': True,
            'total_aksharas': total,
            'potential_yati_positions': potential_positions,
            'first_half': ''.join(aksharas[:mid]),
            'second_half': ''.join(aksharas[mid:])
        }
    
    def analyze_chandas(self, poem: str) -> Dict:
        """
        Analyze chandas (meter) of entire poem.
        """
        lines = [l.strip() for l in poem.split('\n') if l.strip()]
        
        line_analyses = []
        for line in lines:
            analysis = self.preprocessor.identify_chandassu(line)
            line_analyses.append(analysis)
        
        # Check consistency
        if not line_analyses:
            return {'consistent': False, 'analyses': []}
        
        # Find dominant meter
        meter_counts = defaultdict(int)
        for analysis in line_analyses:
            for meter in analysis['possible_meters']:
                meter_counts[meter] += 1
        
        dominant = max(meter_counts.items(), key=lambda x: x[1])[0] if meter_counts else None
        
        return {
            'consistent': len(set(a['akshara_count'] for a in line_analyses)) <= 2,
            'dominant_meter': dominant,
            'line_analyses': line_analyses,
            'meter_distribution': dict(meter_counts)
        }
    
    def get_prosody_features(self, poem: str) -> Dict:
        """Get all prosodic features."""
        lines = [l.strip() for l in poem.split('\n') if l.strip()]
        
        return {
            'praasa': self.analyze_praasa(poem),
            'chandas': self.analyze_chandas(poem),
            'yati': [self.analyze_yati(line) for line in lines],
            'line_count': len(lines),
            'total_aksharas': self.preprocessor.count_aksharas(poem)
        }


class RasaAnalyzer:
    """
    Analyzes emotional essence (rasa) in Telugu poetry.
    Based on Natyashastra's Navarasa theory.
    """
    
    def __init__(self):
        self.preprocessor = AdvancedTeluguPreprocessor()
    
    def detect_rasa(self, poem: str) -> Dict[str, float]:
        """
        Detect dominant rasas in a poem.
        
        Returns:
            Dict mapping rasa names to confidence scores
        """
        scores = {}
        
        for rasa_name, rasa_info in NAVARASA.items():
            keywords = rasa_info['keywords']
            count = sum(1 for kw in keywords if kw in poem)
            # Weighted by keyword significance
            scores[rasa_name] = min(1.0, count / max(1, len(keywords) * 0.3))
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def get_dominant_rasa(self, poem: str) -> Tuple[str, float]:
        """Get the most dominant rasa."""
        scores = self.detect_rasa(poem)
        if not scores:
            return ('శాంతం', 0.0)  # Default to peace/neutral
        
        dominant = max(scores.items(), key=lambda x: x[1])
        return dominant
    
    def get_rasa_embedding(self, poem: str, dim: int = 9) -> List[float]:
        """
        Get a fixed-size embedding representing rasa distribution.
        
        Args:
            poem: Telugu poem text
            dim: Embedding dimension (default 9 for Navarasa)
            
        Returns:
            List of floats representing rasa scores
        """
        scores = self.detect_rasa(poem)
        
        # Create ordered embedding
        rasa_order = list(NAVARASA.keys())
        embedding = [scores.get(rasa, 0.0) for rasa in rasa_order]
        
        return embedding


class ThemeClassifier:
    """
    Classifies thematic content of Telugu poems.
    """
    
    def __init__(self):
        self.preprocessor = AdvancedTeluguPreprocessor()
    
    def detect_themes(self, poem: str) -> Dict[str, float]:
        """
        Detect themes in a poem.
        
        Returns:
            Dict mapping theme names to confidence scores
        """
        scores = {}
        
        for theme_name, theme_info in THEMES.items():
            keywords = theme_info['keywords']
            count = sum(1 for kw in keywords if kw in poem)
            scores[theme_name] = min(1.0, count / max(1, len(keywords) * 0.4))
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def get_primary_themes(self, poem: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k primary themes."""
        scores = self.detect_themes(poem)
        sorted_themes = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_themes[:top_k]


class TeluguPoemInterpreter:
    """
    Comprehensive poem interpretation combining all analysis modules.
    Provides consistent semantic representation for generation feedback.
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """
        Initialize interpreter with all analyzers.
        
        Args:
            knowledge_base_path: Path to knowledge base directory
        """
        self.preprocessor = AdvancedTeluguPreprocessor(knowledge_base_path)
        self.prosody_analyzer = ProsodyAnalyzer()
        self.rasa_analyzer = RasaAnalyzer()
        self.theme_classifier = ThemeClassifier()
    
    def interpret(self, poem: str) -> Dict:
        """
        Comprehensive interpretation of a Telugu poem.
        
        Args:
            poem: Telugu poem text
            
        Returns:
            Complete interpretation dictionary
        """
        # Clean the poem first
        cleaned = self.preprocessor.clean_text(poem)
        
        # Get all analyses
        prosody = self.prosody_analyzer.get_prosody_features(cleaned)
        rasa_scores = self.rasa_analyzer.detect_rasa(cleaned)
        dominant_rasa = self.rasa_analyzer.get_dominant_rasa(cleaned)
        themes = self.theme_classifier.detect_themes(cleaned)
        primary_themes = self.theme_classifier.get_primary_themes(cleaned)
        
        # Identify śatakam
        satakam = self.preprocessor.identify_satakam(cleaned)
        
        # Compute quality metrics
        quality = self._compute_quality_metrics(prosody)
        
        return {
            'text': cleaned,
            'prosody': prosody,
            'rasa': {
                'scores': rasa_scores,
                'dominant': dominant_rasa[0],
                'confidence': dominant_rasa[1]
            },
            'themes': {
                'scores': themes,
                'primary': primary_themes
            },
            'satakam': satakam,
            'quality': quality,
            'embedding': self.get_interpretation_embedding(cleaned)
        }
    
    def _compute_quality_metrics(self, prosody: Dict) -> Dict:
        """Compute quality metrics from prosody analysis."""
        metrics = {
            'praasa_score': prosody['praasa']['match_ratio'] if prosody['praasa']['has_praasa'] else 0.0,
            'meter_consistency': 1.0 if prosody['chandas']['consistent'] else 0.5,
            'structural_score': min(1.0, prosody['line_count'] / 4)  # Ideal: 4 lines
        }
        
        # Overall quality
        metrics['overall'] = (
            0.4 * metrics['praasa_score'] +
            0.3 * metrics['meter_consistency'] +
            0.3 * metrics['structural_score']
        )
        
        return metrics
    
    def get_interpretation_embedding(self, poem: str, dim: int = 32) -> List[float]:
        """
        Get fixed-size embedding representing poem interpretation.
        
        Combines:
        - Rasa embedding (9 dims)
        - Theme embedding (6 dims)
        - Prosody features (8 dims)
        - Quality metrics (4 dims)
        - Padding (5 dims)
        
        Total: 32 dims
        """
        embedding = []
        
        # Rasa (9)
        rasa_emb = self.rasa_analyzer.get_rasa_embedding(poem)
        embedding.extend(rasa_emb)
        
        # Themes (6)
        themes = self.theme_classifier.detect_themes(poem)
        theme_order = list(THEMES.keys())
        theme_emb = [themes.get(t, 0.0) for t in theme_order]
        embedding.extend(theme_emb)
        
        # Prosody features (8)
        prosody = self.prosody_analyzer.get_prosody_features(poem)
        prosody_emb = [
            prosody['praasa']['match_ratio'],
            1.0 if prosody['chandas']['consistent'] else 0.0,
            min(1.0, prosody['line_count'] / 8),
            min(1.0, prosody['total_aksharas'] / 100),
            # Yati presence
            sum(1 for y in prosody['yati'] if y.get('has_yati', False)) / max(1, len(prosody['yati'])),
            # Meter strength
            len(prosody['chandas'].get('meter_distribution', {})) / 5,
            # Reserved
            0.0,
            0.0
        ]
        embedding.extend(prosody_emb)
        
        # Quality (4)
        quality = self._compute_quality_metrics(prosody)
        quality_emb = [
            quality['praasa_score'],
            quality['meter_consistency'],
            quality['structural_score'],
            quality['overall']
        ]
        embedding.extend(quality_emb)
        
        # Padding (5)
        embedding.extend([0.0] * 5)
        
        return embedding[:dim]  # Ensure exact dimension
    
    def compare_poems(self, poem1: str, poem2: str) -> Dict:
        """
        Compare two poems for similarity.
        
        Returns:
            Dict with similarity scores
        """
        interp1 = self.interpret(poem1)
        interp2 = self.interpret(poem2)
        
        # Rasa similarity (cosine)
        rasa1 = torch.tensor(self.rasa_analyzer.get_rasa_embedding(poem1))
        rasa2 = torch.tensor(self.rasa_analyzer.get_rasa_embedding(poem2))
        rasa_sim = F.cosine_similarity(rasa1.unsqueeze(0), rasa2.unsqueeze(0)).item()
        
        # Theme similarity
        theme1 = torch.tensor([interp1['themes']['scores'].get(t, 0) for t in THEMES])
        theme2 = torch.tensor([interp2['themes']['scores'].get(t, 0) for t in THEMES])
        theme_sim = F.cosine_similarity(theme1.unsqueeze(0), theme2.unsqueeze(0)).item()
        
        # Structural similarity
        struct_sim = 1.0 - abs(
            interp1['prosody']['line_count'] - interp2['prosody']['line_count']
        ) / max(interp1['prosody']['line_count'], interp2['prosody']['line_count'], 1)
        
        # Same śatakam?
        same_satakam = interp1['satakam'] == interp2['satakam'] and interp1['satakam'] is not None
        
        return {
            'rasa_similarity': rasa_sim,
            'theme_similarity': theme_sim,
            'structural_similarity': struct_sim,
            'same_satakam': same_satakam,
            'overall_similarity': (0.3 * rasa_sim + 0.3 * theme_sim + 0.2 * struct_sim + 0.2 * float(same_satakam))
        }
    
    def get_generation_feedback(self, generated: str, reference: str = None) -> Dict:
        """
        Get feedback for generation improvement.
        
        Args:
            generated: Generated poem text
            reference: Optional reference poem for comparison
            
        Returns:
            Feedback dictionary with scores and suggestions
        """
        interp = self.interpret(generated)
        
        feedback = {
            'quality': interp['quality'],
            'issues': [],
            'suggestions': []
        }
        
        # Check praasa
        if not interp['prosody']['praasa']['has_praasa']:
            feedback['issues'].append('Missing or inconsistent praasa (rhyme)')
            feedback['suggestions'].append('Ensure second akshara matches across lines')
        
        # Check meter
        if not interp['prosody']['chandas']['consistent']:
            feedback['issues'].append('Inconsistent meter across lines')
            feedback['suggestions'].append('Maintain similar syllable count per line')
        
        # Check structure
        if interp['prosody']['line_count'] < 2:
            feedback['issues'].append('Too few lines for a complete verse')
            feedback['suggestions'].append('Add more lines to complete the verse structure')
        
        # Check emotional coherence
        if interp['rasa']['confidence'] < 0.1:
            feedback['issues'].append('Weak emotional expression')
            feedback['suggestions'].append('Add emotion-specific vocabulary')
        
        # Compare with reference if provided
        if reference:
            comparison = self.compare_poems(generated, reference)
            feedback['comparison'] = comparison
            
            if comparison['overall_similarity'] < 0.5:
                feedback['suggestions'].append('Generated poem differs significantly from reference style')
        
        return feedback


# Neural interpretation model (optional, for training)
class NeuralPoemInterpreter(nn.Module):
    """
    Neural network for learning poem interpretations.
    Can be trained to predict prosody, rasa, and themes.
    """
    
    def __init__(
        self,
        input_dim: int = 768,  # From encoder
        hidden_dim: int = 256,
        num_rasa: int = 9,
        num_themes: int = 6,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Rasa classifier
        self.rasa_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_rasa)
        )
        
        # Theme classifier
        self.theme_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_themes)
        )
        
        # Quality predictor
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, input_dim] encoded poem representation
            
        Returns:
            Dict with rasa_logits, theme_logits, quality_score
        """
        encoded = self.encoder(x)
        
        return {
            'rasa_logits': self.rasa_head(encoded),
            'theme_logits': self.theme_head(encoded),
            'quality_score': self.quality_head(encoded).squeeze(-1)
        }
    
    def get_interpretation(self, x: torch.Tensor) -> Dict:
        """Get interpretation probabilities."""
        with torch.no_grad():
            outputs = self.forward(x)
            
            return {
                'rasa_probs': F.softmax(outputs['rasa_logits'], dim=-1),
                'theme_probs': torch.sigmoid(outputs['theme_logits']),  # Multi-label
                'quality_score': outputs['quality_score']
            }


if __name__ == "__main__":
    # Test the interpretation module
    test_poem = """ఉప్పు కప్పురంబు నొక్కపోలికనుండు
చూడ చూడ రుచులు జాడ వేరు
పురుషులందు పుణ్య పురుషులు వేరయా
విశ్వదాభిరామ వినురవేమ"""
    
    print("Testing Telugu Poem Interpreter\n")
    print("="*60)
    
    interpreter = TeluguPoemInterpreter()
    
    # Full interpretation
    interpretation = interpreter.interpret(test_poem)
    
    print(f"Poem:\n{test_poem}\n")
    print("-"*60)
    print(f"Śatakam: {interpretation['satakam']}")
    print(f"Dominant Rasa: {interpretation['rasa']['dominant']} ({interpretation['rasa']['confidence']:.2f})")
    print(f"Primary Themes: {interpretation['themes']['primary']}")
    print(f"\nProsody:")
    print(f"  Lines: {interpretation['prosody']['line_count']}")
    print(f"  Praasa: {interpretation['prosody']['praasa']['has_praasa']}")
    print(f"  Praasa Akshara: {interpretation['prosody']['praasa']['praasa_akshara']}")
    print(f"  Meter Consistent: {interpretation['prosody']['chandas']['consistent']}")
    print(f"\nQuality Metrics:")
    for metric, score in interpretation['quality'].items():
        print(f"  {metric}: {score:.2f}")
    print(f"\nInterpretation Embedding (32-dim): {interpretation['embedding'][:8]}...")
