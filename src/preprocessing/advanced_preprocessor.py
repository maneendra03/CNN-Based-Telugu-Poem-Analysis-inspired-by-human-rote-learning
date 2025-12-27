"""
Advanced Telugu Poem Preprocessor
=================================
Comprehensive preprocessing for Telugu poetry with:
1. Robust text cleaning and normalization
2. Prosody-aware tokenization
3. Semantic feature extraction
4. Consistent representation for meaning, emotion, and structure

Author: Telugu Poem Generation System
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import json
from pathlib import Path


# Telugu Unicode Constants
TELUGU_RANGE = (0x0C00, 0x0C7F)

# Telugu phonetic categories
TELUGU_VOWELS = set('అఆఇఈఉఊఋౠఎఏఐఒఓఔ')
TELUGU_VOWEL_SIGNS = set('ాిీుూృౄెేైొోౌ')
TELUGU_CONSONANTS = set('కఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరలవశషసహళక్షఱ')
TELUGU_DIGITS = set('౦౧౨౩౪౫౬౭౮౯')
TELUGU_SPECIAL = {'ం': 'anusvara', 'ః': 'visarga', 'ఁ': 'chandrabindu', '్': 'virama'}

# Gana (prosodic unit) patterns: U = guru (long), | = laghu (short)
GANA_PATTERNS = {
    'ya': '|UU',    # యగణం
    'ra': 'U|U',    # రగణం  
    'ta': 'UU|',    # తగణం
    'bha': 'U||',   # భగణం
    'ja': '|U|',    # జగణం
    'sa': '||U',    # సగణం
    'ma': 'UUU',    # మగణం
    'na': '|||',    # నగణం
}

# Common emotion keywords in Telugu poetry
EMOTION_KEYWORDS = {
    'love': ['ప్రేమ', 'ప్రియ', 'మనసు', 'హృదయ', 'అనురాగ', 'వల్లభ', 'కాంత'],
    'devotion': ['భక్తి', 'దైవ', 'ప్రార్థన', 'స్తుతి', 'నమస్కార', 'సేవ', 'శరణ'],
    'nature': ['ప్రకృతి', 'పూవు', 'వాన', 'నది', 'కొండ', 'చంద్ర', 'సూర్య', 'చెట్టు'],
    'wisdom': ['జ్ఞాన', 'విద్య', 'బుద్ధి', 'తెలివి', 'పండిత', 'నీతి', 'సత్య'],
    'sorrow': ['దుఃఖ', 'బాధ', 'కష్ట', 'విరహ', 'శోక', 'కన్నీరు', 'వేదన'],
    'joy': ['ఆనంద', 'సంతోష', 'హర్ష', 'సుఖ', 'ఉత్సాహ', 'ముద'],
}

# Common Śatakam signatures (ending patterns)
SATAKAM_SIGNATURES = {
    'vemana': 'విశ్వదాభిరామ వినురవేమ',
    'sumati': 'సుమతీ',
    'bhadragiri': 'భద్రగిరీశ',
    'bhaskara': 'భాస్కరా',
    'kumara': 'కుమారా',
    'narasimha': 'నరసింహా',
    'kalahasti': 'కాళహస్తీశ్వరా',
}


class TeluguPhonetics:
    """Telugu phonetic analysis utilities."""
    
    @staticmethod
    def is_telugu_char(char: str) -> bool:
        """Check if character is in Telugu Unicode range."""
        if not char or len(char) != 1:
            return False
        return TELUGU_RANGE[0] <= ord(char) <= TELUGU_RANGE[1]
    
    @staticmethod
    def is_vowel(char: str) -> bool:
        """Check if character is a vowel."""
        return char in TELUGU_VOWELS
    
    @staticmethod
    def is_vowel_sign(char: str) -> bool:
        """Check if character is a vowel sign (maatra)."""
        return char in TELUGU_VOWEL_SIGNS
    
    @staticmethod
    def is_consonant(char: str) -> bool:
        """Check if character is a consonant."""
        return char in TELUGU_CONSONANTS
    
    @staticmethod
    def is_virama(char: str) -> bool:
        """Check if character is virama (halant)."""
        return char == '్'
    
    @staticmethod
    def get_syllable_weight(syllable: str) -> str:
        """
        Determine if syllable is guru (U - heavy) or laghu (| - light).
        
        Rules:
        - Vowel with long sound (ఆ, ఈ, ఊ, ఏ, ఐ, ఓ, ఔ) = guru
        - Vowel followed by anusvara or visarga = guru
        - Consonant cluster = guru
        - Short vowel alone = laghu
        """
        if not syllable:
            return '|'
        
        # Check for long vowel signs
        long_signs = set('ాీూేైోౌ')
        if any(c in long_signs for c in syllable):
            return 'U'
        
        # Check for anusvara or visarga
        if 'ం' in syllable or 'ః' in syllable:
            return 'U'
        
        # Check for virama (consonant cluster indicator)
        if '్' in syllable:
            return 'U'
        
        # Check for standalone long vowels
        long_vowels = set('ఆఈఊఏఐఓఔ')
        if any(c in long_vowels for c in syllable):
            return 'U'
        
        return '|'  # Default: laghu


class AdvancedTeluguPreprocessor:
    """
    Advanced preprocessor for Telugu poetry with comprehensive features.
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """
        Initialize preprocessor with optional knowledge base.
        
        Args:
            knowledge_base_path: Path to knowledge base directory
        """
        self.phonetics = TeluguPhonetics()
        
        # Load knowledge base if available
        self.prosody_rules = {}
        self.grammar_rules = {}
        
        if knowledge_base_path:
            kb_path = Path(knowledge_base_path)
            self._load_knowledge_base(kb_path)
    
    def _load_knowledge_base(self, kb_path: Path):
        """Load knowledge base files."""
        try:
            prosody_file = kb_path / 'telugu_prosody.json'
            if prosody_file.exists():
                with open(prosody_file, 'r', encoding='utf-8') as f:
                    self.prosody_rules = json.load(f)
            
            grammar_file = kb_path / 'grammar_rules.json'
            if grammar_file.exists():
                with open(grammar_file, 'r', encoding='utf-8') as f:
                    self.grammar_rules = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load knowledge base: {e}")
    
    # =========================================================================
    # TEXT CLEANING & NORMALIZATION
    # =========================================================================
    
    def clean_text(self, text: str, preserve_structure: bool = True) -> str:
        """
        Clean and normalize Telugu text.
        
        Args:
            text: Raw Telugu text
            preserve_structure: Keep line breaks and verse structure
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Unicode normalization (NFC for composed characters)
        text = unicodedata.normalize('NFC', text)
        
        # Remove invisible characters and zero-width spaces
        text = re.sub(r'[\u200b-\u200f\u2028-\u202f\ufeff]', '', text)
        
        # Normalize various dash types
        text = re.sub(r'[–—−]', '-', text)
        
        # Normalize quotation marks
        text = re.sub(r'[""„‟]', '"', text)
        text = re.sub(r"[''‚‛]", "'", text)
        
        # Keep only Telugu chars and essential punctuation
        cleaned = []
        for char in text:
            if self.phonetics.is_telugu_char(char):
                cleaned.append(char)
            elif char in ' \t':
                cleaned.append(' ')
            elif char in '\n\r' and preserve_structure:
                cleaned.append('\n')
            elif char in '.,!?;:।॥-':
                cleaned.append(char)
        
        text = ''.join(cleaned)
        
        # Normalize whitespace
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' *\n *', '\n', text)
        
        return text.strip()
    
    def normalize_for_training(self, text: str) -> str:
        """
        Normalize text for training (more aggressive cleaning).
        
        Args:
            text: Cleaned Telugu text
            
        Returns:
            Training-ready text
        """
        text = self.clean_text(text, preserve_structure=True)
        
        # Normalize verse separators
        text = re.sub(r'[।॥]+', '।', text)
        
        # Remove extra punctuation
        text = re.sub(r'[,.!?;:]+', ' ', text)
        
        # Ensure consistent line endings
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        return '\n'.join(lines)
    
    # =========================================================================
    # SYLLABLE & AKSHARA EXTRACTION
    # =========================================================================
    
    def extract_aksharas(self, text: str) -> List[str]:
        """
        Extract aksharas (syllables) from Telugu text.
        
        An akshara in Telugu can be:
        - A standalone vowel (అ, ఆ, ...)
        - A consonant + vowel sign (క, కా, కి, ...)
        - A consonant cluster + vowel sign (క్క, క్ష, ...)
        
        Args:
            text: Telugu text
            
        Returns:
            List of aksharas
        """
        aksharas = []
        current = []
        
        for char in text:
            if self.phonetics.is_vowel(char):
                # Standalone vowel starts new akshara
                if current:
                    aksharas.append(''.join(current))
                    current = []
                aksharas.append(char)
            elif self.phonetics.is_consonant(char):
                # New consonant starts new akshara (unless after virama)
                if current and not (current and current[-1] == '్'):
                    aksharas.append(''.join(current))
                    current = []
                current.append(char)
            elif self.phonetics.is_vowel_sign(char) or self.phonetics.is_virama(char):
                # Vowel signs and virama attach to current consonant
                current.append(char)
            elif char in 'ంఃఁ':
                # Special marks attach to current akshara
                current.append(char)
        
        # Don't forget last akshara
        if current:
            aksharas.append(''.join(current))
        
        return aksharas
    
    def count_aksharas(self, text: str) -> int:
        """Count aksharas in text."""
        return len(self.extract_aksharas(text))
    
    def get_akshara_pattern(self, text: str) -> str:
        """
        Get prosodic pattern (guru/laghu) for each akshara.
        
        Returns:
            String of U (guru) and | (laghu)
        """
        aksharas = self.extract_aksharas(text)
        pattern = ''.join(
            TeluguPhonetics.get_syllable_weight(a) for a in aksharas
        )
        return pattern
    
    # =========================================================================
    # PRAASA (RHYME) ANALYSIS
    # =========================================================================
    
    def get_praasa_akshara(self, line: str) -> Optional[str]:
        """
        Get the praasa akshara (second syllable) of a line.
        
        Args:
            line: Single line of Telugu verse
            
        Returns:
            Second akshara or None
        """
        aksharas = self.extract_aksharas(line)
        if len(aksharas) >= 2:
            return aksharas[1]
        return None
    
    def analyze_praasa(self, poem: str) -> Dict:
        """
        Analyze praasa (rhyme scheme) in a Telugu poem.
        
        Praasa rule: Second akshara of each line should match.
        
        Args:
            poem: Telugu poem text
            
        Returns:
            Dict with praasa analysis
        """
        lines = [l.strip() for l in poem.split('\n') if l.strip()]
        
        if len(lines) < 2:
            return {
                'has_praasa': False,
                'praasa_akshara': None,
                'match_ratio': 0.0,
                'details': []
            }
        
        praasa_aksharas = []
        details = []
        
        for i, line in enumerate(lines):
            akshara = self.get_praasa_akshara(line)
            praasa_aksharas.append(akshara)
            details.append({
                'line': i + 1,
                'text_preview': line[:50] + '...' if len(line) > 50 else line,
                'praasa_akshara': akshara
            })
        
        # Filter None values
        valid_aksharas = [a for a in praasa_aksharas if a]
        
        if not valid_aksharas:
            return {
                'has_praasa': False,
                'praasa_akshara': None,
                'match_ratio': 0.0,
                'details': details
            }
        
        # Find most common praasa
        from collections import Counter
        akshara_counts = Counter(valid_aksharas)
        most_common = akshara_counts.most_common(1)[0]
        
        match_ratio = most_common[1] / len(valid_aksharas)
        
        return {
            'has_praasa': match_ratio >= 0.75,  # At least 75% match
            'praasa_akshara': most_common[0],
            'match_ratio': match_ratio,
            'details': details
        }
    
    # =========================================================================
    # CHANDASSU (METER) ANALYSIS
    # =========================================================================
    
    def identify_chandassu(self, line: str) -> Dict:
        """
        Identify the chandas (meter) of a line.
        
        Args:
            line: Single line of Telugu verse
            
        Returns:
            Dict with meter identification
        """
        aksharas = self.extract_aksharas(line)
        count = len(aksharas)
        pattern = self.get_akshara_pattern(line)
        
        # Common Telugu meters and their syllable counts
        meters = {
            # Classic meters
            'ఉత్పలమాల': {'count': 20, 'pattern': None},
            'చంపకమాల': {'count': 21, 'pattern': None},
            'మత్తేభ': {'count': 20, 'pattern': None},
            'శార్దూల': {'count': 19, 'pattern': None},
            
            # Common verse forms
            'కందం': {'count': (10, 12), 'pattern': None},
            'ఆటవెలది': {'count': (8, 10), 'pattern': None},
            'తేటగీతి': {'count': (12, 14), 'pattern': None},
            'సీసం': {'count': (16, 20), 'pattern': None},
        }
        
        possible_meters = []
        
        for meter_name, specs in meters.items():
            meter_count = specs['count']
            
            if isinstance(meter_count, tuple):
                if meter_count[0] <= count <= meter_count[1]:
                    possible_meters.append(meter_name)
            else:
                if count == meter_count:
                    possible_meters.append(meter_name)
        
        return {
            'akshara_count': count,
            'pattern': pattern,
            'possible_meters': possible_meters,
            'gana_analysis': self._analyze_ganas(pattern)
        }
    
    def _analyze_ganas(self, pattern: str) -> List[Dict]:
        """Analyze gana (prosodic unit) pattern."""
        ganas = []
        i = 0
        
        while i < len(pattern):
            if i + 3 <= len(pattern):
                triplet = pattern[i:i+3]
                # Find matching gana
                for gana_name, gana_pattern in GANA_PATTERNS.items():
                    if triplet == gana_pattern:
                        ganas.append({
                            'name': gana_name,
                            'pattern': triplet,
                            'position': i
                        })
                        break
            i += 3
        
        return ganas
    
    # =========================================================================
    # SEMANTIC & EMOTION ANALYSIS
    # =========================================================================
    
    def detect_emotions(self, text: str) -> Dict[str, float]:
        """
        Detect emotional themes in Telugu poem.
        
        Args:
            text: Telugu poem text
            
        Returns:
            Dict of emotion categories with confidence scores
        """
        text_lower = text.lower()
        scores = {}
        
        for emotion, keywords in EMOTION_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text)
            # Normalize by number of keywords
            scores[emotion] = min(1.0, count / (len(keywords) * 0.3))
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def identify_satakam(self, poem: str) -> Optional[str]:
        """
        Identify which śatakam a poem belongs to based on signature.
        
        Args:
            poem: Telugu poem text
            
        Returns:
            Śatakam name or None
        """
        poem_lower = poem.lower() if poem else ""
        
        for satakam_name, signature in SATAKAM_SIGNATURES.items():
            if signature.lower() in poem_lower or signature in poem:
                return satakam_name
        
        return None
    
    def extract_semantic_features(self, poem: str) -> Dict:
        """
        Extract comprehensive semantic features from poem.
        
        Args:
            poem: Telugu poem text
            
        Returns:
            Dict with semantic features
        """
        lines = [l.strip() for l in poem.split('\n') if l.strip()]
        
        return {
            'line_count': len(lines),
            'total_aksharas': self.count_aksharas(poem),
            'avg_aksharas_per_line': self.count_aksharas(poem) / max(1, len(lines)),
            'emotions': self.detect_emotions(poem),
            'satakam': self.identify_satakam(poem),
            'praasa': self.analyze_praasa(poem),
            'has_signature': self.identify_satakam(poem) is not None,
        }
    
    # =========================================================================
    # FULL POEM ANALYSIS
    # =========================================================================
    
    def analyze_poem(self, poem: str) -> Dict:
        """
        Comprehensive analysis of a Telugu poem.
        
        Args:
            poem: Telugu poem text
            
        Returns:
            Complete analysis dictionary
        """
        cleaned = self.clean_text(poem)
        lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
        
        # Meter analysis per line
        meter_analysis = []
        for line in lines:
            meter_analysis.append(self.identify_chandassu(line))
        
        # Praasa analysis
        praasa = self.analyze_praasa(cleaned)
        
        # Semantic features
        semantics = self.extract_semantic_features(cleaned)
        
        # Structure analysis
        structure = {
            'line_count': len(lines),
            'avg_line_length': sum(len(l) for l in lines) / max(1, len(lines)),
            'total_aksharas': self.count_aksharas(cleaned),
            'lines_with_consistent_meter': sum(
                1 for m in meter_analysis if m['possible_meters']
            ),
        }
        
        return {
            'text': cleaned,
            'structure': structure,
            'praasa': praasa,
            'meter': meter_analysis,
            'semantics': semantics,
            'quality_score': self._compute_quality_score(praasa, meter_analysis, structure)
        }
    
    def _compute_quality_score(
        self,
        praasa: Dict,
        meter_analysis: List[Dict],
        structure: Dict
    ) -> float:
        """
        Compute overall quality score for a poem.
        
        Returns:
            Score between 0 and 1
        """
        score = 0.0
        
        # Praasa contribution (30%)
        if praasa['has_praasa']:
            score += 0.3 * praasa['match_ratio']
        
        # Meter consistency (30%)
        if meter_analysis:
            consistent_ratio = structure['lines_with_consistent_meter'] / len(meter_analysis)
            score += 0.3 * consistent_ratio
        
        # Structure (20%)
        if structure['line_count'] >= 2:
            score += 0.2
        
        # Appropriate length (20%)
        avg_aksharas = structure['total_aksharas'] / max(1, structure['line_count'])
        if 8 <= avg_aksharas <= 25:  # Typical range for Telugu poetry
            score += 0.2
        
        return min(1.0, score)
    
    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================
    
    def process_dataset(self, poems: List[Dict]) -> List[Dict]:
        """
        Process entire dataset for training.
        
        Args:
            poems: List of poem dictionaries with 'text' field
            
        Returns:
            List of processed poems with features
        """
        processed = []
        
        for poem in poems:
            text = poem.get('text', '') if isinstance(poem, dict) else str(poem)
            
            # Clean text
            cleaned = self.normalize_for_training(text)
            
            if len(cleaned) < 20:  # Skip too short
                continue
            
            # Extract features
            analysis = self.analyze_poem(cleaned)
            
            processed.append({
                'original': text,
                'cleaned': cleaned,
                'features': analysis,
                'source': poem.get('source', 'unknown') if isinstance(poem, dict) else 'unknown',
                'quality_score': analysis['quality_score']
            })
        
        return processed


# Convenience functions
def preprocess_poem(text: str, preserve_structure: bool = True) -> str:
    """Quick preprocessing function."""
    preprocessor = AdvancedTeluguPreprocessor()
    return preprocessor.clean_text(text, preserve_structure)


def analyze_poem(text: str) -> Dict:
    """Quick analysis function."""
    preprocessor = AdvancedTeluguPreprocessor()
    return preprocessor.analyze_poem(text)


if __name__ == "__main__":
    # Test the preprocessor
    test_poem = """ఉప్పు కప్పురంబు నొక్కపోలికనుండు
చూడ చూడ రుచులు జాడ వేరు
పురుషులందు పుణ్య పురుషులు వేరయా
విశ్వదాభిరామ వినురవేమ"""
    
    print("Testing Advanced Telugu Preprocessor\n")
    print("="*50)
    
    preprocessor = AdvancedTeluguPreprocessor()
    
    # Test cleaning
    cleaned = preprocessor.clean_text(test_poem)
    print(f"Cleaned text:\n{cleaned}\n")
    
    # Test akshara extraction
    aksharas = preprocessor.extract_aksharas(test_poem.split('\n')[0])
    print(f"Aksharas (line 1): {aksharas}")
    print(f"Count: {len(aksharas)}\n")
    
    # Test full analysis
    analysis = preprocessor.analyze_poem(test_poem)
    print("Full Analysis:")
    print(f"  Lines: {analysis['structure']['line_count']}")
    print(f"  Praasa: {analysis['praasa']['has_praasa']} ({analysis['praasa']['praasa_akshara']})")
    print(f"  Quality Score: {analysis['quality_score']:.2f}")
    print(f"  Emotions: {analysis['semantics']['emotions']}")
    print(f"  Śatakam: {analysis['semantics']['satakam']}")
