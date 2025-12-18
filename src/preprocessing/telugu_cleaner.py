"""
Telugu Text Cleaner and Preprocessor
Handles Telugu Unicode text processing for poem analysis.
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional


class TeluguTextCleaner:
    """
    Text cleaner specifically for Telugu language poems.
    Handles Telugu Unicode characters and prosody analysis.
    """
    
    # Telugu Unicode ranges
    TELUGU_RANGE = (0x0C00, 0x0C7F)
    
    # Telugu vowels (అచ్చులు)
    VOWELS = 'అఆఇఈఉఊఋౠఎఏఐఒఓఔ'
    
    # Telugu vowel signs (మాత్రలు)
    VOWEL_SIGNS = 'ాిీుూృౄెేైొోౌ'
    
    # Telugu consonants (హల్లులు)
    CONSONANTS = 'కఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరలవశషసహళక్షఱ'
    
    # Telugu digits
    DIGITS = '౦౧౨౩౪౫౬౭౮౯'
    
    # Chandrabindu and other special marks
    SPECIAL_MARKS = 'ంఃఁ'
    
    def __init__(self):
        """Initialize Telugu text cleaner."""
        self.vowel_set = set(self.VOWELS)
        self.consonant_set = set(self.CONSONANTS)
        self.vowel_sign_set = set(self.VOWEL_SIGNS)
        
    def is_telugu(self, char: str) -> bool:
        """Check if character is Telugu."""
        if not char:
            return False
        code = ord(char)
        return self.TELUGU_RANGE[0] <= code <= self.TELUGU_RANGE[1]
    
    def is_vowel(self, char: str) -> bool:
        """Check if character is a Telugu vowel."""
        return char in self.vowel_set
    
    def is_consonant(self, char: str) -> bool:
        """Check if character is a Telugu consonant."""
        return char in self.consonant_set
    
    def is_vowel_sign(self, char: str) -> bool:
        """Check if character is a Telugu vowel sign (maatra)."""
        return char in self.vowel_sign_set
    
    def clean(self, text: str) -> str:
        """
        Clean Telugu text.
        
        Args:
            text: Raw Telugu text
            
        Returns:
            Cleaned Telugu text
        """
        if not text:
            return ""
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Keep Telugu characters, punctuation, and newlines
        cleaned = []
        for char in text:
            if self.is_telugu(char):
                cleaned.append(char)
            elif char in ' \n\t.,!?;:\'\"()-–—':
                cleaned.append(char)
            elif char.isspace():
                cleaned.append(' ')
        
        text = ''.join(cleaned)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def count_aksharas(self, text: str) -> int:
        """
        Count Telugu aksharas (syllables).
        In Telugu, an akshara = consonant + vowel or standalone vowel.
        
        Args:
            text: Telugu text
            
        Returns:
            Number of aksharas
        """
        count = 0
        prev_consonant = False
        
        for char in text:
            if self.is_vowel(char):
                count += 1
                prev_consonant = False
            elif self.is_consonant(char):
                # Consonant starts a new akshara
                count += 1
                prev_consonant = True
            elif self.is_vowel_sign(char):
                # Vowel sign modifies previous consonant, don't count
                if prev_consonant:
                    prev_consonant = False
            elif char == '్':  # Virama/Halant
                # Consonant cluster, don't count separately
                count -= 1 if count > 0 else 0
                prev_consonant = True
        
        return count
    
    def split_into_lines(self, text: str) -> List[str]:
        """Split text into lines."""
        lines = text.split('\n')
        return [line.strip() for line in lines if line.strip()]
    
    def get_second_letter(self, line: str) -> Optional[str]:
        """
        Get the second akshara of a line (for praasa analysis).
        
        Args:
            line: Telugu text line
            
        Returns:
            Second akshara or None
        """
        aksharas = self.extract_aksharas(line)
        if len(aksharas) >= 2:
            return aksharas[1]
        return None
    
    def extract_aksharas(self, text: str) -> List[str]:
        """
        Extract aksharas (syllables) from Telugu text.
        
        Args:
            text: Telugu text
            
        Returns:
            List of aksharas
        """
        aksharas = []
        current = []
        
        for char in text:
            if self.is_vowel(char):
                if current:
                    aksharas.append(''.join(current))
                    current = []
                aksharas.append(char)
            elif self.is_consonant(char):
                if current and current[-1] != '్':
                    aksharas.append(''.join(current))
                    current = []
                current.append(char)
            elif self.is_vowel_sign(char) or char in self.SPECIAL_MARKS or char == '్':
                current.append(char)
            elif char.isspace():
                if current:
                    aksharas.append(''.join(current))
                    current = []
        
        if current:
            aksharas.append(''.join(current))
        
        return aksharas
    
    def check_praasa(self, lines: List[str]) -> Dict:
        """
        Check praasa (rhyme) in Telugu verse.
        Praasa = second akshara should be same in all lines.
        
        Args:
            lines: List of Telugu text lines
            
        Returns:
            Dict with praasa analysis
        """
        second_aksharas = []
        for line in lines:
            second = self.get_second_letter(line)
            if second:
                second_aksharas.append(second)
        
        if not second_aksharas:
            return {'has_praasa': False, 'aksharas': []}
        
        # Check if all second aksharas match
        first = second_aksharas[0]
        matches = sum(1 for a in second_aksharas if a == first)
        
        return {
            'has_praasa': matches == len(second_aksharas),
            'praasa_akshara': first,
            'match_ratio': matches / len(second_aksharas),
            'aksharas': second_aksharas
        }
    
    def analyze_meter(self, line: str) -> Dict:
        """
        Analyze meter (chandassu) of a line.
        
        Args:
            line: Telugu text line
            
        Returns:
            Dict with meter analysis
        """
        aksharas = self.extract_aksharas(line)
        count = len(aksharas)
        
        # Identify possible chandassu based on syllable count
        possible_meters = []
        
        if count == 20:
            possible_meters.append('ఉత్పలమాల (Utpalamala)')
        elif count == 21:
            possible_meters.append('చంపకమాల (Champakamala)')
        elif count in [10, 11, 12]:
            possible_meters.append('కందం (Kandham)')
        elif count in [8, 9]:
            possible_meters.append('ఆటవెలది (Aataveladi)')
        
        return {
            'aksharas': aksharas,
            'count': count,
            'possible_meters': possible_meters
        }
    
    def get_stats(self, text: str) -> Dict:
        """
        Get statistics about Telugu text.
        
        Args:
            text: Telugu poem text
            
        Returns:
            Dict with various statistics
        """
        lines = self.split_into_lines(text)
        
        telugu_chars = sum(1 for c in text if self.is_telugu(c))
        total_chars = len(text)
        
        aksharas_per_line = [self.count_aksharas(line) for line in lines]
        
        praasa_result = self.check_praasa(lines)
        
        return {
            'num_lines': len(lines),
            'total_characters': total_chars,
            'telugu_characters': telugu_chars,
            'telugu_ratio': telugu_chars / total_chars if total_chars > 0 else 0,
            'aksharas_per_line': aksharas_per_line,
            'avg_aksharas_per_line': sum(aksharas_per_line) / len(aksharas_per_line) if aksharas_per_line else 0,
            'praasa': praasa_result,
            'lines': lines
        }


# Sample Telugu poems for testing
SAMPLE_TELUGU_POEMS = [
    {
        "title": "చందమామ రావే",
        "text": """చందమామ రావే
జాబిల్లి రావే
నీ పాప వచ్చెను
పాలాళి తెచ్చెను""",
        "type": "జానపద గేయం"
    },
    {
        "title": "వేమన పద్యం",
        "text": """ఉప్పు కప్పురంబు నొక్కపోలికనుండు
చూడ చూడ రుచులు జాడ వేరు
పురుషులందు పుణ్య పురుషులు వేరయా
విశ్వదాభిరామ వినురవేమ""",
        "type": "ఆట వెలది"
    },
    {
        "title": "భాగవతం - పోతన",
        "text": """శ్రీకైవల్య పదంబు చేరుటకునై చింతించి నిత్యంబు సం
ధ్యాకాలంబును సంధ్యయందు మదిలో ధ్యానంబు చేయంగడున్""",
        "type": "శార్దూల విక్రీడితం"
    }
]


def create_sample_telugu_dataset(output_path: str = None) -> List[Dict]:
    """
    Create a sample Telugu poem dataset.
    
    Args:
        output_path: Path to save JSON file (optional)
        
    Returns:
        List of poem dictionaries
    """
    import json
    
    poems = []
    
    # Add sample poems
    for poem in SAMPLE_TELUGU_POEMS:
        poems.append({
            'text': poem['text'],
            'title': poem['title'],
            'style': poem['type'],
            'language': 'telugu'
        })
    
    # Add more sample verses
    additional_poems = [
        "తల్లి దండ్రి మీరు దైవము మీరయా\nతల్లి దండ్రి సేవ తపము కంటె మిన్న",
        "మంచి మాట వినుము మంచి బుద్ధి నేర్పు\nచెడ్డమాట వినకు చెడిపోదువు",
        "అరిషడ్వర్గములను అణచి వేయవలయు\nమనసు స్వాధీనమున మసలవలయు"
    ]
    
    for i, text in enumerate(additional_poems):
        poems.append({
            'text': text,
            'title': f'సూక్తి {i+1}',
            'style': 'నీతి పద్యం',
            'language': 'telugu'
        })
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(poems, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(poems)} Telugu poems to {output_path}")
    
    return poems


if __name__ == "__main__":
    # Test Telugu text cleaner
    cleaner = TeluguTextCleaner()
    
    test_text = """చందమామ రావే
జాబిల్లి రావే
నీ పాప వచ్చెను
పాలాళి తెచ్చెను"""
    
    print("Testing Telugu Text Cleaner")
    print("=" * 50)
    print(f"Input:\n{test_text}")
    print()
    
    stats = cleaner.get_stats(test_text)
    print(f"Lines: {stats['num_lines']}")
    print(f"Aksharas per line: {stats['aksharas_per_line']}")
    print(f"Praasa: {stats['praasa']}")
    
    # Create sample dataset
    create_sample_telugu_dataset("data/processed/telugu_sample.json")
