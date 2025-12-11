"""
Text Cleaner Module
Handles text preprocessing for poem data including normalization,
cleaning, and standardization.
"""

import re
import unicodedata
from typing import List, Optional


class TextCleaner:
    """
    Cleans and normalizes poem text for processing.
    
    Handles:
    - Unicode normalization
    - Whitespace standardization
    - Punctuation handling
    - Line break normalization
    - Special character removal
    """
    
    def __init__(
        self,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        preserve_line_breaks: bool = True,
        remove_numbers: bool = False,
        normalize_whitespace: bool = True
    ):
        """
        Initialize the TextCleaner.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove all punctuation marks
            preserve_line_breaks: Keep line break structure (important for poetry)
            remove_numbers: Remove numeric characters
            normalize_whitespace: Normalize multiple spaces to single space
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.preserve_line_breaks = preserve_line_breaks
        self.remove_numbers = remove_numbers
        self.normalize_whitespace = normalize_whitespace
        
        # Punctuation pattern (preserving apostrophes for contractions)
        self.punct_pattern = re.compile(r'[^\w\s\'\-]')
        
        # Multiple whitespace pattern
        self.whitespace_pattern = re.compile(r'[ \t]+')
        
        # Multiple newline pattern
        self.newline_pattern = re.compile(r'\n\s*\n+')
        
    def clean(self, text: str) -> str:
        """
        Clean and normalize the input text.
        
        Args:
            text: Raw poem text
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Unicode normalization (NFKC for compatibility)
        text = unicodedata.normalize('NFKC', text)
        
        # Store line breaks temporarily if preserving
        if self.preserve_line_breaks:
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            lines = text.split('\n')
            cleaned_lines = [self._clean_line(line) for line in lines]
            # Remove excessive blank lines
            text = '\n'.join(cleaned_lines)
            text = self.newline_pattern.sub('\n\n', text)
        else:
            text = text.replace('\n', ' ').replace('\r', ' ')
            text = self._clean_line(text)
        
        return text.strip()
    
    def _clean_line(self, line: str) -> str:
        """Clean a single line of text."""
        # Normalize whitespace within line
        if self.normalize_whitespace:
            line = self.whitespace_pattern.sub(' ', line)
        
        # Remove punctuation if requested
        if self.remove_punctuation:
            line = self.punct_pattern.sub('', line)
        
        # Remove numbers if requested
        if self.remove_numbers:
            line = re.sub(r'\d+', '', line)
        
        # Convert to lowercase if requested
        if self.lowercase:
            line = line.lower()
        
        return line.strip()
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of raw poem texts
            
        Returns:
            List of cleaned texts
        """
        return [self.clean(text) for text in texts]
    
    def extract_lines(self, text: str) -> List[str]:
        """
        Extract individual lines from a poem.
        
        Args:
            text: Cleaned poem text
            
        Returns:
            List of poem lines (non-empty)
        """
        lines = text.split('\n')
        return [line.strip() for line in lines if line.strip()]
    
    def extract_stanzas(self, text: str) -> List[List[str]]:
        """
        Extract stanzas from a poem (separated by blank lines).
        
        Args:
            text: Cleaned poem text
            
        Returns:
            List of stanzas, each stanza is a list of lines
        """
        # Split by double newlines (stanza breaks)
        stanzas = re.split(r'\n\s*\n', text)
        result = []
        
        for stanza in stanzas:
            lines = self.extract_lines(stanza)
            if lines:
                result.append(lines)
        
        return result
    
    @staticmethod
    def count_syllables(word: str) -> int:
        """
        Estimate syllable count for a word (simple heuristic).
        
        Args:
            word: A single word
            
        Returns:
            Estimated syllable count
        """
        word = word.lower().strip()
        if not word:
            return 0
        
        # Remove non-alphabetic characters
        word = re.sub(r'[^a-z]', '', word)
        if not word:
            return 0
        
        # Count vowel groups
        vowels = 'aeiouy'
        count = 0
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        
        # Handle silent 'e' at end
        if word.endswith('e') and count > 1:
            count -= 1
        
        # Ensure at least one syllable
        return max(1, count)
    
    def get_poem_stats(self, text: str) -> dict:
        """
        Get statistics about a poem.
        
        Args:
            text: Cleaned poem text
            
        Returns:
            Dictionary with poem statistics
        """
        lines = self.extract_lines(text)
        stanzas = self.extract_stanzas(text)
        words = text.split()
        
        return {
            'num_lines': len(lines),
            'num_stanzas': len(stanzas),
            'num_words': len(words),
            'num_chars': len(text),
            'avg_words_per_line': len(words) / max(1, len(lines)),
            'avg_lines_per_stanza': len(lines) / max(1, len(stanzas))
        }


if __name__ == "__main__":
    # Example usage
    sample_poem = """
    Shall I compare thee to a summer's day?
    Thou art more lovely and more temperate:
    Rough winds do shake the darling buds of May,
    And summer's lease hath all too short a date.
    
    Sometime too hot the eye of heaven shines,
    And often is his gold complexion dimm'd;
    And every fair from fair sometime declines,
    By chance, or nature's changing course, untrimm'd.
    """
    
    cleaner = TextCleaner(normalize_whitespace=True)
    cleaned = cleaner.clean(sample_poem)
    
    print("Cleaned poem:")
    print(cleaned)
    print("\nStats:", cleaner.get_poem_stats(cleaned))
    print("\nLines:", cleaner.extract_lines(cleaned))
    print("\nStanzas:", cleaner.extract_stanzas(cleaned))
