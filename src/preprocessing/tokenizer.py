"""
Poem Tokenizer Module
Handles tokenization at character, word, and line levels for hierarchical processing.
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import json


class PoemTokenizer:
    """
    Multi-level tokenizer for poems supporting:
    - Character-level tokenization (for rhyme/rhythm patterns)
    - Word-level tokenization (for semantic meaning)
    - Line-level tokenization (for structure)
    """
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"  # Beginning of sequence
    EOS_TOKEN = "<EOS>"  # End of sequence
    SEP_TOKEN = "<SEP>"  # Line separator
    
    def __init__(
        self,
        max_vocab_size: int = 50000,
        min_freq: int = 2,
        char_level: bool = True,
        word_level: bool = True
    ):
        """
        Initialize the tokenizer.
        
        Args:
            max_vocab_size: Maximum vocabulary size for word-level
            min_freq: Minimum frequency for a token to be included
            char_level: Enable character-level tokenization
            word_level: Enable word-level tokenization
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.char_level = char_level
        self.word_level = word_level
        
        # Vocabularies
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        
        # Initialize with special tokens
        self._init_special_tokens()
        
        self.is_fitted = False
        
    def _init_special_tokens(self):
        """Initialize special token mappings."""
        special_tokens = [
            self.PAD_TOKEN, 
            self.UNK_TOKEN, 
            self.BOS_TOKEN, 
            self.EOS_TOKEN,
            self.SEP_TOKEN
        ]
        
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            self.char2idx[token] = idx
            self.idx2char[idx] = token
    
    def fit(self, texts: List[str]) -> 'PoemTokenizer':
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of poem texts
            
        Returns:
            self for chaining
        """
        # Count word frequencies
        word_counter = Counter()
        char_counter = Counter()
        
        for text in texts:
            # Word tokenization
            words = self._tokenize_words(text)
            word_counter.update(words)
            
            # Character tokenization
            for char in text:
                char_counter[char] += 1
        
        # Build word vocabulary
        if self.word_level:
            self._build_word_vocab(word_counter)
        
        # Build character vocabulary
        if self.char_level:
            self._build_char_vocab(char_counter)
        
        self.is_fitted = True
        return self
    
    def _build_word_vocab(self, counter: Counter):
        """Build word vocabulary from counter."""
        # Filter by minimum frequency
        filtered = {w: c for w, c in counter.items() if c >= self.min_freq}
        
        # Sort by frequency and take top max_vocab_size
        sorted_words = sorted(filtered.items(), key=lambda x: -x[1])
        sorted_words = sorted_words[:self.max_vocab_size - len(self.word2idx)]
        
        # Add to vocabulary
        for word, _ in sorted_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def _build_char_vocab(self, counter: Counter):
        """Build character vocabulary from counter."""
        # Add all characters (no min_freq for chars)
        for char, _ in counter.most_common():
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization preserving contractions
        words = re.findall(r"\b[\w']+\b|[^\w\s]", text.lower())
        return words
    
    def encode_words(
        self, 
        text: str, 
        max_length: Optional[int] = None,
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        Encode text to word indices.
        
        Args:
            text: Input text
            max_length: Maximum sequence length (will pad/truncate)
            add_special_tokens: Add BOS and EOS tokens
            
        Returns:
            List of word indices
        """
        words = self._tokenize_words(text)
        
        # Convert to indices
        indices = []
        if add_special_tokens:
            indices.append(self.word2idx[self.BOS_TOKEN])
        
        for word in words:
            idx = self.word2idx.get(word, self.word2idx[self.UNK_TOKEN])
            indices.append(idx)
        
        if add_special_tokens:
            indices.append(self.word2idx[self.EOS_TOKEN])
        
        # Pad or truncate
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                pad_idx = self.word2idx[self.PAD_TOKEN]
                indices.extend([pad_idx] * (max_length - len(indices)))
        
        return indices
    
    def encode_chars(
        self, 
        text: str, 
        max_length: Optional[int] = None,
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        Encode text to character indices.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            add_special_tokens: Add BOS and EOS tokens
            
        Returns:
            List of character indices
        """
        indices = []
        if add_special_tokens:
            indices.append(self.char2idx[self.BOS_TOKEN])
        
        for char in text:
            idx = self.char2idx.get(char, self.char2idx[self.UNK_TOKEN])
            indices.append(idx)
        
        if add_special_tokens:
            indices.append(self.char2idx[self.EOS_TOKEN])
        
        # Pad or truncate
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                pad_idx = self.char2idx[self.PAD_TOKEN]
                indices.extend([pad_idx] * (max_length - len(indices)))
        
        return indices
    
    def encode_lines(
        self,
        text: str,
        max_lines: Optional[int] = None,
        max_words_per_line: Optional[int] = None
    ) -> List[List[int]]:
        """
        Encode text at line level (list of word-encoded lines).
        
        Args:
            text: Input poem text
            max_lines: Maximum number of lines
            max_words_per_line: Maximum words per line
            
        Returns:
            List of encoded lines
        """
        lines = text.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        if max_lines and len(lines) > max_lines:
            lines = lines[:max_lines]
        
        encoded_lines = []
        for line in lines:
            encoded = self.encode_words(
                line, 
                max_length=max_words_per_line,
                add_special_tokens=True
            )
            encoded_lines.append(encoded)
        
        # Pad with empty lines if needed
        if max_lines:
            pad_idx = self.word2idx[self.PAD_TOKEN]
            empty_line = [pad_idx] * (max_words_per_line or 1)
            while len(encoded_lines) < max_lines:
                encoded_lines.append(empty_line)
        
        return encoded_lines
    
    def decode_words(self, indices: List[int], skip_special: bool = True) -> str:
        """
        Decode word indices back to text.
        
        Args:
            indices: List of word indices
            skip_special: Skip special tokens in output
            
        Returns:
            Decoded text
        """
        special = {
            self.word2idx[self.PAD_TOKEN],
            self.word2idx[self.BOS_TOKEN],
            self.word2idx[self.EOS_TOKEN]
        }
        
        words = []
        for idx in indices:
            if skip_special and idx in special:
                continue
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            words.append(word)
        
        return ' '.join(words)
    
    def decode_chars(self, indices: List[int], skip_special: bool = True) -> str:
        """
        Decode character indices back to text.
        
        Args:
            indices: List of character indices
            skip_special: Skip special tokens
            
        Returns:
            Decoded text
        """
        special = {
            self.char2idx[self.PAD_TOKEN],
            self.char2idx[self.BOS_TOKEN],
            self.char2idx[self.EOS_TOKEN]
        }
        
        chars = []
        for idx in indices:
            if skip_special and idx in special:
                continue
            char = self.idx2char.get(idx, '')
            chars.append(char)
        
        return ''.join(chars)
    
    @property
    def word_vocab_size(self) -> int:
        """Get word vocabulary size."""
        return len(self.word2idx)
    
    @property
    def char_vocab_size(self) -> int:
        """Get character vocabulary size."""
        return len(self.char2idx)
    
    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            'word2idx': self.word2idx,
            'char2idx': self.char2idx,
            'max_vocab_size': self.max_vocab_size,
            'min_freq': self.min_freq,
            'char_level': self.char_level,
            'word_level': self.word_level
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'PoemTokenizer':
        """Load tokenizer from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(
            max_vocab_size=data['max_vocab_size'],
            min_freq=data['min_freq'],
            char_level=data['char_level'],
            word_level=data['word_level']
        )
        
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = {int(k): v for k, v in 
                              {v: k for k, v in data['word2idx'].items()}.items()}
        tokenizer.char2idx = data['char2idx']
        tokenizer.idx2char = {int(k): v for k, v in 
                              {v: k for k, v in data['char2idx'].items()}.items()}
        tokenizer.is_fitted = True
        
        return tokenizer


if __name__ == "__main__":
    # Example usage
    poems = [
        """Shall I compare thee to a summer's day?
        Thou art more lovely and more temperate.""",
        """Two roads diverged in a yellow wood,
        And sorry I could not travel both."""
    ]
    
    tokenizer = PoemTokenizer(max_vocab_size=1000, min_freq=1)
    tokenizer.fit(poems)
    
    print(f"Word vocab size: {tokenizer.word_vocab_size}")
    print(f"Char vocab size: {tokenizer.char_vocab_size}")
    
    # Test encoding/decoding
    test = "Shall I compare thee"
    word_encoded = tokenizer.encode_words(test, max_length=10)
    print(f"\nWord encoded: {word_encoded}")
    print(f"Word decoded: {tokenizer.decode_words(word_encoded)}")
    
    char_encoded = tokenizer.encode_chars(test, max_length=30)
    print(f"\nChar encoded: {char_encoded}")
    print(f"Char decoded: {tokenizer.decode_chars(char_encoded)}")
