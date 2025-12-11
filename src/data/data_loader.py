"""
Data Loader Module
Handles loading and processing poem datasets from various sources.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterator
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import random


class PoemDataset(Dataset):
    """
    Dataset for loading poems for training and evaluation.
    
    Supports:
    - JSON format (list of poem dictionaries)
    - Plain text (poems separated by blank lines)
    - CSV/TSV format
    """
    
    def __init__(
        self,
        poems: List[Dict],
        tokenizer,
        max_seq_length: int = 256,
        max_lines: int = 32,
        for_generation: bool = True,
        include_style: bool = True
    ):
        """
        Initialize PoemDataset.
        
        Args:
            poems: List of poem dictionaries with 'text' key
            tokenizer: PoemTokenizer instance
            max_seq_length: Maximum token sequence length
            max_lines: Maximum lines per poem
            for_generation: Whether to prepare for generation (shift targets)
            include_style: Include style labels if available
        """
        self.poems = poems
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_lines = max_lines
        self.for_generation = for_generation
        self.include_style = include_style
    
    def __len__(self) -> int:
        return len(self.poems)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        poem = self.poems[idx]
        
        # Extract text and metadata
        if isinstance(poem, dict):
            text = poem.get('text', '')
            style = poem.get('style', poem.get('style_id', 0))
            title = poem.get('title', '')
            author = poem.get('author', '')
        else:
            text = str(poem)
            style = 0
            title = ''
            author = ''
        
        # Tokenize
        input_ids = self.tokenizer.encode_words(
            text, 
            max_length=self.max_seq_length,
            add_special_tokens=True
        )
        
        # Create target (shifted for autoregressive training)
        if self.for_generation:
            target_ids = input_ids[1:] + [self.tokenizer.word2idx.get('<PAD>', 0)]
        else:
            target_ids = input_ids.copy()
        
        # Create attention mask
        pad_id = self.tokenizer.word2idx.get('<PAD>', 0)
        attention_mask = [1 if token != pad_id else 0 for token in input_ids]
        
        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float)
        }
        
        if self.include_style:
            result['style_labels'] = torch.tensor(style, dtype=torch.long)
        
        return result


class PoemDataLoader:
    """
    Utility class for loading poem data from various sources.
    """
    
    SUPPORTED_FORMATS = ['json', 'txt', 'csv', 'tsv']
    
    @staticmethod
    def load_from_file(filepath: str) -> List[Dict]:
        """
        Load poems from a file.
        
        Args:
            filepath: Path to data file
            
        Returns:
            List of poem dictionaries
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        suffix = path.suffix.lower().lstrip('.')
        
        if suffix == 'json':
            return PoemDataLoader._load_json(path)
        elif suffix == 'txt':
            return PoemDataLoader._load_text(path)
        elif suffix in ('csv', 'tsv'):
            return PoemDataLoader._load_csv(path, delimiter=',' if suffix == 'csv' else '\t')
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    @staticmethod
    def _load_json(path: Path) -> List[Dict]:
        """Load JSON format poems."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return [{'text': item} if isinstance(item, str) else item for item in data]
        elif isinstance(data, dict):
            if 'poems' in data:
                return data['poems']
            elif 'data' in data:
                return data['data']
        
        return data
    
    @staticmethod
    def _load_text(path: Path) -> List[Dict]:
        """Load plain text format (poems separated by blank lines)."""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines
        poems = content.split('\n\n\n')
        if len(poems) == 1:
            poems = content.split('\n\n')
        
        return [{'text': p.strip()} for p in poems if p.strip()]
    
    @staticmethod
    def _load_csv(path: Path, delimiter: str = ',') -> List[Dict]:
        """Load CSV/TSV format poems."""
        import csv
        
        poems = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                poem = {}
                # Find text column
                for key in ['text', 'poem', 'content', 'body']:
                    if key in row:
                        poem['text'] = row[key]
                        break
                
                # Find other metadata
                if 'title' in row:
                    poem['title'] = row['title']
                if 'author' in row:
                    poem['author'] = row['author']
                if 'style' in row:
                    poem['style'] = int(row['style']) if row['style'].isdigit() else 0
                
                if 'text' in poem:
                    poems.append(poem)
        
        return poems
    
    @staticmethod
    def load_poetry_foundation(filepath: str) -> List[Dict]:
        """
        Load Poetry Foundation dataset format.
        Expected: CSV with 'Content' column for poem text.
        """
        import csv
        
        poems = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get('Content', row.get('Poem', ''))
                if text.strip():
                    poems.append({
                        'text': text.strip(),
                        'title': row.get('Title', ''),
                        'author': row.get('Author', row.get('Poet', '')),
                        'style': 0  # Could be derived from tags
                    })
        
        return poems
    
    @staticmethod
    def create_sample_dataset(num_poems: int = 100) -> List[Dict]:
        """
        Create a sample dataset for testing.
        
        Args:
            num_poems: Number of poems to generate
            
        Returns:
            List of sample poem dictionaries
        """
        sample_poems = [
            {
                'text': """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate.
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date.""",
                'title': 'Sonnet 18',
                'author': 'Shakespeare',
                'style': 0
            },
            {
                'text': """Two roads diverged in a yellow wood,
And sorry I could not travel both
And be one traveler, long I stood
And looked down one as far as I could
To where it bent in the undergrowth.""",
                'title': 'The Road Not Taken',
                'author': 'Robert Frost',
                'style': 1
            },
            {
                'text': """I wandered lonely as a cloud
That floats on high o'er vales and hills,
When all at once I saw a crowd,
A host, of golden daffodils;
Beside the lake, beneath the trees,
Fluttering and dancing in the breeze.""",
                'title': 'Daffodils',
                'author': 'William Wordsworth',
                'style': 0
            },
            {
                'text': """Hope is the thing with feathers
That perches in the soul,
And sings the tune without the words,
And never stops at all.""",
                'title': 'Hope is the thing with feathers',
                'author': 'Emily Dickinson',
                'style': 2
            },
            {
                'text': """The fog comes on little cat feet.
It sits looking over harbor and city
on silent haunches and then moves on.""",
                'title': 'Fog',
                'author': 'Carl Sandburg',
                'style': 2
            },
            {
                'text': """Do not go gentle into that good night,
Old age should burn and rave at close of day;
Rage, rage against the dying of the light.""",
                'title': 'Do Not Go Gentle',
                'author': 'Dylan Thomas',
                'style': 1
            },
            {
                'text': """Because I could not stop for Death,
He kindly stopped for me;
The carriage held but just ourselves
And Immortality.""",
                'title': 'Because I could not stop for Death',
                'author': 'Emily Dickinson',
                'style': 2
            },
            {
                'text': """Tyger Tyger, burning bright,
In the forests of the night;
What immortal hand or eye,
Could frame thy fearful symmetry?""",
                'title': 'The Tyger',
                'author': 'William Blake',
                'style': 0
            },
        ]
        
        # Repeat to reach desired count
        result = []
        while len(result) < num_poems:
            for poem in sample_poems:
                if len(result) >= num_poems:
                    break
                result.append(poem.copy())
        
        return result


def create_data_loaders(
    train_data: List[Dict],
    val_data: Optional[List[Dict]],
    tokenizer,
    config: Dict
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation data loaders.
    
    Args:
        train_data: Training poems
        val_data: Validation poems (optional)
        tokenizer: PoemTokenizer instance
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = PoemDataset(
        poems=train_data,
        tokenizer=tokenizer,
        max_seq_length=config.get('data', {}).get('max_seq_length', 256)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('training', {}).get('batch_size', 32),
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = None
    if val_data:
        val_dataset = PoemDataset(
            poems=val_data,
            tokenizer=tokenizer,
            max_seq_length=config.get('data', {}).get('max_seq_length', 256)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('training', {}).get('batch_size', 32),
            shuffle=False,
            num_workers=0
        )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loader
    from src.preprocessing.tokenizer import PoemTokenizer
    
    # Create sample data
    poems = PoemDataLoader.create_sample_dataset(50)
    print(f"Created {len(poems)} sample poems")
    
    # Create tokenizer
    tokenizer = PoemTokenizer()
    texts = [p['text'] for p in poems]
    tokenizer.fit(texts)
    print(f"Tokenizer vocab size: {tokenizer.word_vocab_size}")
    
    # Create dataset
    dataset = PoemDataset(poems, tokenizer)
    print(f"Dataset size: {len(dataset)}")
    
    # Test item
    sample = dataset[0]
    print(f"\nSample item:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  target_ids shape: {sample['target_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
