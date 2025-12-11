#!/usr/bin/env python
"""
Training Script for CNN-Based Poem Learner
Usage: python scripts/train.py --config config/config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import json

from src.models.poem_learner import PoemLearner, create_poem_learner
from src.preprocessing.tokenizer import PoemTokenizer
from src.preprocessing.text_cleaner import TextCleaner
from src.training.trainer import Trainer
from src.utils.helpers import set_seed, load_config, get_device


class PoemDataset(Dataset):
    """Dataset for loading poems."""
    
    def __init__(
        self,
        poems: list,
        tokenizer: PoemTokenizer,
        max_length: int = 256,
        for_generation: bool = True
    ):
        self.poems = poems
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.for_generation = for_generation
    
    def __len__(self):
        return len(self.poems)
    
    def __getitem__(self, idx):
        poem = self.poems[idx]
        
        if isinstance(poem, dict):
            text = poem.get('text', '')
            style = poem.get('style', 0)
        else:
            text = poem
            style = 0
        
        # Encode
        input_ids = self.tokenizer.encode_words(
            text, max_length=self.max_length
        )
        
        if self.for_generation:
            # For language modeling, target is shifted input
            target_ids = input_ids[1:] + [0]  # Shift left, pad end
        else:
            target_ids = input_ids
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(
                [1 if i != 0 else 0 for i in input_ids], 
                dtype=torch.float
            ),
            'style_labels': torch.tensor(style, dtype=torch.long)
        }


def load_poems(data_path: str) -> list:
    """Load poems from file."""
    path = Path(data_path)
    
    if not path.exists():
        print(f"Data file not found: {path}")
        print("Creating sample poems for demonstration...")
        return create_sample_poems()
    
    if path.suffix == '.json':
        with open(path) as f:
            data = json.load(f)
        return data
    elif path.suffix == '.txt':
        with open(path) as f:
            # Assume poems are separated by blank lines
            content = f.read()
            poems = content.split('\n\n')
            return [p.strip() for p in poems if p.strip()]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def create_sample_poems() -> list:
    """Create sample poems for testing."""
    return [
        {
            'text': """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate.
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date.""",
            'style': 0
        },
        {
            'text': """Two roads diverged in a yellow wood,
And sorry I could not travel both
And be one traveler, long I stood
And looked down one as far as I could.""",
            'style': 1
        },
        {
            'text': """The fog comes on little cat feet.
It sits looking over harbor and city
on silent haunches and then moves on.""",
            'style': 2
        },
        {
            'text': """I wandered lonely as a cloud
That floats on high o'er vales and hills,
When all at once I saw a crowd,
A host, of golden daffodils.""",
            'style': 0
        },
        {
            'text': """Hope is the thing with feathers
That perches in the soul,
And sings the tune without the words,
And never stops at all.""",
            'style': 1
        },
        {
            'text': """Do not go gentle into that good night,
Old age should burn and rave at close of day;
Rage, rage against the dying of the light.""",
            'style': 2
        },
        {
            'text': """The road not taken makes all the difference,
I shall be telling this with a sigh
Somewhere ages and ages hence.""",
            'style': 1
        },
        {
            'text': """Because I could not stop for Death,
He kindly stopped for me;
The carriage held but just ourselves
And Immortality.""",
            'style': 0
        },
    ] * 50  # Repeat for larger dataset


def main():
    parser = argparse.ArgumentParser(description='Train Poem Learner')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data', type=str, default='data/processed/train.json',
                       help='Path to training data')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    print("=" * 60)
    print("CNN-Based Poem Learner Training")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    poems = load_poems(args.data)
    print(f"Loaded {len(poems)} poems")
    
    # Clean poems
    cleaner = TextCleaner()
    if isinstance(poems[0], dict):
        for p in poems:
            p['text'] = cleaner.clean(p['text'])
        texts = [p['text'] for p in poems]
    else:
        poems = [cleaner.clean(p) for p in poems]
        texts = poems
    
    # Build tokenizer
    print("\nBuilding tokenizer...")
    tokenizer = PoemTokenizer(max_vocab_size=10000, min_freq=1)
    tokenizer.fit(texts)
    print(f"Vocabulary size: {tokenizer.word_vocab_size}")
    
    # Create dataset
    max_length = config['data']['max_seq_length']
    dataset = PoemDataset(poems, tokenizer, max_length=max_length)
    
    # Split data
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=0
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create model
    print("\nCreating model...")
    config['vocab_size'] = tokenizer.word_vocab_size
    model = create_poem_learner(config)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Get device
    device = get_device(
        use_cuda=config.get('device', {}).get('use_cuda', True)
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    print("\nStarting training...")
    summary = trainer.train()
    
    # Save tokenizer
    tokenizer.save('checkpoints/tokenizer.json')
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {summary['best_val_loss']:.4f}")
    print(f"Final train loss: {summary['final_train_loss']:.4f}")
    print("Checkpoints saved to: checkpoints/")


if __name__ == '__main__':
    main()
