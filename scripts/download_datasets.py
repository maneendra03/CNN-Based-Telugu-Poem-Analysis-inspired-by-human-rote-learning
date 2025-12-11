#!/usr/bin/env python
"""
Dataset Download Script
Downloads and prepares poem datasets from various sources.
"""

import os
import json
import csv
import urllib.request
import zipfile
import shutil
from pathlib import Path
from typing import List, Dict


def download_file(url: str, dest_path: Path, description: str = ""):
    """Download a file from URL."""
    print(f"Downloading {description or url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  ✓ Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_shakespeare_sonnets(output_dir: Path) -> List[Dict]:
    """
    Download Shakespeare's sonnets from Project Gutenberg.
    """
    print("\n📚 Downloading Shakespeare Sonnets...")
    
    url = "https://www.gutenberg.org/cache/epub/1041/pg1041.txt"
    temp_file = output_dir / "shakespeare_raw.txt"
    
    if not download_file(url, temp_file, "Shakespeare Sonnets"):
        # Fallback: use built-in sonnets
        return get_builtin_shakespeare()
    
    # Parse sonnets from the text
    with open(temp_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    sonnets = []
    # Find sonnets by number
    import re
    
    # Split by "I", "II", etc. or numbers
    pattern = r'\n\s*([IVX]+|\d+)\s*\n'
    parts = re.split(pattern, content)
    
    current_num = 0
    for i, part in enumerate(parts):
        if re.match(r'^[IVX]+$|^\d+$', part.strip()):
            current_num += 1
            if i + 1 < len(parts):
                text = parts[i + 1].strip()
                if len(text) > 50 and len(text) < 2000:  # Reasonable sonnet length
                    sonnets.append({
                        'text': text,
                        'title': f'Sonnet {current_num}',
                        'author': 'William Shakespeare',
                        'style': 0  # Shakespearean style
                    })
    
    # Clean up
    if temp_file.exists():
        temp_file.unlink()
    
    print(f"  ✓ Extracted {len(sonnets)} sonnets")
    return sonnets[:154]  # Shakespeare wrote 154 sonnets


def get_builtin_shakespeare() -> List[Dict]:
    """Fallback built-in Shakespeare sonnets."""
    sonnets = [
        {
            'text': """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate.
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date.
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimmed;
And every fair from fair sometime declines,
By chance or nature's changing course untrimmed.
But thy eternal summer shall not fade
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st.
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.""",
            'title': 'Sonnet 18',
            'author': 'William Shakespeare',
            'style': 0
        },
        {
            'text': """When, in disgrace with fortune and men's eyes,
I all alone beweep my outcast state
And trouble deaf heaven with my bootless cries
And look upon myself and curse my fate,
Wishing me like to one more rich in hope,
Featured like him, like him with friends possess'd,
Desiring this man's art and that man's scope,
With what I most enjoy contented least;
Yet in these thoughts myself almost despising,
Haply I think on thee, and then my state,
Like to the lark at break of day arising
From sullen earth, sings hymns at heaven's gate;
For thy sweet love remember'd such wealth brings
That then I scorn to change my state with kings.""",
            'title': 'Sonnet 29',
            'author': 'William Shakespeare',
            'style': 0
        },
        {
            'text': """Let me not to the marriage of true minds
Admit impediments. Love is not love
Which alters when it alteration finds,
Or bends with the remover to remove:
O no! it is an ever-fixed mark
That looks on tempests and is never shaken;
It is the star to every wandering bark,
Whose worth's unknown, although his height be taken.
Love's not Time's fool, though rosy lips and cheeks
Within his bending sickle's compass come:
Love alters not with his brief hours and weeks,
But bears it out even to the edge of doom.
If this be error and upon me proved,
I never writ, nor no man ever loved.""",
            'title': 'Sonnet 116',
            'author': 'William Shakespeare',
            'style': 0
        },
    ]
    return sonnets * 50  # Repeat for training


def download_poetry_foundation_sample(output_dir: Path) -> List[Dict]:
    """
    Create Poetry Foundation style sample dataset.
    Note: Full dataset requires Kaggle API authentication.
    """
    print("\n📚 Creating Poetry Foundation sample...")
    
    # Sample poems from various poets (public domain)
    poems = [
        {
            'text': """I wandered lonely as a cloud
That floats on high o'er vales and hills,
When all at once I saw a crowd,
A host, of golden daffodils;
Beside the lake, beneath the trees,
Fluttering and dancing in the breeze.

Continuous as the stars that shine
And twinkle on the milky way,
They stretched in never-ending line
Along the margin of a bay:
Ten thousand saw I at a glance,
Tossing their heads in sprightly dance.""",
            'title': 'I Wandered Lonely as a Cloud',
            'author': 'William Wordsworth',
            'style': 1  # Romantic style
        },
        {
            'text': """Two roads diverged in a yellow wood,
And sorry I could not travel both
And be one traveler, long I stood
And looked down one as far as I could
To where it bent in the undergrowth;

Then took the other, as just as fair,
And having perhaps the better claim,
Because it was grassy and wanted wear;
Though as for that the passing there
Had worn them really about the same.""",
            'title': 'The Road Not Taken',
            'author': 'Robert Frost',
            'style': 2  # Modern style
        },
        {
            'text': """Hope is the thing with feathers
That perches in the soul,
And sings the tune without the words,
And never stops at all,

And sweetest in the gale is heard;
And sore must be the storm
That could abash the little bird
That kept so many warm.""",
            'title': 'Hope is the thing with feathers',
            'author': 'Emily Dickinson',
            'style': 3  # Dickinson style
        },
        {
            'text': """Do not go gentle into that good night,
Old age should burn and rave at close of day;
Rage, rage against the dying of the light.

Though wise men at their end know dark is right,
Because their words had forked no lightning they
Do not go gentle into that good night.""",
            'title': 'Do Not Go Gentle Into That Good Night',
            'author': 'Dylan Thomas',
            'style': 2  # Modern style
        },
        {
            'text': """Tyger Tyger, burning bright,
In the forests of the night;
What immortal hand or eye,
Could frame thy fearful symmetry?

In what distant deeps or skies,
Burnt the fire of thine eyes?
On what wings dare he aspire?
What the hand, dare seize the fire?""",
            'title': 'The Tyger',
            'author': 'William Blake',
            'style': 1  # Romantic style
        },
        {
            'text': """Because I could not stop for Death –
He kindly stopped for me –
The Carriage held but just Ourselves –
And Immortality.

We slowly drove – He knew no haste
And I had put away
My labor and my leisure too,
For His Civility –""",
            'title': 'Because I could not stop for Death',
            'author': 'Emily Dickinson',
            'style': 3  # Dickinson style
        },
        {
            'text': """The fog comes
on little cat feet.

It sits looking
over harbor and city
on silent haunches
and then moves on.""",
            'title': 'Fog',
            'author': 'Carl Sandburg',
            'style': 2  # Modern/Imagist
        },
        {
            'text': """What happens to a dream deferred?

Does it dry up
like a raisin in the sun?
Or fester like a sore—
And then run?
Does it stink like rotten meat?
Or crust and sugar over—
like a syrupy sweet?

Maybe it just sags
like a heavy load.

Or does it explode?""",
            'title': 'Harlem',
            'author': 'Langston Hughes',
            'style': 4  # Harlem Renaissance
        },
        {
            'text': """I have eaten
the plums
that were in
the icebox

and which
you were probably
saving
for breakfast

Forgive me
they were delicious
so sweet
and so cold""",
            'title': 'This Is Just to Say',
            'author': 'William Carlos Williams',
            'style': 2  # Imagist/Modern
        },
        {
            'text': """The world is too much with us; late and soon,
Getting and spending, we lay waste our powers:
Little we see in Nature that is ours;
We have given our hearts away, a sordid boon!
This Sea that bares her bosom to the moon;
The winds that will be howling at all hours,
And are up-gathered now like sleeping flowers;
For this, for everything, we are out of tune.""",
            'title': 'The World Is Too Much With Us',
            'author': 'William Wordsworth',
            'style': 1  # Romantic
        },
    ]
    
    # Expand dataset by repeating with variations
    expanded = []
    for i, poem in enumerate(poems):
        for j in range(20):  # 20 copies each
            expanded.append({
                'text': poem['text'],
                'title': poem['title'],
                'author': poem['author'],
                'style': poem['style'],
                'id': len(expanded)
            })
    
    print(f"  ✓ Created {len(expanded)} poem samples")
    return expanded


def create_combined_dataset(output_dir: Path) -> Dict:
    """Create a combined dataset from all sources."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_poems = []
    
    # Download/create each dataset
    shakespeare = download_shakespeare_sonnets(output_dir)
    all_poems.extend(shakespeare)
    
    poetry_foundation = download_poetry_foundation_sample(output_dir)
    all_poems.extend(poetry_foundation)
    
    # Shuffle
    import random
    random.shuffle(all_poems)
    
    # Split into train/val/test
    n = len(all_poems)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    train_data = all_poems[:train_size]
    val_data = all_poems[train_size:train_size + val_size]
    test_data = all_poems[train_size + val_size:]
    
    # Save datasets
    processed_dir = output_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    with open(processed_dir / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(processed_dir / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    with open(processed_dir / 'test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    stats = {
        'total': n,
        'train': len(train_data),
        'val': len(val_data),
        'test': len(test_data),
        'styles': list(set(p['style'] for p in all_poems)),
        'authors': list(set(p.get('author', 'Unknown') for p in all_poems))
    }
    
    with open(processed_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main():
    print("=" * 60)
    print("📥 Poem Dataset Downloader")
    print("=" * 60)
    
    # Create data directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    # Download and combine datasets
    stats = create_combined_dataset(data_dir)
    
    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print("=" * 60)
    print(f"\nTotal poems: {stats['total']}")
    print(f"  Train: {stats['train']}")
    print(f"  Val: {stats['val']}")
    print(f"  Test: {stats['test']}")
    print(f"\nStyles: {len(stats['styles'])}")
    print(f"Authors: {len(stats['authors'])}")
    print(f"\nData saved to: {data_dir / 'processed'}/")


if __name__ == '__main__':
    main()
