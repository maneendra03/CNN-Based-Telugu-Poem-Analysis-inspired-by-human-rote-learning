#!/usr/bin/env python3
"""
Telugu Poetry Dataset Downloader & Merger
==========================================
Downloads and merges Telugu poems from multiple sources to create a large dataset (10,000+ poems).

Sources:
1. HuggingFace: SuryaKrishna02/aya-telugu-poems (~5,115 poems)
2. Kaggle: boddusripavan111/chandassu (~4,651 poems with prosodic annotations)
3. Local: Existing poems from data/processed/telugu_poems.json
4. Generated: Poems from scripts/dataset_part1-6.py

Usage:
    python scripts/download_datasets.py                    # Download all sources
    python scripts/download_datasets.py --skip-kaggle      # Skip Kaggle (no API key)
    python scripts/download_datasets.py --include-generated # Include generated poems from dataset_part files
"""

import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging():
    """Simple logging setup"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()


def check_dependencies():
    """Check if required packages are installed"""
    missing = []
    
    try:
        import datasets
    except ImportError:
        missing.append('datasets')
    
    try:
        import tqdm
    except ImportError:
        missing.append('tqdm')
    
    try:
        import pandas
    except ImportError:
        missing.append('pandas')
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def is_telugu_char(char: str) -> bool:
    """Check if character is Telugu (Unicode range: 0C00-0C7F)"""
    if len(char) != 1:
        return False
    code = ord(char)
    return 0x0C00 <= code <= 0x0C7F


def extract_telugu_text(text: str) -> str:
    """Extract only Telugu characters and essential punctuation"""
    if not text:
        return ""
    
    result = []
    for char in text:
        if is_telugu_char(char):
            result.append(char)
        elif char in ' \n।,.!?':
            result.append(char)
    
    return ''.join(result).strip()


def count_telugu_chars(text: str) -> int:
    """Count Telugu characters in text"""
    return sum(1 for c in text if is_telugu_char(c))


def normalize_for_comparison(text: str) -> str:
    """Normalize text for duplicate comparison"""
    if not text:
        return ""
    
    # Extract only Telugu characters (no spaces or punctuation)
    telugu_only = ''.join(c for c in text if is_telugu_char(c))
    return telugu_only


def get_poem_signature(text: str, length: int = 100) -> str:
    """
    Get a signature for poem comparison.
    Uses first N Telugu characters as the signature.
    This avoids the hash collision issue from before.
    """
    normalized = normalize_for_comparison(text)
    return normalized[:length] if len(normalized) >= length else normalized


def clean_poem_text(text: str) -> str:
    """Clean and normalize poem text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Normalize line breaks (keep structure)
    text = re.sub(r'\s*\n\s*', '\n', text)
    
    return text


# ============================================================================
# HuggingFace Dataset Loader
# ============================================================================

def download_huggingface_dataset() -> List[Dict]:
    """
    Download Telugu poems from multiple HuggingFace datasets.
    """
    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        logger.error("❌ Please install: pip install datasets tqdm")
        return []
    
    all_poems = []
    seen_poems = set()
    
    # Dataset 1: SuryaKrishna02/aya-telugu-poems (~5,115 poems)
    logger.info("📥 Downloading: SuryaKrishna02/aya-telugu-poems")
    try:
        dataset = load_dataset("SuryaKrishna02/aya-telugu-poems", split="train")
        logger.info(f"   ✓ Loaded {len(dataset)} records")
        
        for item in tqdm(dataset, desc="Processing aya-telugu-poems"):
            # The poem is in the 'inputs' field (the question contains the poem)
            # 'targets' is just the explanation
            input_text = item.get('inputs', '') or ''
            
            # Extract the poem from the input
            # Format: "క్రింద ఇచ్చిన XXX శతకంలోని పద్యానికి ...: \n<POEM>"
            # The poem is typically after the colon
            if ':' in input_text:
                parts = input_text.split(':', 1)
                if len(parts) > 1:
                    poem_text = parts[1].strip()
                else:
                    poem_text = input_text
            else:
                poem_text = input_text
            
            if not poem_text:
                continue
            
            poem_text = clean_poem_text(poem_text)
            telugu_count = count_telugu_chars(poem_text)
            if telugu_count < 20:
                continue
            
            signature = get_poem_signature(poem_text, length=80)
            if signature in seen_poems:
                continue
            seen_poems.add(signature)
            
            # Extract śatakam name from input
            source = "HuggingFace-aya"
            if 'వేమన' in input_text:
                source = "వేమన శతకం"
            elif 'సుమతీ' in input_text:
                source = "సుమతీ శతకం"
            elif 'భర్తృహరి' in input_text:
                source = "భర్తృహరి శతకం"
            elif 'కుమార' in input_text:
                source = "కుమార శతకం"
            elif 'నీతి' in input_text:
                source = "నీతి శతకం"
            elif 'భాస్కర' in input_text:
                source = "భాస్కర శతకం"
            elif 'దాశరథి' in input_text:
                source = "దాశరథి శతకం"
            elif 'కాళహస్తీశ్వర' in input_text:
                source = "కాళహస్తీశ్వర శతకం"
            elif 'నరసింహ' in input_text:
                source = "నరసింహ శతకం"
            
            all_poems.append({
                'text': poem_text,
                'source': source,
                'dataset': 'huggingface-aya',
                'telugu_char_count': telugu_count
            })
    except Exception as e:
        logger.warning(f"   ⚠️ Failed to load aya-telugu-poems: {e}")
    
    logger.info(f"   ✓ Extracted {len(all_poems)} poems so far")
    
    # Dataset 2: AravindGillella/telugu-personal-poems-tinker
    logger.info("📥 Downloading: AravindGillella/telugu-personal-poems-tinker")
    try:
        dataset2 = load_dataset("AravindGillella/telugu-personal-poems-tinker", split="train")
        logger.info(f"   ✓ Loaded {len(dataset2)} records")
        
        count_before = len(all_poems)
        for item in tqdm(dataset2, desc="Processing personal-poems"):
            # Try different column names
            poem_text = None
            for col in ['poem', 'text', 'content', 'output', 'targets']:
                if col in item and item[col]:
                    poem_text = item[col]
                    break
            
            if not poem_text:
                # Try first column with Telugu text
                for key, val in item.items():
                    if val and isinstance(val, str) and count_telugu_chars(val) > 20:
                        poem_text = val
                        break
            
            if not poem_text:
                continue
            
            poem_text = clean_poem_text(poem_text)
            telugu_count = count_telugu_chars(poem_text)
            if telugu_count < 20:
                continue
            
            signature = get_poem_signature(poem_text, length=80)
            if signature in seen_poems:
                continue
            seen_poems.add(signature)
            
            all_poems.append({
                'text': poem_text,
                'source': 'personal-poems',
                'dataset': 'huggingface-personal',
                'telugu_char_count': telugu_count
            })
        
        logger.info(f"   ✓ Added {len(all_poems) - count_before} new poems")
    except Exception as e:
        logger.warning(f"   ⚠️ Failed to load personal-poems: {e}")
    
    # Dataset 3: community-datasets/telugu_books (extract poetry-like content)
    logger.info("📥 Downloading: community-datasets/telugu_books")
    try:
        dataset3 = load_dataset("community-datasets/telugu_books", split="train")
        logger.info(f"   ✓ Loaded {len(dataset3)} records")
        
        count_before = len(all_poems)
        for item in tqdm(dataset3, desc="Processing telugu_books"):
            text = item.get('text', '') or item.get('content', '')
            if not text:
                continue
            
            # Split into potential poems (paragraphs)
            paragraphs = text.split('\n\n')
            
            for para in paragraphs:
                para = clean_poem_text(para)
                telugu_count = count_telugu_chars(para)
                
                # Only take short poetic paragraphs (likely poems)
                if telugu_count < 30 or telugu_count > 500:
                    continue
                
                # Check if it looks like a poem (has line breaks or is short)
                lines = para.strip().split('\n')
                if len(lines) < 2 or len(lines) > 10:
                    continue
                
                signature = get_poem_signature(para, length=80)
                if signature in seen_poems:
                    continue
                seen_poems.add(signature)
                
                all_poems.append({
                    'text': para,
                    'source': 'telugu-books',
                    'dataset': 'huggingface-books',
                    'telugu_char_count': telugu_count
                })
        
        logger.info(f"   ✓ Added {len(all_poems) - count_before} new poems")
    except Exception as e:
        logger.warning(f"   ⚠️ Failed to load telugu_books: {e}")
    
    # Dataset 4: Try to find more Telugu literary datasets
    additional_datasets = [
        ("Suchinthana/Telugu-Poems", "train"),
        ("Telugu-LLM-Labs/telugu_poems", "train"),
        ("indicnlp/telugu-poetry", "train"),
    ]
    
    for ds_name, split in additional_datasets:
        logger.info(f"📥 Trying: {ds_name}")
        try:
            ds = load_dataset(ds_name, split=split)
            logger.info(f"   ✓ Loaded {len(ds)} records")
            
            count_before = len(all_poems)
            for item in tqdm(ds, desc=f"Processing {ds_name.split('/')[-1]}"):
                poem_text = None
                for col in ['poem', 'text', 'content', 'output', 'targets', 'verse']:
                    if col in item and item[col]:
                        poem_text = item[col]
                        break
                
                if not poem_text:
                    for key, val in item.items():
                        if val and isinstance(val, str) and count_telugu_chars(val) > 20:
                            poem_text = val
                            break
                
                if not poem_text:
                    continue
                
                poem_text = clean_poem_text(poem_text)
                telugu_count = count_telugu_chars(poem_text)
                if telugu_count < 20:
                    continue
                
                signature = get_poem_signature(poem_text, length=80)
                if signature in seen_poems:
                    continue
                seen_poems.add(signature)
                
                all_poems.append({
                    'text': poem_text,
                    'source': ds_name.split('/')[-1],
                    'dataset': f'huggingface-{ds_name.split("/")[-1]}',
                    'telugu_char_count': telugu_count
                })
            
            logger.info(f"   ✓ Added {len(all_poems) - count_before} new poems")
        except Exception as e:
            logger.info(f"   ℹ️ Dataset not available: {ds_name}")
    
    logger.info(f"\n✓ Total: Extracted {len(all_poems)} unique poems from HuggingFace")
    return all_poems


# ============================================================================
# Kaggle Dataset Loader
# ============================================================================

def download_kaggle_dataset() -> List[Dict]:
    """
    Download Telugu poems from Kaggle dataset.
    Dataset: boddusripavan111/chandassu
    Contains ~4,651 poems with prosodic (chandas) annotations.
    
    Requires Kaggle API credentials in ~/.kaggle/kaggle.json
    """
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.warning("⚠️  Kaggle package not installed. Install with: pip install kaggle")
        return []
    
    # Check for Kaggle credentials
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        logger.warning("⚠️  Kaggle credentials not found at ~/.kaggle/kaggle.json")
        logger.info("   To use Kaggle datasets:")
        logger.info("   1. Go to kaggle.com → Account → Create API Token")
        logger.info("   2. Save kaggle.json to ~/.kaggle/")
        logger.info("   3. Run: chmod 600 ~/.kaggle/kaggle.json")
        return []
    
    logger.info("📥 Downloading Kaggle dataset: boddusripavan111/chandassu")
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        # Download to temp directory
        download_dir = project_root / 'data' / 'raw' / 'kaggle'
        download_dir.mkdir(parents=True, exist_ok=True)
        
        api.dataset_download_files(
            'boddusripavan111/chandassu',
            path=str(download_dir),
            unzip=True
        )
        logger.info(f"✓ Downloaded to {download_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to download Kaggle dataset: {e}")
        return []
    
    # Find and parse the CSV file
    poems = []
    seen_poems = set()
    
    try:
        import pandas as pd
        from tqdm import tqdm
        
        # Look for CSV files
        csv_files = list(download_dir.glob('*.csv'))
        
        for csv_file in csv_files:
            logger.info(f"   Processing: {csv_file.name}")
            
            try:
                df = pd.read_csv(csv_file)
                
                # Common column names for poem text
                text_columns = ['poem', 'text', 'padyam', 'పద్యం', 'content', 'verse']
                text_col = None
                
                for col in text_columns:
                    if col in df.columns:
                        text_col = col
                        break
                
                if text_col is None:
                    # Use first column that has Telugu text
                    for col in df.columns:
                        sample = str(df[col].iloc[0]) if len(df) > 0 else ""
                        if count_telugu_chars(sample) > 10:
                            text_col = col
                            break
                
                if text_col is None:
                    logger.warning(f"   ⚠️  Could not find poem column in {csv_file.name}")
                    continue
                
                # Extract chandas (meter) info if available
                chandas_columns = ['chandas', 'छन्दस्', 'meter', 'ఛందస్సు', 'type']
                chandas_col = None
                for col in chandas_columns:
                    if col in df.columns:
                        chandas_col = col
                        break
                
                for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_file.name}"):
                    poem_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                    poem_text = clean_poem_text(poem_text)
                    
                    telugu_count = count_telugu_chars(poem_text)
                    if telugu_count < 20:
                        continue
                    
                    signature = get_poem_signature(poem_text, length=80)
                    if signature in seen_poems:
                        continue
                    seen_poems.add(signature)
                    
                    chandas = str(row[chandas_col]) if chandas_col and pd.notna(row.get(chandas_col)) else "unknown"
                    
                    poems.append({
                        'text': poem_text,
                        'source': 'Kaggle-Chandassu',
                        'dataset': 'kaggle',
                        'chandas': chandas,
                        'telugu_char_count': telugu_count
                    })
                
            except Exception as e:
                logger.warning(f"   ⚠️  Error processing {csv_file.name}: {e}")
                continue
        
    except ImportError:
        logger.error("❌ Please install pandas: pip install pandas")
        return []
    
    logger.info(f"✓ Extracted {len(poems)} unique poems from Kaggle dataset")
    return poems


# ============================================================================
# Local Dataset Loader
# ============================================================================

def load_chandassu_csv() -> List[Dict]:
    """Load poems from local Chandassu_Dataset.csv"""
    csv_file = project_root / 'data' / 'Chandassu_Dataset.csv'
    
    if not csv_file.exists():
        logger.info("ℹ️  No Chandassu_Dataset.csv found")
        return []
    
    try:
        import pandas as pd
        from tqdm import tqdm
        
        df = pd.read_csv(csv_file)
        logger.info(f"✓ Loaded {len(df)} records from Chandassu_Dataset.csv")
        
        poems = []
        seen = set()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Chandassu"):
            # Get poem text from 'raw_padyam_text' column
            poem_text = str(row.get('raw_padyam_text', '')) if pd.notna(row.get('raw_padyam_text')) else ''
            poem_text = clean_poem_text(poem_text)
            
            telugu_count = count_telugu_chars(poem_text)
            if telugu_count < 20:
                continue
            
            sig = get_poem_signature(poem_text, 80)
            if sig in seen:
                continue
            seen.add(sig)
            
            # Get metadata
            chandas_type = str(row.get('type', 'unknown')) if pd.notna(row.get('type')) else 'unknown'
            satakam = str(row.get('satakam', 'Chandassu')) if pd.notna(row.get('satakam')) else 'Chandassu'
            poem_class = str(row.get('class', '')) if pd.notna(row.get('class')) else ''
            
            poems.append({
                'text': poem_text,
                'source': satakam,
                'dataset': 'chandassu-local',
                'chandas': chandas_type,
                'class': poem_class,
                'telugu_char_count': telugu_count
            })
        
        logger.info(f"✓ Extracted {len(poems)} unique poems from Chandassu_Dataset.csv")
        return poems
        
    except Exception as e:
        logger.error(f"❌ Error loading Chandassu_Dataset.csv: {e}")
        return []


def load_existing_poems() -> List[Dict]:
    """Load existing poems from data/processed/telugu_poems.json"""
    poems_file = project_root / 'data' / 'processed' / 'telugu_poems.json'
    
    if not poems_file.exists():
        logger.info("ℹ️  No existing poems file found")
        return []
    
    try:
        with open(poems_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        poems = []
        seen = set()
        
        items = data if isinstance(data, list) else data.get('poems', [])
        
        for item in items:
            text = item.get('text', '') or item.get('poem', '')
            text = clean_poem_text(text)
            
            if count_telugu_chars(text) < 20:
                continue
            
            sig = get_poem_signature(text, 80)
            if sig in seen:
                continue
            seen.add(sig)
            
            poems.append({
                'text': text,
                'source': item.get('source', 'local'),
                'dataset': 'local',
                'telugu_char_count': count_telugu_chars(text)
            })
        
        logger.info(f"✓ Loaded {len(poems)} existing poems from local file")
        return poems
        
    except Exception as e:
        logger.error(f"❌ Error loading existing poems: {e}")
        return []


def load_generated_poems() -> List[Dict]:
    """Load poems from dataset_part1-6.py files"""
    poems = []
    seen = set()
    
    def is_valid_poem(text: str) -> bool:
        """Check if text looks like a valid poem"""
        # Must have minimum Telugu chars
        if count_telugu_chars(text) < 40:
            return False
        # Must have line breaks (poems have verses)
        if '\n' not in text:
            return False
        # Shouldn't have Python code
        if 'import ' in text or 'def ' in text or '= [' in text or 'Path(' in text:
            return False
        # Shouldn't be too long (prose)
        if len(text) > 1000:
            return False
        return True
    
    for i in range(1, 7):
        part_file = project_root / 'scripts' / f'dataset_part{i}.py'
        if not part_file.exists():
            continue
        
        try:
            # Read the file content
            with open(part_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find poem strings - they're between quotes and contain \n
            # Pattern matches: "poem text with \n" 
            pattern = r'"((?:[^"\\]|\\.)*)"'
            matches = re.findall(pattern, content)
            
            count_before = len(poems)
            for match in matches:
                # Unescape newlines
                text = match.replace('\\n', '\n')
                text = clean_poem_text(text)
                
                if not is_valid_poem(text):
                    continue
                
                telugu_count = count_telugu_chars(text)
                
                sig = get_poem_signature(text, 80)
                if sig in seen:
                    continue
                seen.add(sig)
                
                poems.append({
                    'text': text,
                    'source': f'generated-part{i}',
                    'dataset': 'generated',
                    'telugu_char_count': telugu_count
                })
            
            part_count = len(poems) - count_before
            if part_count > 0:
                logger.info(f"   Part {i}: {part_count} poems")
        
        except Exception as e:
            logger.warning(f"⚠️  Error loading {part_file.name}: {e}")
            continue
    
    if poems:
        logger.info(f"✓ Loaded {len(poems)} poems from generated dataset files")
    else:
        logger.info("ℹ️  No poems extracted from generated dataset files")
    
    return poems


# ============================================================================
# Dataset Augmentation - Generate more poems to reach 10,000+
# ============================================================================

def augment_with_variations(poems: List[Dict], target_count: int = 10000) -> List[Dict]:
    """
    Augment the dataset by creating meaningful variations of existing poems.
    This helps reach the 10,000+ target while maintaining quality.
    
    Techniques used:
    1. Line reordering for multi-line poems
    2. Combining half-verses from different poems of same śatakam
    3. Creating verse fragments as standalone poems
    """
    import random
    random.seed(42)  # Reproducibility
    
    if len(poems) >= target_count:
        logger.info(f"✓ Already have {len(poems)} poems (target: {target_count})")
        return poems
    
    augmented = poems.copy()
    seen_signatures = {get_poem_signature(p['text'], 60) for p in poems}  # Shorter signature for variations
    
    # Group poems by source (śatakam)
    poems_by_source = defaultdict(list)
    for p in poems:
        poems_by_source[p.get('source', 'unknown')].append(p)
    
    logger.info(f"   Sources for augmentation: {list(poems_by_source.keys())}")
    
    attempts = 0
    max_attempts = (target_count - len(poems)) * 5
    created = 0
    
    while len(augmented) < target_count and attempts < max_attempts:
        attempts += 1
        
        # Pick a random source poem
        source_poem = random.choice(poems)
        source_text = source_poem.get('text', '')
        
        # Split by newline or common verse separators
        if '\n' in source_text:
            lines = source_text.strip().split('\n')
        else:
            # Try splitting by Telugu punctuation
            lines = re.split(r'[।॥,]', source_text)
            lines = [l.strip() for l in lines if l.strip() and count_telugu_chars(l) > 10]
        
        if len(lines) < 2:
            continue
        
        new_text = None
        
        # Strategy 1: Reverse line order (but keep ending marker at end)
        if len(lines) >= 3:
            # Check if last line is a signature (short ending like "వినురవేమ")
            if count_telugu_chars(lines[-1]) < 30:
                # Reverse middle lines, keep first and last
                middle = lines[1:-1]
                random.shuffle(middle)
                new_lines = [lines[0]] + middle + [lines[-1]]
            else:
                # Just shuffle all lines
                new_lines = lines.copy()
                random.shuffle(new_lines)
            
            new_text = '\n'.join(new_lines)
        
        # Strategy 2: Take subset of lines
        elif len(lines) >= 4 and random.random() > 0.5:
            # Take 2-3 lines from the poem
            start = random.randint(0, len(lines) - 2)
            end = min(start + random.randint(2, 3), len(lines))
            new_text = '\n'.join(lines[start:end])
        
        # Strategy 3: Combine lines from two poems of same source
        if new_text is None and len(lines) >= 2:
            source_type = source_poem.get('source', 'unknown')
            same_source_poems = poems_by_source.get(source_type, [])
            
            if len(same_source_poems) > 1:
                other_poem = random.choice(same_source_poems)
                if other_poem['text'] != source_text:
                    other_text = other_poem.get('text', '')
                    if '\n' in other_text:
                        other_lines = other_text.strip().split('\n')
                    else:
                        other_lines = re.split(r'[।॥,]', other_text)
                        other_lines = [l.strip() for l in other_lines if l.strip() and count_telugu_chars(l) > 10]
                    
                    if len(other_lines) >= 2:
                        # Take half from each
                        half1 = lines[:len(lines)//2]
                        half2 = other_lines[len(other_lines)//2:]
                        new_text = '\n'.join(half1 + half2)
        
        if new_text:
            new_text = clean_poem_text(new_text)
            telugu_count = count_telugu_chars(new_text)
            
            if telugu_count >= 30:
                sig = get_poem_signature(new_text, 60)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    augmented.append({
                        'text': new_text,
                        'source': source_poem.get('source', 'unknown') + '-augmented',
                        'dataset': 'augmented',
                        'telugu_char_count': telugu_count
                    })
                    created += 1
    
    logger.info(f"✓ Augmented: {len(poems)} → {len(augmented)} poems (+{created} new)")
    return augmented


# ============================================================================
# Dataset Merger & Deduplication
# ============================================================================

def deduplicate_poems(poems: List[Dict], signature_length: int = 80) -> List[Dict]:
    """
    Deduplicate poems using content-based signatures.
    
    Uses first N Telugu characters as signature to detect duplicates.
    This is more robust than exact hash matching.
    
    Args:
        poems: List of poem dictionaries
        signature_length: Number of Telugu characters to use for signature
    
    Returns:
        Deduplicated list of poems
    """
    seen_signatures = {}
    unique_poems = []
    duplicates = 0
    
    for poem in poems:
        text = poem.get('text', '')
        signature = get_poem_signature(text, signature_length)
        
        if not signature:
            continue
        
        if signature in seen_signatures:
            duplicates += 1
            # Keep the one with more metadata or longer text
            existing_idx = seen_signatures[signature]
            existing = unique_poems[existing_idx]
            
            # Prefer poems with chandas info or longer text
            if poem.get('chandas') and not existing.get('chandas'):
                unique_poems[existing_idx] = poem
            elif len(text) > len(existing.get('text', '')):
                unique_poems[existing_idx] = poem
        else:
            seen_signatures[signature] = len(unique_poems)
            unique_poems.append(poem)
    
    logger.info(f"🔍 Deduplication: {len(poems)} → {len(unique_poems)} poems ({duplicates} duplicates removed)")
    return unique_poems


def merge_datasets(*datasets: List[Dict]) -> List[Dict]:
    """Merge multiple poem datasets and deduplicate"""
    all_poems = []
    
    for dataset in datasets:
        all_poems.extend(dataset)
    
    logger.info(f"📊 Total poems before deduplication: {len(all_poems)}")
    
    # Deduplicate across all sources
    unique_poems = deduplicate_poems(all_poems)
    
    return unique_poems


# ============================================================================
# Dataset Statistics & Validation
# ============================================================================

def compute_statistics(poems: List[Dict]) -> Dict:
    """Compute dataset statistics"""
    if not poems:
        return {}
    
    # Source distribution
    sources = defaultdict(int)
    datasets = defaultdict(int)
    
    # Length statistics
    lengths = []
    telugu_counts = []
    
    for poem in poems:
        sources[poem.get('source', 'unknown')] += 1
        datasets[poem.get('dataset', 'unknown')] += 1
        lengths.append(len(poem.get('text', '')))
        telugu_counts.append(poem.get('telugu_char_count', 0))
    
    stats = {
        'total_poems': len(poems),
        'sources': dict(sources),
        'datasets': dict(datasets),
        'avg_length': sum(lengths) / len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'avg_telugu_chars': sum(telugu_counts) / len(telugu_counts),
        'total_telugu_chars': sum(telugu_counts)
    }
    
    return stats


def print_statistics(stats: Dict):
    """Print dataset statistics"""
    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    print(f"Total poems: {stats['total_poems']:,}")
    print(f"Average length: {stats['avg_length']:.0f} characters")
    print(f"Total Telugu characters: {stats['total_telugu_chars']:,}")
    print()
    print("By Dataset Source:")
    for ds, count in sorted(stats['datasets'].items(), key=lambda x: -x[1]):
        print(f"  • {ds}: {count:,} poems")
    print()
    print("By Original Source:")
    for src, count in sorted(stats['sources'].items(), key=lambda x: -x[1])[:10]:
        print(f"  • {src}: {count:,} poems")
    print("="*60 + "\n")


# ============================================================================
# Save Functions
# ============================================================================

def save_dataset(poems: List[Dict], output_path: Path, split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
    """
    Save the merged dataset with train/val/test splits.
    
    Args:
        poems: List of poem dictionaries
        output_path: Directory to save the files
        split_ratio: (train, val, test) ratios
    """
    import random
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Shuffle poems
    poems_shuffled = poems.copy()
    random.seed(42)
    random.shuffle(poems_shuffled)
    
    # Calculate split indices
    n = len(poems_shuffled)
    train_end = int(n * split_ratio[0])
    val_end = train_end + int(n * split_ratio[1])
    
    train_poems = poems_shuffled[:train_end]
    val_poems = poems_shuffled[train_end:val_end]
    test_poems = poems_shuffled[val_end:]
    
    # Save files
    files = {
        'telugu_poems.json': poems,
        'telugu_train.json': train_poems,
        'telugu_val.json': val_poems,
        'telugu_test.json': test_poems
    }
    
    for filename, data in files.items():
        filepath = output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved {len(data):,} poems to {filename}")
    
    # Save statistics
    stats = compute_statistics(poems)
    stats['splits'] = {
        'train': len(train_poems),
        'val': len(val_poems),
        'test': len(test_poems)
    }
    
    stats_path = output_path / 'telugu_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return stats


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Download and merge Telugu poetry datasets (10,000+ poems)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--skip-kaggle', 
        action='store_true',
        help='Skip Kaggle dataset (if no API credentials)'
    )
    parser.add_argument(
        '--skip-huggingface',
        action='store_true',
        help='Skip HuggingFace dataset'
    )
    parser.add_argument(
        '--include-generated',
        action='store_true',
        help='Include poems from dataset_part1-6.py files'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Augment dataset to reach 10,000+ poems'
    )
    parser.add_argument(
        '--target-count',
        type=int,
        default=10000,
        help='Target number of poems (default: 10000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(project_root / 'data' / 'processed'),
        help='Output directory for processed data'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🌸 Telugu Poetry Dataset Downloader")
    print("   Target: 10,000+ unique poems")
    print("="*60 + "\n")
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    all_poems = []
    
    # 1. Load local Chandassu CSV (if available)
    logger.info("\n📚 Step 1: Loading local Chandassu dataset...")
    chandassu_poems = load_chandassu_csv()
    all_poems.extend(chandassu_poems)
    
    # 2. Load existing processed poems
    logger.info("\n📚 Step 2: Loading existing processed poems...")
    existing_poems = load_existing_poems()
    all_poems.extend(existing_poems)
    
    # 3. Download HuggingFace dataset
    if not args.skip_huggingface:
        logger.info("\n📚 Step 3: Downloading HuggingFace dataset...")
        hf_poems = download_huggingface_dataset()
        all_poems.extend(hf_poems)
    else:
        logger.info("\n⏭️  Skipping HuggingFace dataset")
    
    # 4. Download Kaggle dataset
    if not args.skip_kaggle:
        logger.info("\n📚 Step 4: Downloading Kaggle dataset...")
        kaggle_poems = download_kaggle_dataset()
        all_poems.extend(kaggle_poems)
    else:
        logger.info("\n⏭️  Skipping Kaggle dataset")
    
    # 5. Load generated poems (optional)
    if args.include_generated:
        logger.info("\n📚 Step 5: Loading generated poems from dataset_part files...")
        generated_poems = load_generated_poems()
        all_poems.extend(generated_poems)
    
    # 5. Merge and deduplicate
    logger.info("\n🔄 Merging and deduplicating all sources...")
    unique_poems = deduplicate_poems(all_poems)
    
    if not unique_poems:
        logger.error("❌ No poems collected! Please check your internet connection.")
        return 1
    
    # 6. Augment if requested
    if args.augment:
        logger.info(f"\n🔄 Augmenting dataset to reach {args.target_count:,} poems...")
        unique_poems = augment_with_variations(unique_poems, args.target_count)
    
    # 7. Compute and print statistics
    stats = compute_statistics(unique_poems)
    print_statistics(stats)
    
    # 8. Save dataset
    output_dir = Path(args.output_dir)
    logger.info(f"\n💾 Saving dataset to {output_dir}...")
    save_dataset(unique_poems, output_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"📊 Total unique poems: {len(unique_poems):,}")
    print(f"📁 Output directory: {output_dir}")
    print("\nFiles created:")
    print("  • telugu_poems.json - All poems")
    print("  • telugu_train.json - Training set (80%)")
    print("  • telugu_val.json   - Validation set (10%)")
    print("  • telugu_test.json  - Test set (10%)")
    print("  • telugu_stats.json - Statistics")
    print("="*60 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
