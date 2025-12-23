#!/usr/bin/env python
"""
విస్తృత తెలుగు కవితా డేటాసెట్ - Large Scale (520+ poems)
Comprehensive Telugu Poem Dataset Generator
"""

import json
import random
from pathlib import Path

# Import all dataset parts
from dataset_part1 import VEMANA_POEMS, SUMATI_SATAKAM
from dataset_part2 import ANNAMAYYA_KEERTANAS, FOLK_SONGS, POTHANA_BHAGAVATAM
from dataset_part3 import MODERN_POETRY, NEETI_POEMS
from dataset_part4 import SRINATHA_POEMS, TYAGARAJA_KEERTANAS, GURAJADA_POEMS
from dataset_part5 import TIKKANA_POEMS, RAMADASU_KEERTANAS, KRISHNASASTRI_POEMS, CHILAKAMARTHI_POEMS
from dataset_part6 import BHARTRUHARI_POEMS, NANNAYA_POEMS, SRISRI_POEMS


def create_large_telugu_dataset():
    """Create large-scale Telugu dataset with 520+ poems."""
    
    print("=" * 70)
    print("📥 విస్తృత తెలుగు కవితా డేటాసెట్ (Large-Scale Telugu Dataset)")
    print("=" * 70)
    
    all_poems = []
    
    # Define categories with metadata
    categories = [
        (VEMANA_POEMS, 'వేమన పద్యం', 'వేమన', 'ఆట వెలది', '18వ శతాబ్దం'),
        (SUMATI_SATAKAM, 'సుమతీ శతకం', 'బద్దెన', 'కందం', '14వ శతాబ్దం'),
        (ANNAMAYYA_KEERTANAS, 'అన్నమయ్య కీర్తన', 'అన్నమయ్య', 'సంకీర్తన', '15వ శతాబ్దం'),
        (FOLK_SONGS, 'జానపద గేయం', 'జానపద', 'గేయం', 'సంప్రదాయ'),
        (POTHANA_BHAGAVATAM, 'భాగవత పద్యం', 'పోతన', 'ఉత్పలమాల', '15వ శతాబ్దం'),
        (MODERN_POETRY, 'ఆధునిక కవిత', 'ఆధునిక కవి', 'వచన కవిత', '21వ శతాబ్దం'),
        (NEETI_POEMS, 'నీతి పద్యం', 'సంప్రదాయ', 'నీతి శతకం', 'సంప్రదాయ'),
        (SRINATHA_POEMS, 'శ్రీనాథ పద్యం', 'శ్రీనాథుడు', 'ప్రబంధం', '15వ శతాబ్దం'),
        (TYAGARAJA_KEERTANAS, 'త్యాగరాజ కీర్తన', 'త్యాగరాజు', 'కర్ణాటక సంగీతం', '18వ శతాబ్దం'),
        (GURAJADA_POEMS, 'గురజాడ కవిత', 'గురజాడ అప్పారావు', 'సామాజిక కవిత్వం', '20వ శతాబ్దం'),
        (TIKKANA_POEMS, 'తిక్కన పద్యం', 'తిక్కన', 'మహాభారతం', '13వ శతాబ్దం'),
        (RAMADASU_KEERTANAS, 'రామదాసు కీర్తన', 'రామదాసు', 'భక్తి కీర్తన', '17వ శతాబ్దం'),
        (KRISHNASASTRI_POEMS, 'కృష్ణశాస్త్రి కవిత', 'దేవులపల్లి కృష్ణశాస్త్రి', 'భావ కవిత్వం', '20వ శతాబ్దం'),
        (CHILAKAMARTHI_POEMS, 'చిలకమర్తి రచన', 'చిలకమర్తి', 'హాస్య కవిత్వం', '20వ శతాబ్దం'),
        (BHARTRUHARI_POEMS, 'భర్తృహరి సుభాషితం', 'భర్తృహరి', 'సుభాషిత శతకం', 'ప్రాచీన'),
        (NANNAYA_POEMS, 'నన్నయ పద్యం', 'నన్నయ', 'మహాభారతం', '11వ శతాబ్దం'),
        (SRISRI_POEMS, 'శ్రీశ్రీ కవిత', 'శ్రీశ్రీ', 'అభ్యుదయ కవిత్వం', '20వ శతాబ్దం'),
    ]
    
    # Process each category
    for poems, title_prefix, author, style, era in categories:
        print(f"\n📚 {title_prefix}: {len(poems)} poems")
        for i, text in enumerate(poems):
            all_poems.append({
                'text': text.strip(),
                'title': f'{title_prefix} {i+1}',
                'author': author,
                'style': style,
                'era': era,
                'language': 'telugu'
            })
    
    # Save dataset
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split (80/10/10)
    random.seed(42)
    random.shuffle(all_poems)
    
    n = len(all_poems)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    train_data = all_poems[:train_size]
    val_data = all_poems[train_size:train_size + val_size]
    test_data = all_poems[train_size + val_size:]
    
    # Save files
    with open(output_dir / 'telugu_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / 'telugu_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / 'telugu_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / 'telugu_poems.json', 'w', encoding='utf-8') as f:
        json.dump(all_poems, f, ensure_ascii=False, indent=2)
    
    # Collect stats
    styles = list(set(p['style'] for p in all_poems))
    authors = list(set(p['author'] for p in all_poems))
    eras = list(set(p['era'] for p in all_poems))
    
    stats = {
        'total_poems': len(all_poems),
        'train_poems': len(train_data),
        'val_poems': len(val_data),
        'test_poems': len(test_data),
        'styles': styles,
        'authors': authors,
        'eras': eras,
        'language': 'telugu'
    }
    
    with open(output_dir / 'telugu_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ తెలుగు డేటాసెట్ సృష్టించబడింది! (Telugu Dataset Created!)")
    print("=" * 70)
    print(f"\n📊 మొత్తం కవితలు (Total Poems): {stats['total_poems']}")
    print(f"   శిక్షణ (Train): {stats['train_poems']}")
    print(f"   ధృవీకరణ (Validation): {stats['val_poems']}")
    print(f"   పరీక్ష (Test): {stats['test_poems']}")
    print(f"\n📝 శైలులు (Styles): {len(styles)}")
    for s in sorted(styles):
        print(f"   • {s}")
    print(f"\n✍️ కవులు (Authors): {len(authors)}")
    for a in sorted(authors):
        print(f"   • {a}")
    print(f"\n📅 యుగాలు (Eras): {len(eras)}")
    for e in sorted(eras):
        print(f"   • {e}")
    
    print(f"\n📁 Output directory: {output_dir}")
    return all_poems


if __name__ == "__main__":
    create_large_telugu_dataset()
