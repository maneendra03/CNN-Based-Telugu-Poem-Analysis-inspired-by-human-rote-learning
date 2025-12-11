#!/usr/bin/env python
"""
Evaluation Script for CNN-Based Poem Learner
Usage: python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --data data/processed/test.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import json

from src.models.poem_learner import PoemLearner, create_poem_learner
from src.preprocessing.tokenizer import PoemTokenizer
from src.preprocessing.text_cleaner import TextCleaner
from src.evaluation.metrics import PoemMetrics, MemorizationCurve
from src.evaluation.visualizations import ResultsVisualizer
from src.utils.helpers import get_device


def main():
    parser = argparse.ArgumentParser(description='Evaluate Poem Learner')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='checkpoints/tokenizer.json',
                       help='Path to tokenizer')
    parser.add_argument('--data', type=str, default='data/processed/test.json',
                       help='Path to test data')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to generate for evaluation')
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = PoemTokenizer.load(args.tokenizer)
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    config['vocab_size'] = tokenizer.word_vocab_size
    
    model = create_poem_learner(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded with {model.count_parameters():,} parameters")
    
    # Load test data
    print("\nLoading test data...")
    test_data = []
    data_path = Path(args.data)
    
    if data_path.exists():
        with open(data_path) as f:
            if data_path.suffix == '.json':
                test_data = json.load(f)
            else:
                content = f.read()
                test_data = [{'text': p.strip()} for p in content.split('\n\n') if p.strip()]
    else:
        print("Test data not found, using sample data...")
        test_data = [
            {'text': "Shall I compare thee to a summer's day?\nThou art more lovely and more temperate."},
            {'text': "Two roads diverged in a yellow wood,\nAnd sorry I could not travel both."},
        ]
    
    # Clean data
    cleaner = TextCleaner()
    references = [cleaner.clean(d['text'] if isinstance(d, dict) else d) for d in test_data]
    
    print(f"Loaded {len(references)} test poems")
    
    # Generate poems
    print("\nGenerating poems for evaluation...")
    predictions = []
    
    for i in range(min(args.num_samples, len(references))):
        # Use first line as prompt
        ref = references[i]
        first_line = ref.split('\n')[0]
        
        input_ids = tokenizer.encode_words(first_line, max_length=50)
        input_ids = torch.tensor([input_ids], device=device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_length=100,
                start_token_id=tokenizer.word2idx[tokenizer.BOS_TOKEN],
                end_token_id=tokenizer.word2idx[tokenizer.EOS_TOKEN],
                temperature=0.8,
                top_k=50
            )
        
        generated_ids = output['generated_ids'][0].tolist()
        generated_text = tokenizer.decode_words(generated_ids)
        predictions.append(generated_text)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i+1}/{min(args.num_samples, len(references))} poems")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = PoemMetrics()
    metrics.update(predictions, references[:len(predictions)])
    results = metrics.compute_all()
    
    # Get memorization metrics
    mem_metrics = model.get_memorization_metrics()
    results.update(mem_metrics)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    for name, value in results.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, float) else v for k, v in results.items()}, f, indent=2)
    
    # Save sample generations
    with open(output_dir / 'generations.json', 'w') as f:
        json.dump({
            'predictions': predictions[:10],
            'references': references[:10]
        }, f, indent=2)
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizer = ResultsVisualizer(save_dir=str(output_dir / 'figures'))
    
    # If training history available
    if 'history' in checkpoint.get('metrics', {}):
        visualizer.plot_training_curves(checkpoint['metrics']['history'])
    
    # Metrics comparison (vs baseline)
    visualizer.plot_metrics_comparison({
        'PoemLearner': {k: v for k, v in results.items() if isinstance(v, float)}
    })
    
    print(f"\nResults saved to {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
