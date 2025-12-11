#!/usr/bin/env python
"""
Generation Script for CNN-Based Poem Learner
Usage: python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "roses are red"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.poem_learner import PoemLearner, create_poem_learner
from src.preprocessing.tokenizer import PoemTokenizer
from src.utils.helpers import get_device, load_config


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    config['vocab_size'] = config.get('vocab_size', 10000)
    
    model = create_poem_learner(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description='Generate poems')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='checkpoints/tokenizer.json',
                       help='Path to tokenizer')
    parser.add_argument('--prompt', type=str, default='',
                       help='Starting prompt for generation')
    parser.add_argument('--max-length', type=int, default=100,
                       help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Nucleus sampling threshold')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of poems to generate')
    parser.add_argument('--beam-size', type=int, default=0,
                       help='Beam size (0 for sampling)')
    parser.add_argument('--style', type=int, default=None,
                       help='Style ID to generate in')
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = PoemTokenizer.load(args.tokenizer)
    
    # Load model
    print("Loading model...")
    model, config = load_model(args.checkpoint, device)
    print(f"Model loaded with {model.count_parameters():,} parameters")
    
    # Prepare prompt
    if args.prompt:
        input_ids = tokenizer.encode_words(args.prompt, max_length=50)
        input_ids = torch.tensor([input_ids], device=device)
    else:
        # Start with just BOS token
        input_ids = torch.tensor([[tokenizer.word2idx[tokenizer.BOS_TOKEN]]], device=device)
    
    # Style conditioning
    style_id = None
    if args.style is not None:
        style_id = torch.tensor([args.style], device=device)
    
    print("\n" + "=" * 60)
    print("Generating Poems...")
    print("=" * 60)
    
    for i in range(args.num_samples):
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_length=args.max_length,
                start_token_id=tokenizer.word2idx[tokenizer.BOS_TOKEN],
                end_token_id=tokenizer.word2idx[tokenizer.EOS_TOKEN],
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                style_id=style_id,
                beam_size=args.beam_size
            )
        
        # Decode
        generated_ids = output['generated_ids'][0].tolist()
        generated_text = tokenizer.decode_words(generated_ids)
        
        print(f"\n--- Poem {i+1} ---")
        if args.prompt:
            print(f"[Prompt: {args.prompt}]\n")
        print(generated_text)
        
        # Print memorization metrics
        if i == 0:
            metrics = model.get_memorization_metrics()
            if metrics:
                print(f"\n[Memory Retention Score: {metrics.get('retention_score', 0):.4f}]")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
