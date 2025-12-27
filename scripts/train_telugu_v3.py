#!/usr/bin/env python3
"""
Master Training Script for Telugu Poem Generator V3
====================================================
Complete training pipeline with:
1. Data loading and preprocessing
2. Enhanced generator model
3. Professional training loop
4. Evaluation and generation testing

Usage:
    python scripts/train_telugu_v3.py --epochs 50 --batch_size 16
    python scripts/train_telugu_v3.py --test-only  # Just test generation
"""

import argparse
import sys
from pathlib import Path
import json
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.enhanced_generator import (
    TeluguPoemGeneratorV3,
    GenerationConfig,
    create_enhanced_generator
)
from src.training.enhanced_trainer import (
    EnhancedTrainer,
    TrainingConfig,
    TeluguPoemDataset,
    train_enhanced_model
)
from src.preprocessing.advanced_preprocessor import AdvancedTeluguPreprocessor
from src.interpretation.poem_interpreter import TeluguPoemInterpreter


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Telugu Poem Generator V3'
    )
    
    # Model
    parser.add_argument('--model-type', type=str, default='mbert',
                       choices=['indic-bert', 'mbert', 'xlm-roberta'],
                       help='Pretrained encoder type')
    parser.add_argument('--freeze-encoder', action='store_true', default=True,
                       help='Freeze encoder weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--grad-accum', type=int, default=2,
                       help='Gradient accumulation steps')
    
    # Data
    parser.add_argument('--train-data', type=str,
                       default='data/processed/telugu_train.json',
                       help='Training data path')
    parser.add_argument('--val-data', type=str,
                       default='data/processed/telugu_val.json',
                       help='Validation data path')
    parser.add_argument('--max-length', type=int, default=128,
                       help='Maximum sequence length')
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str,
                       default='checkpoints/telugu_v3',
                       help='Checkpoint directory')
    parser.add_argument('--output-dir', type=str,
                       default='results',
                       help='Output directory for results')
    
    # Testing
    parser.add_argument('--test-only', action='store_true',
                       help='Only test generation (no training)')
    parser.add_argument('--load-checkpoint', type=str,
                       help='Load from checkpoint')
    
    return parser.parse_args()


def test_generation(model, num_samples: int = 5):
    """Test poem generation with various prompts."""
    print("\n" + "="*60)
    print("🎭 Testing Poem Generation")
    print("="*60)
    
    prompts = [
        "తెలుగు భాష గొప్పది",
        "అమ్మ ప్రేమ అనంతం",
        "విద్య నేర్చుకోవాలి",
        "ధర్మం మార్గం చూపును",
        "స్నేహం మంచిది"
    ]
    
    config = GenerationConfig(
        max_length=100,
        min_length=30,
        temperature=0.85,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.8,
        no_repeat_ngram_size=4,
        diversity_penalty=0.5
    )
    
    # Initialize interpreter for analysis
    interpreter = TeluguPoemInterpreter()
    
    for i, prompt in enumerate(prompts[:num_samples]):
        print(f"\n{'─'*50}")
        print(f"📝 Prompt {i+1}: {prompt}")
        
        # Generate
        output = model.generate(prompt, config)
        print(f"📜 Generated:\n{output}")
        
        # Analyze
        try:
            interpretation = interpreter.interpret(output)
            print(f"\n📊 Analysis:")
            print(f"   Rasa: {interpretation.rasa}")
            print(f"   Theme: {interpretation.theme}")
            print(f"   Quality: {interpretation.quality_metrics.get('overall', 0):.2f}")
        except Exception as e:
            print(f"   (Analysis error: {e})")
    
    print("\n" + "="*60)


def evaluate_model(model, test_data_path: str):
    """Evaluate model on test set."""
    print("\n" + "="*60)
    print("📊 Evaluating Model")
    print("="*60)
    
    # Load test data
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if isinstance(test_data, dict) and 'poems' in test_data:
        poems = test_data['poems']
    else:
        poems = test_data
    
    # Sample for evaluation
    sample_size = min(50, len(poems))
    sample_poems = poems[:sample_size]
    
    # Initialize metrics
    interpreter = TeluguPoemInterpreter()
    preprocessor = AdvancedTeluguPreprocessor()
    
    metrics = {
        'total': 0,
        'has_praasa': 0,
        'avg_quality': 0.0,
        'rasa_distribution': {},
        'theme_distribution': {}
    }
    
    config = GenerationConfig(
        max_length=80,
        min_length=20,
        temperature=0.85,
        repetition_penalty=1.8
    )
    
    for poem_data in sample_poems:
        if isinstance(poem_data, str):
            text = poem_data[:50]  # Use first 50 chars as prompt
        else:
            text = poem_data.get('text', poem_data.get('content', ''))[:50]
        
        if not text:
            continue
        
        # Generate
        generated = model.generate(text, config)
        
        # Analyze
        try:
            interpretation = interpreter.interpret(generated)
            
            metrics['total'] += 1
            metrics['avg_quality'] += interpretation.quality_metrics.get('overall', 0)
            
            rasa = interpretation.rasa
            metrics['rasa_distribution'][rasa] = metrics['rasa_distribution'].get(rasa, 0) + 1
            
            theme = interpretation.theme
            metrics['theme_distribution'][theme] = metrics['theme_distribution'].get(theme, 0) + 1
            
            # Check praasa
            analysis = preprocessor.process(generated)
            if analysis['prosodic_analysis']['praasa_pattern']:
                metrics['has_praasa'] += 1
                
        except Exception:
            pass
    
    # Calculate averages
    if metrics['total'] > 0:
        metrics['avg_quality'] /= metrics['total']
        metrics['praasa_rate'] = metrics['has_praasa'] / metrics['total']
    
    print(f"\n📊 Evaluation Results ({metrics['total']} samples):")
    print(f"   Average Quality Score: {metrics['avg_quality']:.3f}")
    print(f"   Praasa Consistency Rate: {metrics.get('praasa_rate', 0):.2%}")
    print(f"\n   Rasa Distribution:")
    for rasa, count in sorted(metrics['rasa_distribution'].items(), key=lambda x: -x[1])[:5]:
        print(f"      {rasa}: {count}")
    print(f"\n   Theme Distribution:")
    for theme, count in sorted(metrics['theme_distribution'].items(), key=lambda x: -x[1])[:5]:
        print(f"      {theme}: {count}")
    
    return metrics


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("🎭 Telugu Poem Generator V3 - Training Pipeline")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    print("\n📦 Creating model...")
    model = create_enhanced_generator(
        model_type=args.model_type,
        freeze_encoder=args.freeze_encoder
    )
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"📥 Loading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    
    # Print model info
    total, trainable = model.count_parameters()
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    
    if args.test_only:
        # Just test generation
        test_generation(model, num_samples=5)
    else:
        # Full training
        print("\n📚 Loading data...")
        
        train_path = PROJECT_ROOT / args.train_data
        val_path = PROJECT_ROOT / args.val_data
        
        if not train_path.exists():
            print(f"❌ Training data not found: {train_path}")
            print("   Run: python scripts/download_datasets.py")
            return
        
        # Training config
        config = TrainingConfig(
            model_name=args.model_type,
            freeze_encoder=args.freeze_encoder,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            max_length=args.max_length,
            checkpoint_dir=args.checkpoint_dir,
            use_amp=torch.cuda.is_available()
        )
        
        print("\n" + "-"*60)
        print("Training Configuration:")
        print("-"*60)
        for key, value in config.__dict__.items():
            print(f"  {key}: {value}")
        print("-"*60)
        
        # Train
        results = train_enhanced_model(
            model=model,
            train_data_path=str(train_path),
            val_data_path=str(val_path) if val_path.exists() else None,
            config=config
        )
        
        # Save final results
        output_path = PROJECT_ROOT / args.output_dir / 'training_results_v3.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to {output_path}")
        
        # Test generation after training
        test_generation(model, num_samples=5)
        
        # Evaluate
        test_path = PROJECT_ROOT / 'data/processed/telugu_test.json'
        if test_path.exists():
            metrics = evaluate_model(model, str(test_path))
            
            # Save metrics
            metrics_path = PROJECT_ROOT / args.output_dir / 'evaluation_metrics_v3.json'
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
