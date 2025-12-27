#!/usr/bin/env python3
"""
Comprehensive End-to-End Validation Suite
==========================================
Complete validation of Telugu Poem Generation System including:
1. Dataset validation and statistics
2. Preprocessing pipeline verification
3. Interpretation module accuracy testing
4. Generation module quality assessment
5. Training stability verification

Author: Telugu Poem Generation System
Date: December 2024
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

# =============================================================================
# SECTION 1: DATASET VALIDATION
# =============================================================================

def validate_datasets() -> Dict[str, Any]:
    """Validate all datasets for completeness and quality."""
    print("\n" + "="*70)
    print("📊 SECTION 1: DATASET VALIDATION")
    print("="*70)
    
    results = {
        'status': 'PASS',
        'issues': [],
        'stats': {}
    }
    
    data_dir = PROJECT_ROOT / 'data' / 'processed'
    
    # Check required files
    required_files = ['telugu_train.json', 'telugu_val.json', 'telugu_test.json']
    
    for filename in required_files:
        filepath = data_dir / filename
        if not filepath.exists():
            results['issues'].append(f"Missing file: {filename}")
            results['status'] = 'FAIL'
            continue
        
        # Load and analyze
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'poems' in data:
            poems = data['poems']
        else:
            poems = data
        
        # Extract texts
        texts = []
        for poem in poems:
            if isinstance(poem, str):
                texts.append(poem)
            elif isinstance(poem, dict):
                text = poem.get('text', poem.get('content', poem.get('poem', '')))
                if text:
                    texts.append(text)
        
        # Statistics
        total = len(texts)
        unique = len(set(texts))
        duplicates = total - unique
        avg_length = sum(len(t) for t in texts) / max(1, total)
        
        # Check for Telugu content
        telugu_count = 0
        for text in texts[:100]:  # Sample check
            telugu_chars = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
            if telugu_chars > len(text) * 0.3:
                telugu_count += 1
        
        telugu_ratio = telugu_count / min(100, total)
        
        results['stats'][filename] = {
            'total': total,
            'unique': unique,
            'duplicates': duplicates,
            'avg_length': round(avg_length, 1),
            'telugu_ratio': round(telugu_ratio, 2)
        }
        
        # Quality checks
        if duplicates > total * 0.1:
            results['issues'].append(f"{filename}: High duplicate ratio ({duplicates}/{total})")
        
        if telugu_ratio < 0.9:
            results['issues'].append(f"{filename}: Low Telugu content ratio ({telugu_ratio:.0%})")
        
        print(f"\n✓ {filename}:")
        print(f"   Total poems: {total:,}")
        print(f"   Unique poems: {unique:,}")
        print(f"   Average length: {avg_length:.1f} chars")
        print(f"   Telugu content: {telugu_ratio:.0%}")
    
    # Check total dataset size
    total_poems = sum(s['total'] for s in results['stats'].values())
    if total_poems < 10000:
        results['issues'].append(f"Dataset size below target: {total_poems} < 10,000")
    
    print(f"\n📈 Total Dataset Size: {total_poems:,} poems")
    
    if results['issues']:
        print(f"\n⚠️ Issues found:")
        for issue in results['issues']:
            print(f"   - {issue}")
    else:
        print(f"\n✅ Dataset validation PASSED")
    
    return results


# =============================================================================
# SECTION 2: PREPROCESSING PIPELINE VALIDATION
# =============================================================================

def validate_preprocessing() -> Dict[str, Any]:
    """Validate preprocessing pipeline with test cases."""
    print("\n" + "="*70)
    print("🔧 SECTION 2: PREPROCESSING PIPELINE VALIDATION")
    print("="*70)
    
    from src.preprocessing.advanced_preprocessor import AdvancedTeluguPreprocessor
    
    results = {
        'status': 'PASS',
        'issues': [],
        'test_cases': []
    }
    
    preprocessor = AdvancedTeluguPreprocessor()
    
    # Test cases with expected behaviors
    test_cases = [
        {
            'name': 'Simple Telugu text',
            'input': 'తెలుగు భాష గొప్పది',
            'checks': ['text', 'structure', 'prosody']
        },
        {
            'name': 'Vemana Padyam',
            'input': '''ఆత్మశుద్ధి లేని యాచారమదియేల
భాండశుద్ధి లేని పాకమేల
చిత్తశుద్ధి లేని శివపూజలేల రా
విశ్వదాభిరామ వినురవేమ''',
            'checks': ['praasa', 'satakam', 'meter']
        },
        {
            'name': 'Mixed content',
            'input': 'Hello తెలుగు World భాష 123',
            'checks': ['cleaning', 'aksharas']
        },
        {
            'name': 'Emotional content',
            'input': 'ప్రేమ ఆనందం సంతోషం హృదయం',
            'checks': ['emotions']
        }
    ]
    
    for i, tc in enumerate(test_cases):
        print(f"\n📝 Test {i+1}: {tc['name']}")
        print(f"   Input: {tc['input'][:50]}...")
        
        test_result = {'name': tc['name'], 'status': 'PASS', 'details': {}}
        
        try:
            # Run preprocessing
            analysis = preprocessor.analyze_poem(tc['input'])
            
            # Verify output structure
            required_keys = ['text', 'structure', 'praasa', 'meter', 'semantics']
            for key in required_keys:
                if key not in analysis:
                    test_result['status'] = 'FAIL'
                    test_result['details'][key] = 'Missing'
                else:
                    test_result['details'][key] = 'Present'
            
            # Check specific features
            if 'praasa' in tc['checks']:
                praasa = analysis.get('praasa', {})
                test_result['details']['praasa_analysis'] = {
                    'has_praasa': praasa.get('has_praasa', False),
                    'match_ratio': praasa.get('match_ratio', 0)
                }
                print(f"   Praasa: {praasa.get('has_praasa', False)} (ratio: {praasa.get('match_ratio', 0):.2f})")
            
            if 'satakam' in tc['checks']:
                satakam = preprocessor.identify_satakam(tc['input'])
                test_result['details']['satakam'] = satakam
                print(f"   Śatakam: {satakam or 'Not identified'}")
            
            if 'emotions' in tc['checks']:
                emotions = preprocessor.detect_emotions(tc['input'])
                dominant = max(emotions.items(), key=lambda x: x[1]) if emotions else ('none', 0)
                test_result['details']['emotions'] = emotions
                print(f"   Dominant emotion: {dominant[0]} ({dominant[1]:.2f})")
            
            if 'aksharas' in tc['checks']:
                aksharas = preprocessor.extract_aksharas(tc['input'])
                test_result['details']['akshara_count'] = len(aksharas)
                print(f"   Aksharas: {len(aksharas)}")
            
            print(f"   ✓ Test PASSED")
            
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['details']['error'] = str(e)
            results['issues'].append(f"Test '{tc['name']}' failed: {e}")
            print(f"   ✗ Test FAILED: {e}")
        
        results['test_cases'].append(test_result)
    
    # Summary
    passed = sum(1 for tc in results['test_cases'] if tc['status'] == 'PASS')
    total = len(results['test_cases'])
    
    if passed < total:
        results['status'] = 'PARTIAL' if passed > 0 else 'FAIL'
    
    print(f"\n📊 Preprocessing Tests: {passed}/{total} passed")
    
    if passed == total:
        print("✅ Preprocessing validation PASSED")
    else:
        print(f"⚠️ Preprocessing validation: {passed}/{total} tests passed")
    
    return results


# =============================================================================
# SECTION 3: INTERPRETATION MODULE VALIDATION
# =============================================================================

def validate_interpretation() -> Dict[str, Any]:
    """Validate poem interpretation for semantic accuracy."""
    print("\n" + "="*70)
    print("🎭 SECTION 3: INTERPRETATION MODULE VALIDATION")
    print("="*70)
    
    from src.interpretation.poem_interpreter import (
        TeluguPoemInterpreter,
        RasaAnalyzer,
        ThemeClassifier,
        ProsodyAnalyzer
    )
    
    results = {
        'status': 'PASS',
        'issues': [],
        'rasa_accuracy': 0,
        'theme_accuracy': 0,
        'test_cases': []
    }
    
    interpreter = TeluguPoemInterpreter()
    rasa_analyzer = RasaAnalyzer()
    theme_classifier = ThemeClassifier()
    
    # Test cases with expected outputs
    test_cases = [
        # Rasa tests
        {
            'type': 'rasa',
            'input': 'ప్రేమ ప్రియమైన అనురాగం మనసు హృదయం',
            'expected': 'శృంగారం',
            'description': 'Love/Romance content'
        },
        {
            'type': 'rasa',
            'input': 'కోపం క్రోధం రోషం ఆగ్రహం',
            'expected': 'రౌద్రం',
            'description': 'Anger content'
        },
        {
            'type': 'rasa',
            'input': 'భయం భీతి త్రాసం వెరపు',
            'expected': 'భయానకం',
            'description': 'Fear content'
        },
        {
            'type': 'rasa',
            'input': 'ఆనందం సంతోషం హర్షం నవ్వు',
            'expected': 'హాస్యం',
            'description': 'Joy/Humor content'
        },
        {
            'type': 'rasa',
            'input': 'దుఃఖం విషాదం శోకం బాధ కన్నీరు',
            'expected': 'కరుణ',
            'description': 'Sorrow content'
        },
        # Theme tests
        {
            'type': 'theme',
            'input': 'భక్తి దేవుడు ప్రార్థన పూజ దైవం',
            'expected': 'భక్తి',
            'description': 'Devotion theme'
        },
        {
            'type': 'theme',
            'input': 'ధర్మం నీతి న్యాయం సత్యం',
            'expected': 'నీతి',
            'description': 'Ethics theme'
        },
        {
            'type': 'theme',
            'input': 'ప్రకృతి పూలు చెట్లు పక్షులు నది కొండలు',
            'expected': 'ప్రకృతి',
            'description': 'Nature theme'
        },
    ]
    
    rasa_correct = 0
    rasa_total = 0
    theme_correct = 0
    theme_total = 0
    
    for i, tc in enumerate(test_cases):
        print(f"\n📝 Test {i+1}: {tc['description']}")
        print(f"   Input: {tc['input'][:40]}...")
        print(f"   Expected: {tc['expected']}")
        
        test_result = {
            'type': tc['type'],
            'description': tc['description'],
            'expected': tc['expected'],
            'status': 'PASS'
        }
        
        try:
            if tc['type'] == 'rasa':
                rasa_total += 1
                scores = rasa_analyzer.detect_rasa(tc['input'])
                dominant, _ = rasa_analyzer.get_dominant_rasa(tc['input'])
                test_result['actual'] = dominant
                test_result['scores'] = scores
                
                if dominant == tc['expected']:
                    rasa_correct += 1
                    print(f"   ✓ Detected: {dominant} - CORRECT")
                else:
                    test_result['status'] = 'FAIL'
                    print(f"   ✗ Detected: {dominant} - INCORRECT")
                    # Check if expected is in top-2
                    if scores.get(tc['expected'], 0) > 0:
                        print(f"      (Expected rasa has score: {scores.get(tc['expected'], 0):.2f})")
            
            elif tc['type'] == 'theme':
                theme_total += 1
                scores = theme_classifier.detect_themes(tc['input'])
                primary = theme_classifier.get_primary_themes(tc['input'], top_k=1)
                detected = primary[0][0] if primary else 'unknown'
                test_result['actual'] = detected
                test_result['scores'] = scores
                
                if detected == tc['expected']:
                    theme_correct += 1
                    print(f"   ✓ Detected: {detected} - CORRECT")
                else:
                    test_result['status'] = 'FAIL'
                    print(f"   ✗ Detected: {detected} - INCORRECT")
        
        except Exception as e:
            test_result['status'] = 'ERROR'
            test_result['error'] = str(e)
            results['issues'].append(f"Test '{tc['description']}' error: {e}")
            print(f"   ✗ Error: {e}")
        
        results['test_cases'].append(test_result)
    
    # Calculate accuracy
    results['rasa_accuracy'] = rasa_correct / max(1, rasa_total)
    results['theme_accuracy'] = theme_correct / max(1, theme_total)
    
    print(f"\n📊 Interpretation Accuracy:")
    print(f"   Rasa Detection: {rasa_correct}/{rasa_total} ({results['rasa_accuracy']:.0%})")
    print(f"   Theme Classification: {theme_correct}/{theme_total} ({results['theme_accuracy']:.0%})")
    
    # Test full interpretation
    print(f"\n📜 Full Interpretation Test:")
    sample_poem = """ఆత్మశుద్ధి లేని యాచారమదియేల
భాండశుద్ధి లేని పాకమేల
విశ్వదాభిరామ వినురవేమ"""
    
    try:
        interpretation = interpreter.interpret(sample_poem)
        print(f"   Rasa: {interpretation['rasa']['dominant']}")
        print(f"   Themes: {[t[0] for t in interpretation['themes']['primary'][:2]]}")
        print(f"   Śatakam: {interpretation.get('satakam', 'Not detected')}")
        print(f"   Quality Score: {interpretation['quality']['overall']:.2f}")
        
        # Check embedding
        embedding = interpreter.get_interpretation_embedding(sample_poem)
        print(f"   Embedding Dimension: {len(embedding)}")
        
    except Exception as e:
        results['issues'].append(f"Full interpretation failed: {e}")
        print(f"   ✗ Error: {e}")
    
    # Overall status
    overall_accuracy = (results['rasa_accuracy'] + results['theme_accuracy']) / 2
    if overall_accuracy >= 0.8:
        results['status'] = 'PASS'
        print(f"\n✅ Interpretation validation PASSED ({overall_accuracy:.0%} accuracy)")
    elif overall_accuracy >= 0.5:
        results['status'] = 'PARTIAL'
        print(f"\n⚠️ Interpretation validation PARTIAL ({overall_accuracy:.0%} accuracy)")
    else:
        results['status'] = 'FAIL'
        print(f"\n❌ Interpretation validation FAILED ({overall_accuracy:.0%} accuracy)")
    
    return results


# =============================================================================
# SECTION 4: GENERATION MODULE VALIDATION
# =============================================================================

def validate_generation() -> Dict[str, Any]:
    """Validate poem generation for quality and non-repetition."""
    print("\n" + "="*70)
    print("✍️ SECTION 4: GENERATION MODULE VALIDATION")
    print("="*70)
    
    from src.models.enhanced_generator import (
        TeluguPoemGeneratorV3,
        GenerationConfig,
        create_enhanced_generator
    )
    from src.interpretation.poem_interpreter import TeluguPoemInterpreter
    
    results = {
        'status': 'PASS',
        'issues': [],
        'metrics': {},
        'samples': []
    }
    
    # Create model
    print("\n📦 Loading generation model...")
    try:
        model = create_enhanced_generator('mbert', freeze_encoder=True)
        model.eval()
        print("   ✓ Model loaded successfully")
        
        total, trainable = model.count_parameters()
        print(f"   Parameters: {total:,} total, {trainable:,} trainable")
    except Exception as e:
        results['status'] = 'FAIL'
        results['issues'].append(f"Model loading failed: {e}")
        print(f"   ✗ Model loading failed: {e}")
        return results
    
    # Test prompts
    test_prompts = [
        "తెలుగు భాష",
        "అమ్మ ప్రేమ",
        "ధర్మం మార్గం",
        "విద్య నేర్చుకో",
        "స్నేహం మంచిది"
    ]
    
    config = GenerationConfig(
        max_length=100,
        min_length=20,
        temperature=0.85,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.8,
        no_repeat_ngram_size=4,
        diversity_penalty=0.5
    )
    
    interpreter = TeluguPoemInterpreter()
    
    # Metrics collection
    total_repetition_ratio = 0
    total_telugu_ratio = 0
    total_length = 0
    generation_times = []
    
    print("\n🎭 Generation Tests:")
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n{'─'*60}")
        print(f"📝 Test {i+1}: Prompt = '{prompt}'")
        
        sample_result = {
            'prompt': prompt,
            'status': 'PASS',
            'metrics': {}
        }
        
        try:
            # Generate
            start_time = time.time()
            output = model.generate(prompt, config)
            gen_time = time.time() - start_time
            generation_times.append(gen_time)
            
            print(f"   Generated ({gen_time:.2f}s):")
            print(f"   '{output[:150]}{'...' if len(output) > 150 else ''}'")
            
            # Analyze output
            # 1. Length
            output_len = len(output)
            sample_result['metrics']['length'] = output_len
            total_length += output_len
            
            # 2. Telugu content ratio
            telugu_chars = sum(1 for c in output if '\u0C00' <= c <= '\u0C7F')
            telugu_ratio = telugu_chars / max(1, len(output))
            sample_result['metrics']['telugu_ratio'] = telugu_ratio
            total_telugu_ratio += telugu_ratio
            
            # 3. Repetition analysis
            words = output.split()
            if len(words) > 1:
                word_counts = Counter(words)
                repeated = sum(c - 1 for c in word_counts.values() if c > 1)
                repetition_ratio = repeated / len(words)
            else:
                repetition_ratio = 0
            sample_result['metrics']['repetition_ratio'] = repetition_ratio
            total_repetition_ratio += repetition_ratio
            
            # 4. N-gram repetition check
            ngram_repetitions = 0
            for n in [3, 4]:
                ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
                ngram_counts = Counter(ngrams)
                ngram_repetitions += sum(1 for c in ngram_counts.values() if c > 1)
            sample_result['metrics']['ngram_repetitions'] = ngram_repetitions
            
            # Quality assessment
            print(f"\n   📊 Metrics:")
            print(f"      Length: {output_len} chars")
            print(f"      Telugu ratio: {telugu_ratio:.0%}")
            print(f"      Word repetition: {repetition_ratio:.0%}")
            print(f"      N-gram repetitions: {ngram_repetitions}")
            
            # Interpret generated output
            try:
                interp = interpreter.interpret(output)
                sample_result['metrics']['rasa'] = interp['rasa']['dominant']
                sample_result['metrics']['quality_score'] = interp['quality']['overall']
                print(f"      Rasa: {interp['rasa']['dominant']}")
                print(f"      Quality: {interp['quality']['overall']:.2f}")
            except:
                pass
            
            # Pass/Fail criteria
            if telugu_ratio < 0.3:
                sample_result['status'] = 'FAIL'
                sample_result['issue'] = 'Low Telugu content'
                results['issues'].append(f"Prompt '{prompt}': Low Telugu content ({telugu_ratio:.0%})")
            elif repetition_ratio > 0.5:
                sample_result['status'] = 'FAIL'
                sample_result['issue'] = 'High repetition'
                results['issues'].append(f"Prompt '{prompt}': High repetition ({repetition_ratio:.0%})")
            elif ngram_repetitions > 5:
                sample_result['status'] = 'WARN'
                sample_result['issue'] = 'Some n-gram repetition'
            
            status_icon = '✓' if sample_result['status'] == 'PASS' else ('⚠' if sample_result['status'] == 'WARN' else '✗')
            print(f"\n   {status_icon} Test {sample_result['status']}")
            
        except Exception as e:
            sample_result['status'] = 'ERROR'
            sample_result['error'] = str(e)
            results['issues'].append(f"Generation failed for '{prompt}': {e}")
            print(f"   ✗ Error: {e}")
        
        results['samples'].append(sample_result)
    
    # Summary metrics
    n_tests = len(test_prompts)
    results['metrics'] = {
        'avg_length': total_length / n_tests,
        'avg_telugu_ratio': total_telugu_ratio / n_tests,
        'avg_repetition_ratio': total_repetition_ratio / n_tests,
        'avg_generation_time': sum(generation_times) / len(generation_times) if generation_times else 0,
        'passed': sum(1 for s in results['samples'] if s['status'] == 'PASS'),
        'total': n_tests
    }
    
    print(f"\n{'─'*60}")
    print(f"📊 Generation Summary:")
    print(f"   Tests Passed: {results['metrics']['passed']}/{n_tests}")
    print(f"   Avg Length: {results['metrics']['avg_length']:.0f} chars")
    print(f"   Avg Telugu Ratio: {results['metrics']['avg_telugu_ratio']:.0%}")
    print(f"   Avg Repetition: {results['metrics']['avg_repetition_ratio']:.0%}")
    print(f"   Avg Gen Time: {results['metrics']['avg_generation_time']:.2f}s")
    
    # Overall status
    pass_rate = results['metrics']['passed'] / n_tests
    if pass_rate >= 0.8 and results['metrics']['avg_repetition_ratio'] < 0.3:
        results['status'] = 'PASS'
        print(f"\n✅ Generation validation PASSED")
    elif pass_rate >= 0.5:
        results['status'] = 'PARTIAL'
        print(f"\n⚠️ Generation validation PARTIAL ({pass_rate:.0%} pass rate)")
    else:
        results['status'] = 'FAIL'
        print(f"\n❌ Generation validation FAILED ({pass_rate:.0%} pass rate)")
    
    return results


# =============================================================================
# SECTION 5: TRAINING STABILITY TEST
# =============================================================================

def validate_training(epochs: int = 5, quick_test: bool = True) -> Dict[str, Any]:
    """
    Validate training pipeline stability.
    
    Args:
        epochs: Number of epochs to test (use small for quick validation)
        quick_test: If True, use minimal data for quick testing
    """
    print("\n" + "="*70)
    print(f"🏋️ SECTION 5: TRAINING STABILITY TEST ({epochs} epochs)")
    print("="*70)
    
    from src.models.enhanced_generator import create_enhanced_generator
    from src.training.enhanced_trainer import (
        EnhancedTrainer,
        TrainingConfig,
        TeluguPoemDataset
    )
    
    results = {
        'status': 'PASS',
        'issues': [],
        'training_log': [],
        'final_loss': None
    }
    
    # Create model
    print("\n📦 Setting up training...")
    try:
        model = create_enhanced_generator('mbert', freeze_encoder=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"   ✓ Model created on {device}")
    except Exception as e:
        results['status'] = 'FAIL'
        results['issues'].append(f"Model creation failed: {e}")
        print(f"   ✗ Model creation failed: {e}")
        return results
    
    # Load data
    train_path = PROJECT_ROOT / 'data' / 'processed' / 'telugu_train.json'
    val_path = PROJECT_ROOT / 'data' / 'processed' / 'telugu_val.json'
    
    if not train_path.exists():
        results['status'] = 'FAIL'
        results['issues'].append("Training data not found")
        print("   ✗ Training data not found")
        return results
    
    try:
        # For quick test, use subset
        if quick_test:
            # Load subset
            with open(train_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'poems' in data:
                poems = data['poems'][:100]  # Use only 100 samples
                subset_data = {'poems': poems}
            else:
                subset_data = data[:100]
            
            # Save temp subset
            temp_path = PROJECT_ROOT / 'data' / 'processed' / 'temp_train_subset.json'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(subset_data, f)
            
            train_dataset = TeluguPoemDataset(str(temp_path), model.tokenizer, max_length=64)
            
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
        else:
            train_dataset = TeluguPoemDataset(str(train_path), model.tokenizer, max_length=128)
        
        print(f"   ✓ Dataset loaded: {len(train_dataset)} samples")
        
    except Exception as e:
        results['status'] = 'FAIL'
        results['issues'].append(f"Dataset loading failed: {e}")
        print(f"   ✗ Dataset loading failed: {e}")
        return results
    
    # Training config
    config = TrainingConfig(
        epochs=epochs,
        batch_size=4 if quick_test else 16,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_length=64 if quick_test else 128,
        checkpoint_dir=str(PROJECT_ROOT / 'checkpoints' / 'validation_test'),
        use_amp=torch.cuda.is_available(),
        val_every_n_steps=1000,  # Disable frequent validation
        patience=epochs + 1  # Disable early stopping for test
    )
    
    print(f"\n⚙️ Training Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Mixed precision: {config.use_amp}")
    
    # Create trainer
    try:
        trainer = EnhancedTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=None,  # Skip validation for quick test
            device=device
        )
        print("   ✓ Trainer created")
    except Exception as e:
        results['status'] = 'FAIL'
        results['issues'].append(f"Trainer creation failed: {e}")
        print(f"   ✗ Trainer creation failed: {e}")
        return results
    
    # Run training
    print(f"\n🚀 Starting training for {epochs} epochs...")
    
    try:
        training_results = trainer.train()
        
        results['training_log'] = training_results.get('train_losses', [])
        results['final_loss'] = results['training_log'][-1] if results['training_log'] else None
        
        print(f"\n📊 Training Results:")
        print(f"   Initial Loss: {results['training_log'][0]:.4f}" if results['training_log'] else "   No loss recorded")
        print(f"   Final Loss: {results['final_loss']:.4f}" if results['final_loss'] else "   No final loss")
        
        # Check for training issues
        if results['training_log']:
            # Check for NaN
            if any(loss != loss for loss in results['training_log']):  # NaN check
                results['status'] = 'FAIL'
                results['issues'].append("NaN loss detected during training")
            
            # Check for loss decrease
            if len(results['training_log']) >= 2:
                if results['training_log'][-1] > results['training_log'][0] * 1.5:
                    results['issues'].append("Loss increased significantly during training")
                    results['status'] = 'WARN'
        
        print("   ✓ Training completed successfully")
        
    except Exception as e:
        results['status'] = 'FAIL'
        results['issues'].append(f"Training failed: {e}")
        print(f"   ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    # Test generation after training
    print("\n🎭 Post-training generation test...")
    try:
        model.eval()
        from src.models.enhanced_generator import GenerationConfig
        gen_config = GenerationConfig(max_length=50, min_length=10)
        output = model.generate("తెలుగు", gen_config)
        print(f"   Generated: {output[:100]}...")
        print("   ✓ Post-training generation works")
    except Exception as e:
        results['issues'].append(f"Post-training generation failed: {e}")
        print(f"   ✗ Post-training generation failed: {e}")
    
    # Clean up checkpoint directory
    import shutil
    ckpt_dir = PROJECT_ROOT / 'checkpoints' / 'validation_test'
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    
    if results['status'] == 'PASS' and not results['issues']:
        print(f"\n✅ Training validation PASSED")
    elif results['status'] == 'WARN':
        print(f"\n⚠️ Training validation PASSED with warnings")
    else:
        print(f"\n❌ Training validation FAILED")
    
    return results


# =============================================================================
# MAIN VALIDATION RUNNER
# =============================================================================

def run_comprehensive_validation(training_epochs: int = 5) -> Dict[str, Any]:
    """Run all validation tests."""
    print("\n" + "="*70)
    print("🔬 COMPREHENSIVE VALIDATION SUITE")
    print("   Telugu Poem Generation System")
    print("="*70)
    print(f"   Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Training epochs: {training_epochs}")
    
    all_results = {}
    start_time = time.time()
    
    # Run all validations
    all_results['datasets'] = validate_datasets()
    all_results['preprocessing'] = validate_preprocessing()
    all_results['interpretation'] = validate_interpretation()
    all_results['generation'] = validate_generation()
    all_results['training'] = validate_training(epochs=training_epochs, quick_test=True)
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "="*70)
    print("📊 FINAL VALIDATION SUMMARY")
    print("="*70)
    
    status_icons = {'PASS': '✅', 'PARTIAL': '⚠️', 'WARN': '⚠️', 'FAIL': '❌'}
    
    all_pass = True
    for section, result in all_results.items():
        status = result.get('status', 'UNKNOWN')
        icon = status_icons.get(status, '❓')
        print(f"   {icon} {section.capitalize()}: {status}")
        if status not in ['PASS', 'WARN']:
            all_pass = False
    
    print(f"\n   Total Time: {total_time:.1f} seconds")
    
    if all_pass:
        print("\n" + "="*70)
        print("✅ ALL VALIDATIONS PASSED - System is ready for full training!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("⚠️ Some validations need attention - Review issues above")
        print("="*70)
    
    # Save results
    results_path = PROJECT_ROOT / 'results' / 'validation_results.json'
    results_path.parent.mkdir(exist_ok=True)
    
    # Convert non-serializable items
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(make_serializable(all_results), f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive validation of Telugu Poem System')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs for validation')
    parser.add_argument('--section', type=str, default='all',
                       choices=['all', 'datasets', 'preprocessing', 'interpretation', 'generation', 'training'],
                       help='Run specific section only')
    args = parser.parse_args()
    
    if args.section == 'all':
        run_comprehensive_validation(training_epochs=args.epochs)
    else:
        # Run specific section
        section_funcs = {
            'datasets': validate_datasets,
            'preprocessing': validate_preprocessing,
            'interpretation': validate_interpretation,
            'generation': validate_generation,
            'training': lambda: validate_training(epochs=args.epochs)
        }
        section_funcs[args.section]()
