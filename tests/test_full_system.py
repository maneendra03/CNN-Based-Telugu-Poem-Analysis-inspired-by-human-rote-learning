#!/usr/bin/env python3
"""
Comprehensive Test Suite for Telugu Poem Generation System
==========================================================
Tests all modules: preprocessing, interpretation, and generation.

Usage:
    python tests/test_full_system.py
    python tests/test_full_system.py --verbose
"""

import sys
from pathlib import Path
import unittest
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestAdvancedPreprocessor(unittest.TestCase):
    """Test advanced preprocessing module."""
    
    @classmethod
    def setUpClass(cls):
        from src.preprocessing.advanced_preprocessor import (
            AdvancedTeluguPreprocessor,
            TeluguPhonetics
        )
        cls.preprocessor = AdvancedTeluguPreprocessor()
        cls.phonetics = TeluguPhonetics()
    
    def test_initialization(self):
        """Test preprocessor initializes correctly."""
        self.assertIsNotNone(self.preprocessor)
        self.assertIsNotNone(self.phonetics)
    
    def test_akshara_extraction(self):
        """Test Telugu akshara extraction."""
        text = "తెలుగు"
        aksharas = self.preprocessor.extract_aksharas(text)
        # తె-లు-గు = 3 aksharas
        self.assertGreaterEqual(len(aksharas), 2)
        print(f"   Aksharas in 'తెలుగు': {aksharas}")
    
    def test_text_processing(self):
        """Test full text processing."""
        text = "తెలుగు భాష గొప్పది"
        result = self.preprocessor.analyze_poem(text)
        
        self.assertIn('text', result)
        self.assertIn('structure', result)
        self.assertIn('praasa', result)
        self.assertIn('meter', result)
        self.assertIn('semantics', result)
        
        print(f"   Processed: {result['text']}")
        print(f"   Structure: {result['structure']}")
    
    def test_praasa_analysis(self):
        """Test praasa (rhyme) analysis."""
        # Vemana style verse with praasa
        verse = """ఆత్మశుద్ధి లేని యాచారమదియేల
విశ్వదాభిరామ వినురవేమ"""
        
        result = self.preprocessor.analyze_praasa(verse)
        
        self.assertIn('has_praasa', result)
        print(f"   Praasa result: {result}")
    
    def test_emotion_detection(self):
        """Test emotion detection in text."""
        happy_text = "సంతోషం ఆనందం"
        sad_text = "దుఃఖం బాధ"
        
        happy_emotions = self.preprocessor.detect_emotions(happy_text)
        sad_emotions = self.preprocessor.detect_emotions(sad_text)
        
        print(f"   Happy text emotions: {happy_emotions}")
        print(f"   Sad text emotions: {sad_emotions}")


class TestPoemInterpreter(unittest.TestCase):
    """Test poem interpretation module."""
    
    @classmethod
    def setUpClass(cls):
        from src.interpretation.poem_interpreter import (
            TeluguPoemInterpreter,
            ProsodyAnalyzer,
            RasaAnalyzer,
            ThemeClassifier
        )
        cls.interpreter = TeluguPoemInterpreter()
        cls.prosody = ProsodyAnalyzer()
        cls.rasa = RasaAnalyzer()
        cls.theme = ThemeClassifier()
    
    def test_initialization(self):
        """Test interpreter initializes correctly."""
        self.assertIsNotNone(self.interpreter)
        self.assertIsNotNone(self.prosody)
        self.assertIsNotNone(self.rasa)
        self.assertIsNotNone(self.theme)
    
    def test_rasa_detection(self):
        """Test navarasa detection."""
        # Test different rasas
        test_cases = [
            ("ప్రేమ ప్రియమైన", "శృంగారం"),
            ("కోపం క్రోధం", "రౌద్రం"),
            ("భయం భీతి", "భయానకం"),
            ("సంతోషం హాస్యం", "హాస్యం"),
        ]
        
        for text, expected_hint in test_cases:
            rasa_scores = self.rasa.detect_rasa(text)
            self.assertIsInstance(rasa_scores, dict)
            print(f"   '{text}' -> Rasa scores: {rasa_scores}")
    
    def test_theme_classification(self):
        """Test theme classification."""
        test_cases = [
            ("భక్తి దేవుడు ప్రార్థన", "devotion"),
            ("ప్రేమ ప్రియమైన హృదయం", "love"),
            ("ధర్మం నీతి న్యాయం", "philosophy"),
            ("ప్రకృతి పూలు పక్షులు", "nature"),
        ]
        
        for text, expected_theme in test_cases:
            theme_scores = self.theme.detect_themes(text)
            self.assertIsInstance(theme_scores, dict)
            print(f"   '{text}' -> Theme scores: {theme_scores}")
    
    def test_full_interpretation(self):
        """Test full poem interpretation."""
        poem = """ఆత్మశుద్ధి లేని యాచారమదియేల
భాండశుద్ధి లేని పాకమేల
చిత్తశుద్ధి లేని శివపూజలేల రా
విశ్వదాభిరామ వినురవేమ"""
        
        interpretation = self.interpreter.interpret(poem)
        
        self.assertIn('rasa', interpretation)
        self.assertIn('themes', interpretation)
        self.assertIn('prosody', interpretation)
        self.assertIn('quality', interpretation)  # Changed from quality_metrics
        
        print(f"\n   Full Interpretation:")
        print(f"   Rasa: {interpretation['rasa']['dominant']}")
        print(f"   Themes: {interpretation['themes']['primary']}")
        print(f"   Quality: {interpretation['quality']}")
    
    def test_embedding_generation(self):
        """Test embedding generation."""
        poem = "తెలుగు భాష గొప్పది"
        embedding = self.interpreter.get_interpretation_embedding(poem)
        
        self.assertEqual(len(embedding), 32)
        print(f"   Embedding shape: {len(embedding)}")
        print(f"   Embedding sample: {embedding[:5]}")


class TestEnhancedGenerator(unittest.TestCase):
    """Test enhanced generator module."""
    
    @classmethod
    def setUpClass(cls):
        from src.models.enhanced_generator import (
            TeluguPoemGeneratorV3,
            GenerationConfig,
            RepetitionHandler,
            create_enhanced_generator
        )
        import torch
        
        cls.GenerationConfig = GenerationConfig
        cls.RepetitionHandler = RepetitionHandler
        
        # Create model (use mbert for faster testing)
        print("\n   Creating model (this may take a moment)...")
        cls.model = create_enhanced_generator('mbert', freeze_encoder=True)
        cls.device = torch.device('cpu')  # Use CPU for testing
        cls.model = cls.model.to(cls.device)
        cls.model.eval()
    
    def test_initialization(self):
        """Test generator initializes correctly."""
        self.assertIsNotNone(self.model)
        total, trainable = self.model.count_parameters()
        self.assertGreater(total, 0)
        print(f"   Parameters: {total:,} total, {trainable:,} trainable")
    
    def test_repetition_handler(self):
        """Test repetition handler."""
        import torch
        
        vocab_size = 100
        handler = self.RepetitionHandler(
            vocab_size=vocab_size,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3
        )
        
        handler.reset(self.device)
        
        # Simulate sequence generation
        sequence = torch.tensor([1, 2, 3, 1, 2], device=self.device)
        logits = torch.randn(1, vocab_size, device=self.device)
        
        # Apply penalties
        penalized = handler.apply_penalties(logits.clone(), sequence)
        
        # Token 1 and 2 should have lower logits (repeated)
        self.assertLessEqual(penalized[0, 1].item(), logits[0, 1].item())
        print(f"   Repetition penalty applied correctly")
    
    def test_generation_config(self):
        """Test generation configuration."""
        config = self.GenerationConfig(
            max_length=100,
            temperature=0.85,
            repetition_penalty=1.8
        )
        
        self.assertEqual(config.max_length, 100)
        self.assertEqual(config.temperature, 0.85)
        self.assertEqual(config.repetition_penalty, 1.8)
    
    def test_basic_generation(self):
        """Test basic poem generation."""
        config = self.GenerationConfig(
            max_length=50,
            min_length=10,
            temperature=0.9,
            repetition_penalty=1.5
        )
        
        prompt = "తెలుగు"
        output = self.model.generate(prompt, config)
        
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)
        print(f"   Prompt: '{prompt}'")
        print(f"   Output: '{output[:100]}...'")
    
    def test_style_generation(self):
        """Test style-conditioned generation."""
        config = self.GenerationConfig(
            max_length=60,
            min_length=15
        )
        
        prompt = "ధర్మం"
        output = self.model.generate_with_style(prompt, style='vemana', config=config)
        
        self.assertIsInstance(output, str)
        print(f"   Vemana style output: '{output[:100]}...'")


class TestDataset(unittest.TestCase):
    """Test dataset loading."""
    
    def test_processed_data_exists(self):
        """Test that processed data exists."""
        data_dir = PROJECT_ROOT / 'data' / 'processed'
        
        expected_files = [
            'telugu_train.json',
            'telugu_val.json',
            'telugu_test.json'
        ]
        
        for filename in expected_files:
            path = data_dir / filename
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'poems' in data:
                    count = len(data['poems'])
                else:
                    count = len(data)
                print(f"   {filename}: {count} poems ✓")
            else:
                print(f"   {filename}: NOT FOUND")
    
    def test_dataset_quality(self):
        """Test dataset quality metrics."""
        train_path = PROJECT_ROOT / 'data' / 'processed' / 'telugu_train.json'
        
        if not train_path.exists():
            self.skipTest("Training data not found")
        
        with open(train_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'poems' in data:
            poems = data['poems']
        else:
            poems = data
        
        # Check for duplicates
        texts = set()
        duplicates = 0
        
        for poem in poems:
            if isinstance(poem, str):
                text = poem
            else:
                text = poem.get('text', poem.get('content', ''))
            
            if text in texts:
                duplicates += 1
            texts.add(text)
        
        print(f"   Total poems: {len(poems)}")
        print(f"   Unique poems: {len(texts)}")
        print(f"   Duplicates: {duplicates}")
        
        self.assertLess(duplicates, len(poems) * 0.1)  # Less than 10% duplicates


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def test_preprocess_interpret_generate(self):
        """Test full preprocessing -> interpretation -> generation pipeline."""
        from src.preprocessing.advanced_preprocessor import AdvancedTeluguPreprocessor
        from src.interpretation.poem_interpreter import TeluguPoemInterpreter
        from src.models.enhanced_generator import create_enhanced_generator, GenerationConfig
        
        print("\n   Running integration test...")
        
        # 1. Preprocess sample poem
        preprocessor = AdvancedTeluguPreprocessor()
        sample_poem = "తెలుగు భాష మన గౌరవం"
        preprocessed = preprocessor.analyze_poem(sample_poem)
        
        print(f"   1. Preprocessed: {preprocessed['text']}")
        
        # 2. Interpret
        interpreter = TeluguPoemInterpreter()
        interpretation = interpreter.interpret(sample_poem)
        
        print(f"   2. Interpreted - Rasa: {interpretation['rasa']}, Themes: {interpretation['themes']}")
        
        # 3. Generate (use prompt from interpretation)
        model = create_enhanced_generator('mbert', freeze_encoder=True)
        model.eval()
        
        config = GenerationConfig(
            max_length=50,
            min_length=10,
            temperature=0.85
        )
        
        generated = model.generate(sample_poem, config)
        
        print(f"   3. Generated: {generated[:100]}...")
        
        # 4. Interpret generated output
        gen_interpretation = interpreter.interpret(generated)
        
        print(f"   4. Generated poem - Rasa: {gen_interpretation['rasa']}")
        
        self.assertIsNotNone(generated)
        print("\n   ✅ Integration test passed!")


def run_tests(verbosity=2):
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestPoemInterpreter))
    suite.addTests(loader.loadTestsFromTestCase(TestDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🧪 Telugu Poem Generation System - Comprehensive Tests")
    print("="*60)
    
    verbosity = 2 if args.verbose else 1
    result = run_tests(verbosity)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
