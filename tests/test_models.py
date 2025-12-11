"""
Unit Tests for Poem Learner
Run with: pytest tests/ -v
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPreprocessing:
    """Tests for preprocessing modules."""
    
    def test_text_cleaner(self):
        from src.preprocessing.text_cleaner import TextCleaner
        
        cleaner = TextCleaner()
        text = "  Hello   World!  \n  This is   a test.  "
        cleaned = cleaner.clean(text)
        
        assert "Hello" in cleaned
        assert "  " not in cleaned  # No double spaces
    
    def test_text_cleaner_syllables(self):
        from src.preprocessing.text_cleaner import TextCleaner
        
        assert TextCleaner.count_syllables("hello") == 2
        assert TextCleaner.count_syllables("beautiful") == 3
        assert TextCleaner.count_syllables("poetry") >= 2  # Heuristic may vary
    
    def test_tokenizer_fit(self):
        from src.preprocessing.tokenizer import PoemTokenizer
        
        texts = ["hello world", "world of poetry", "poetry is beautiful"]
        tokenizer = PoemTokenizer(min_freq=1)
        tokenizer.fit(texts)
        
        assert tokenizer.word_vocab_size > 5  # Special tokens + words
        assert "hello" in tokenizer.word2idx
        assert "world" in tokenizer.word2idx
    
    def test_tokenizer_encode_decode(self):
        from src.preprocessing.tokenizer import PoemTokenizer
        
        texts = ["shall i compare thee to a summer's day"]
        tokenizer = PoemTokenizer(min_freq=1)
        tokenizer.fit(texts)
        
        encoded = tokenizer.encode_words(texts[0], max_length=20)
        assert len(encoded) == 20
        
        decoded = tokenizer.decode_words(encoded)
        assert "shall" in decoded.lower()


class TestModels:
    """Tests for model modules."""
    
    def test_cnn_feature_extractor(self):
        from src.models.cnn_module import CNNFeatureExtractor
        
        batch_size, seq_len, input_dim = 4, 50, 256
        model = CNNFeatureExtractor(input_dim=input_dim, num_filters=128)
        
        x = torch.randn(batch_size, seq_len, input_dim)
        seq_out, pooled = model(x)
        
        assert seq_out.shape == (batch_size, seq_len, model.get_output_dim())
        assert pooled.shape == (batch_size, model.get_output_dim())
    
    def test_hierarchical_rnn(self):
        from src.models.hierarchical_rnn import SimpleHierarchicalRNN
        
        batch_size, seq_len, input_dim = 4, 50, 256
        model = SimpleHierarchicalRNN(input_dim=input_dim, hidden_size=256)
        
        x = torch.randn(batch_size, seq_len, input_dim)
        line_out, poem_repr = model(x)
        
        assert line_out.shape[0] == batch_size
        assert poem_repr.shape == (batch_size, model.get_output_dim())
    
    def test_memory_attention(self):
        from src.models.memory_attention import MemoryAttentionModule
        
        batch_size, seq_len, input_dim = 4, 50, 256
        model = MemoryAttentionModule(input_dim=input_dim, attention_dim=256)
        
        x = torch.randn(batch_size, seq_len, input_dim)
        output, context = model(x)
        
        assert output.shape == (batch_size, seq_len, model.get_output_dim())
        assert context.shape == (batch_size, model.get_output_dim())
    
    def test_knowledge_base(self):
        from src.models.knowledge_base import KnowledgeBase
        
        batch_size, seq_len, embed_dim = 4, 50, 256
        kb = KnowledgeBase(embed_dim=embed_dim)
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        output = kb(x)
        
        assert 'knowledge_features' in output
        assert output['knowledge_features'].shape == x.shape
    
    def test_knowledge_base_rhyme_scoring(self):
        from src.models.knowledge_base import KnowledgeBase
        
        kb = KnowledgeBase(embed_dim=256)
        
        # These should rhyme
        score1 = kb.score_rhyme("day", "way")
        assert score1 > 0.5
        
        # These should not rhyme well
        score2 = kb.score_rhyme("day", "night")
        assert score2 < score1
    
    def test_feedback_loop(self):
        from src.models.feedback_loop import FeedbackLoop
        
        batch_size, seq_len, input_dim = 4, 50, 256
        model = FeedbackLoop(input_dim=input_dim, num_iterations=3)
        
        x = torch.randn(batch_size, seq_len, input_dim)
        result = model(x)
        
        assert 'refined' in result
        assert 'quality' in result
        assert result['refined'].shape == x.shape
    
    def test_decoder(self):
        from src.models.decoder import PoemDecoder
        
        batch_size, src_len, tgt_len, vocab_size = 4, 50, 30, 1000
        hidden_size = 256
        
        decoder = PoemDecoder(vocab_size=vocab_size, hidden_size=hidden_size)
        
        encoder_outputs = torch.randn(batch_size, src_len, hidden_size)
        context = torch.randn(batch_size, hidden_size)
        target_ids = torch.randint(0, vocab_size, (batch_size, tgt_len))
        
        logits, attn = decoder(encoder_outputs, context, target_ids)
        
        assert logits.shape == (batch_size, tgt_len, vocab_size)
    
    def test_poem_learner_forward(self):
        from src.models.poem_learner import PoemLearner
        
        batch_size, seq_len, vocab_size = 4, 50, 1000
        
        model = PoemLearner(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256
        )
        
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(1, vocab_size, (batch_size, 30))
        
        output = model(input_ids, target_ids)
        
        assert 'encoded' in output
        assert 'logits' in output
        assert output['logits'].shape[2] == vocab_size
    
    def test_poem_learner_encode(self):
        from src.models.poem_learner import PoemLearner
        
        batch_size, seq_len, vocab_size = 4, 50, 1000
        
        model = PoemLearner(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256)
        
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
        encoded = model.encode(input_ids)
        
        assert 'encoded' in encoded
        assert 'context' in encoded
        assert 'poem_representation' in encoded


class TestTraining:
    """Tests for training modules."""
    
    def test_poem_loss(self):
        from src.training.losses import PoemLoss
        
        batch_size, seq_len, vocab_size = 4, 30, 1000
        
        loss_fn = PoemLoss(vocab_size=vocab_size)
        
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        losses = loss_fn(logits, targets)
        
        assert 'total_loss' in losses
        assert 'lm_loss' in losses
        assert losses['total_loss'].item() > 0  # Loss should be positive


class TestEvaluation:
    """Tests for evaluation modules."""
    
    def test_poem_metrics(self):
        from src.evaluation.metrics import PoemMetrics
        
        metrics = PoemMetrics()
        
        predictions = ["Roses are red\nViolets are blue"]
        references = ["Roses are red\nViolets are blue"]
        
        metrics.update(predictions, references)
        results = metrics.compute_all()
        
        assert 'bleu' in results
        assert 'rhyme_accuracy' in results
        assert results['bleu'] > 0  # Should be high for identical texts


class TestDataLoading:
    """Tests for data loading."""
    
    def test_poem_data_loader_sample(self):
        from src.data.data_loader import PoemDataLoader
        
        poems = PoemDataLoader.create_sample_dataset(20)
        
        assert len(poems) == 20
        assert all('text' in p for p in poems)
    
    def test_poem_dataset(self):
        from src.data.data_loader import PoemDataset, PoemDataLoader
        from src.preprocessing.tokenizer import PoemTokenizer
        
        # Create sample data
        poems = PoemDataLoader.create_sample_dataset(10)
        
        # Create tokenizer
        tokenizer = PoemTokenizer(min_freq=1)
        texts = [p['text'] for p in poems]
        tokenizer.fit(texts)
        
        # Create dataset
        dataset = PoemDataset(poems, tokenizer, max_seq_length=100)
        
        assert len(dataset) == 10
        
        sample = dataset[0]
        assert 'input_ids' in sample
        assert 'target_ids' in sample
        assert sample['input_ids'].shape[0] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
