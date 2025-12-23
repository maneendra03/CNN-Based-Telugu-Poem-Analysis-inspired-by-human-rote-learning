#!/usr/bin/env python
"""Test CNN-based poem interpretation"""

from src.models.poem_learner import PoemLearner
from src.preprocessing.tokenizer import PoemTokenizer
import torch
import json

# Load Telugu poems
with open('data/processed/telugu_poems.json', 'r') as f:
    poems = json.load(f)

print(f"📚 Loaded {len(poems)} poems")

# Create tokenizer
tokenizer = PoemTokenizer(min_freq=1)
tokenizer.fit([p['text'] for p in poems[:100]])

print(f"✅ Tokenizer created: vocab_size={tokenizer.word_vocab_size}")

# Create interpretation model
model = PoemLearner(
    vocab_size=tokenizer.word_vocab_size, 
    embedding_dim=128, 
    hidden_dim=256
)
model.eval()

print(f"✅ Model created: {sum(p.numel() for p in model.parameters())} parameters")

# Test on sample poem
sample = poems[0]['text']
print(f"\n📝 Testing interpretation on: {sample[:100]}...")

input_ids = torch.tensor([tokenizer.encode(sample)[:50]])
target_ids = torch.tensor([tokenizer.encode(sample)[1:31]])

output = model(input_ids, target_ids)
print(f"\n✅ Interpretation successful!")
print(f"   Output shape: {output['logits'].shape}")
print(f"   Features extracted: {output['poem_representation'].shape}")