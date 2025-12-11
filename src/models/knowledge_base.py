"""
Knowledge Base Module
Stores and provides access to poetic rules, grammar, and stylistic references.

This module enables:
- Rule-based constraints on poem generation
- Stylistic pattern matching
- Grammar and structure validation
- Rhyme scheme enforcement
"""

import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from pathlib import Path
import re


class PoeticRule:
    """Represents a single poetic rule or constraint."""
    
    def __init__(
        self,
        name: str,
        rule_type: str,
        pattern: str,
        weight: float = 1.0,
        description: str = ""
    ):
        """
        Initialize a poetic rule.
        
        Args:
            name: Rule name
            rule_type: Type of rule (rhyme, meter, structure, grammar)
            pattern: Pattern or constraint specification
            weight: Importance weight for this rule
            description: Human-readable description
        """
        self.name = name
        self.rule_type = rule_type
        self.pattern = pattern
        self.weight = weight
        self.description = description
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'rule_type': self.rule_type,
            'pattern': self.pattern,
            'weight': self.weight,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PoeticRule':
        return cls(**data)


class KnowledgeBase(nn.Module):
    """
    Neural Knowledge Base for poetic rules and patterns.
    
    Provides:
    1. Learnable embeddings for poetic concepts
    2. Rule-based scoring for generated text
    3. Style transfer capabilities
    4. Grammar and structure validation
    
    This is the knowledge-grounded component that differentiates
    this approach from pure statistical learning.
    """
    
    # Common rhyme scheme patterns
    RHYME_SCHEMES = {
        'couplet': 'AA',
        'alternate': 'ABAB',
        'enclosed': 'ABBA',
        'sonnet_english': 'ABAB CDCD EFEF GG',
        'sonnet_italian': 'ABBA ABBA CDE CDE',
        'limerick': 'AABBA',
        'villanelle': 'ABA ABA ABA ABA ABA ABAA',
        'free': None
    }
    
    # Common meter patterns (stressed/unstressed)
    METER_PATTERNS = {
        'iambic': 'uS',  # unstressed-Stressed
        'trochaic': 'Su',  # Stressed-unstressed
        'anapestic': 'uuS',  # 2 unstressed, 1 stressed
        'dactylic': 'Suu',  # 1 stressed, 2 unstressed
        'spondaic': 'SS',  # both stressed
    }
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_rules: int = 100,
        num_styles: int = 20,
        knowledge_path: Optional[str] = None,
        dropout: float = 0.1
    ):
        """
        Initialize Knowledge Base.
        
        Args:
            embed_dim: Dimension for knowledge embeddings
            num_rules: Maximum number of rules to embed
            num_styles: Number of style embeddings
            knowledge_path: Path to knowledge base JSON files
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_rules = num_rules
        self.num_styles = num_styles
        
        # Rule embeddings (learnable representations of rules)
        self.rule_embeddings = nn.Embedding(num_rules, embed_dim)
        
        # Style embeddings (different poetic styles)
        self.style_embeddings = nn.Embedding(num_styles, embed_dim)
        
        # Knowledge projection
        self.knowledge_proj = nn.Linear(embed_dim, embed_dim)
        
        # Rule encoder (converts rule features to embeddings)
        self.rule_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Style classifier
        self.style_classifier = nn.Linear(embed_dim, num_styles)
        
        # Rule scorer (how well input follows rules)
        self.rule_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
        # Storage for explicit rules
        self.rules: List[PoeticRule] = []
        self._init_default_rules()
        
        # Load from file if provided
        if knowledge_path:
            self.load_knowledge(knowledge_path)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def _init_default_rules(self):
        """Initialize default poetic rules."""
        default_rules = [
            PoeticRule(
                name="end_rhyme",
                rule_type="rhyme",
                pattern="line_end_similarity",
                weight=1.0,
                description="Lines should rhyme at the end according to scheme"
            ),
            PoeticRule(
                name="meter_consistency",
                rule_type="meter",
                pattern="syllable_stress_pattern",
                weight=0.8,
                description="Maintain consistent syllable stress patterns"
            ),
            PoeticRule(
                name="line_length",
                rule_type="structure",
                pattern="syllable_count_per_line",
                weight=0.6,
                description="Lines should have appropriate syllable counts"
            ),
            PoeticRule(
                name="grammar_basic",
                rule_type="grammar",
                pattern="sentence_structure",
                weight=0.7,
                description="Basic grammatical correctness"
            ),
            PoeticRule(
                name="imagery",
                rule_type="style",
                pattern="metaphor_simile",
                weight=0.5,
                description="Use of figurative language"
            ),
            PoeticRule(
                name="alliteration",
                rule_type="style",
                pattern="consonant_repetition",
                weight=0.4,
                description="Repetition of initial consonant sounds"
            ),
            PoeticRule(
                name="assonance",
                rule_type="style",
                pattern="vowel_repetition",
                weight=0.4,
                description="Repetition of vowel sounds"
            ),
        ]
        
        self.rules = default_rules
    
    def forward(
        self,
        x: torch.Tensor,
        style_id: Optional[torch.Tensor] = None,
        rule_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass incorporating knowledge.
        
        Args:
            x: Input features [batch_size, seq_len, embed_dim]
            style_id: Target style IDs [batch_size] (optional)
            rule_ids: Rule IDs to apply [batch_size, num_rules] (optional)
            
        Returns:
            Dictionary with:
            - knowledge_features: Knowledge-enhanced features
            - style_logits: Style classification logits
            - rule_scores: Scores for each rule
        """
        batch_size = x.size(0)
        
        # Get style embedding
        if style_id is not None:
            style_emb = self.style_embeddings(style_id)  # [batch, embed_dim]
        else:
            # Use average style
            style_emb = self.style_embeddings.weight.mean(dim=0, keepdim=True)
            style_emb = style_emb.expand(batch_size, -1)
        
        # Get rule embeddings
        if rule_ids is not None:
            rule_emb = self.rule_embeddings(rule_ids)  # [batch, num_rules, embed_dim]
            rule_emb = rule_emb.mean(dim=1)  # Average over rules
        else:
            # Use all rules
            rule_emb = self.rule_embeddings.weight.mean(dim=0, keepdim=True)
            rule_emb = rule_emb.expand(batch_size, -1)
        
        # Encode rules
        rule_features = self.rule_encoder(rule_emb)
        
        # Combine knowledge sources
        knowledge = style_emb + rule_features
        knowledge = self.knowledge_proj(knowledge)
        knowledge = self.layer_norm(knowledge)
        
        # Add knowledge to each position
        knowledge_expanded = knowledge.unsqueeze(1).expand_as(x)
        knowledge_features = x + knowledge_expanded
        knowledge_features = self.dropout(knowledge_features)
        
        # Classify style
        pooled = x.mean(dim=1)  # [batch, embed_dim]
        style_logits = self.style_classifier(pooled)
        
        # Score rule compliance
        rule_input = torch.cat([pooled, rule_emb], dim=-1)
        rule_scores = self.rule_scorer(rule_input)
        
        return {
            'knowledge_features': knowledge_features,
            'style_logits': style_logits,
            'rule_scores': rule_scores,
            'style_embedding': style_emb,
            'rule_embedding': rule_emb
        }
    
    def get_style_embedding(self, style_id: int) -> torch.Tensor:
        """Get embedding for a specific style."""
        idx = torch.tensor([style_id])
        return self.style_embeddings(idx).squeeze(0)
    
    def score_rhyme(self, line1: str, line2: str) -> float:
        """
        Score rhyme quality between two lines.
        
        Args:
            line1: First line
            line2: Second line
            
        Returns:
            Rhyme score (0-1)
        """
        # Extract last words
        words1 = line1.strip().split()
        words2 = line2.strip().split()
        
        if not words1 or not words2:
            return 0.0
        
        last1 = words1[-1].lower().rstrip('.,!?;:')
        last2 = words2[-1].lower().rstrip('.,!?;:')
        
        # Simple phonetic similarity (could be enhanced with CMU dict)
        # Check suffix match
        min_len = min(len(last1), len(last2))
        if min_len < 2:
            return 0.0
        
        for i in range(min_len, 0, -1):
            if last1[-i:] == last2[-i:]:
                return min(1.0, i / 3)  # Normalize
        
        return 0.0
    
    def check_rhyme_scheme(
        self, 
        lines: List[str], 
        scheme: str
    ) -> Dict[str, Any]:
        """
        Check if lines follow a rhyme scheme.
        
        Args:
            lines: List of poem lines
            scheme: Rhyme scheme (e.g., "ABAB")
            
        Returns:
            Dictionary with compliance details
        """
        scheme = scheme.replace(' ', '')
        if len(lines) < len(scheme):
            return {'valid': False, 'score': 0.0, 'details': 'Not enough lines'}
        
        # Group lines by rhyme letter
        rhyme_groups: Dict[str, List[int]] = {}
        for i, letter in enumerate(scheme):
            if letter not in rhyme_groups:
                rhyme_groups[letter] = []
            rhyme_groups[letter].append(i)
        
        # Check rhymes within groups
        total_score = 0
        checks = 0
        
        for letter, indices in rhyme_groups.items():
            if len(indices) > 1:
                for i in range(len(indices) - 1):
                    idx1, idx2 = indices[i], indices[i + 1]
                    if idx1 < len(lines) and idx2 < len(lines):
                        score = self.score_rhyme(lines[idx1], lines[idx2])
                        total_score += score
                        checks += 1
        
        avg_score = total_score / max(1, checks)
        
        return {
            'valid': avg_score > 0.5,
            'score': avg_score,
            'details': f'Checked {checks} rhyme pairs'
        }
    
    def count_syllables(self, text: str) -> int:
        """
        Count syllables in text (simple heuristic).
        
        Args:
            text: Input text
            
        Returns:
            Estimated syllable count
        """
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        
        total = 0
        for word in words:
            # Count vowel groups
            count = len(re.findall(r'[aeiouy]+', word))
            # Handle silent e
            if word.endswith('e') and count > 1:
                count -= 1
            total += max(1, count)
        
        return total
    
    def load_knowledge(self, path: str):
        """
        Load knowledge from JSON file.
        
        Args:
            path: Path to knowledge directory
        """
        path = Path(path)
        
        # Load rules
        rules_file = path / 'poetic_rules.json'
        if rules_file.exists():
            with open(rules_file, 'r') as f:
                rules_data = json.load(f)
            self.rules = [PoeticRule.from_dict(r) for r in rules_data]
    
    def save_knowledge(self, path: str):
        """
        Save knowledge to JSON file.
        
        Args:
            path: Path to save directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save rules
        rules_data = [r.to_dict() for r in self.rules]
        with open(path / 'poetic_rules.json', 'w') as f:
            json.dump(rules_data, f, indent=2)
    
    def get_output_dim(self) -> int:
        return self.embed_dim


if __name__ == "__main__":
    # Test Knowledge Base
    batch_size = 4
    seq_len = 50
    embed_dim = 256
    
    # Create knowledge base
    kb = KnowledgeBase(embed_dim=embed_dim)
    
    # Random input (simulating poem features)
    x = torch.randn(batch_size, seq_len, embed_dim)
    style_id = torch.randint(0, 20, (batch_size,))
    
    # Forward pass
    output = kb(x, style_id)
    
    print(f"Knowledge features shape: {output['knowledge_features'].shape}")
    print(f"Style logits shape: {output['style_logits'].shape}")
    print(f"Rule scores shape: {output['rule_scores'].shape}")
    
    # Test rhyme scoring
    line1 = "Shall I compare thee to a summer's day"
    line2 = "Thou art more lovely and more temperate"
    line3 = "And summer's lease hath all too short a stay"
    
    print(f"\nRhyme score (day/temperate): {kb.score_rhyme(line1, line2):.2f}")
    print(f"Rhyme score (day/stay): {kb.score_rhyme(line1, line3):.2f}")
    
    # Test rhyme scheme
    lines = [
        "Roses are red",
        "Violets are blue",
        "Sugar is sweet",
        "And so are you"
    ]
    result = kb.check_rhyme_scheme(lines, "ABAB")
    print(f"\nRhyme scheme check: {result}")
    
    # Test syllable counting
    print(f"\nSyllables in 'beautiful': {kb.count_syllables('beautiful')}")
    print(f"Syllables in line: {kb.count_syllables(line1)}")
