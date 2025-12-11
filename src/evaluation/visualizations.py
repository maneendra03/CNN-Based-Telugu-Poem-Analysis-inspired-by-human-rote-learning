"""
Visualization Module
Creates visualizations for training progress and model analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


class ResultsVisualizer:
    """
    Creates visualizations for poem learning experiments.
    
    Visualizations:
    - Training curves
    - Memorization curves
    - Attention maps
    - Feature distributions
    """
    
    def __init__(self, save_dir: str = "figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    def plot_training_curves(
        self,
        history: List[Dict],
        save_name: str = "training_curves"
    ):
        """
        Plot training and validation loss curves.
        
        Args:
            history: List of epoch dictionaries with 'train' and 'val' losses
            save_name: Filename for saving
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train']['loss'] for h in history]
        
        # Training loss
        axes[0].plot(epochs, train_loss, color=self.colors[0], linewidth=2, label='Train')
        
        if 'val' in history[0] and history[0]['val']:
            val_loss = [h['val'].get('loss', 0) for h in history]
            axes[0].plot(epochs, val_loss, color=self.colors[1], linewidth=2, label='Validation')
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        
        # Perplexity (if available)
        if 'val' in history[0] and history[0]['val'] and 'perplexity' in history[0]['val']:
            perplexity = [h['val'].get('perplexity', 0) for h in history]
            axes[1].plot(epochs, perplexity, color=self.colors[2], linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Perplexity')
            axes[1].set_title('Validation Perplexity')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training curves to {self.save_dir / save_name}.png")
    
    def plot_memorization_curve(
        self,
        curve_data: Dict[str, List[float]],
        save_name: str = "memorization_curve"
    ):
        """
        Plot the memorization curve - a key visualization for the rote learning paradigm.
        
        Args:
            curve_data: Dictionary with 'epochs', 'retention', 'recognition', 'recall'
            save_name: Filename for saving
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = curve_data['epochs']
        
        ax.plot(epochs, curve_data['retention'], 
                color=self.colors[0], linewidth=2.5, label='Memory Retention', marker='o')
        ax.plot(epochs, curve_data['recognition'], 
                color=self.colors[1], linewidth=2.5, label='Pattern Recognition', marker='s')
        ax.plot(epochs, curve_data['recall'], 
                color=self.colors[2], linewidth=2.5, label='Recall Accuracy', marker='^')
        
        # Add threshold line
        ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.7, label='Target Threshold')
        
        ax.set_xlabel('Training Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Rote Learning Memorization Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved memorization curve to {self.save_dir / save_name}.png")
    
    def plot_attention_map(
        self,
        attention_weights: np.ndarray,
        source_tokens: List[str],
        target_tokens: List[str],
        save_name: str = "attention_map"
    ):
        """
        Plot attention visualization.
        
        Args:
            attention_weights: Attention matrix [tgt_len, src_len]
            source_tokens: Source token labels
            target_tokens: Target token labels
            save_name: Filename for saving
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        
        # Labels
        ax.set_xticks(range(len(source_tokens)))
        ax.set_yticks(range(len(target_tokens)))
        ax.set_xticklabels(source_tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(target_tokens, fontsize=8)
        
        ax.set_xlabel('Source (Input Poem)')
        ax.set_ylabel('Target (Generated)')
        ax.set_title('Attention Map')
        
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_comparison(
        self,
        metrics: Dict[str, Dict[str, float]],
        save_name: str = "metrics_comparison"
    ):
        """
        Plot comparison of metrics across different models/experiments.
        
        Args:
            metrics: Dict of {model_name: {metric_name: value}}
            save_name: Filename for saving
        """
        models = list(metrics.keys())
        metric_names = list(next(iter(metrics.values())).keys())
        
        x = np.arange(len(metric_names))
        width = 0.8 / len(models)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, model in enumerate(models):
            values = [metrics[model].get(m, 0) for m in metric_names]
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model, color=self.colors[i % len(self.colors)])
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_memory_strength_distribution(
        self,
        memory_strengths: List[float],
        save_name: str = "memory_strength_dist"
    ):
        """
        Plot distribution of memory cell strengths.
        
        Args:
            memory_strengths: List of memory strength values
            save_name: Filename for saving
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        axes[0].hist(memory_strengths, bins=20, color=self.colors[0], edgecolor='white', alpha=0.8)
        axes[0].axvline(np.mean(memory_strengths), color='red', linestyle='--', label=f'Mean: {np.mean(memory_strengths):.2f}')
        axes[0].set_xlabel('Memory Strength')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Memory Strength Distribution')
        axes[0].legend()
        
        # Bar plot for individual cells
        cell_indices = range(len(memory_strengths))
        axes[1].bar(cell_indices, memory_strengths, color=self.colors[1], edgecolor='white')
        axes[1].axhline(1.0, color='gray', linestyle='--', alpha=0.7, label='Initial Value')
        axes[1].set_xlabel('Memory Cell Index')
        axes[1].set_ylabel('Strength')
        axes[1].set_title('Memory Cell Strengths')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_paper_figure(
        self,
        history: List[Dict],
        memorization_curve: Dict[str, List[float]],
        final_metrics: Dict[str, float],
        save_name: str = "paper_figure"
    ):
        """
        Create a comprehensive figure suitable for paper publication.
        
        Args:
            history: Training history
            memorization_curve: Memorization curve data
            final_metrics: Final evaluation metrics
            save_name: Filename for saving
        """
        fig = plt.figure(figsize=(14, 10))
        
        # Training curve (top left)
        ax1 = fig.add_subplot(2, 2, 1)
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train']['loss'] for h in history]
        ax1.plot(epochs, train_loss, color=self.colors[0], linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('(a) Training Progress')
        
        # Memorization curve (top right)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(memorization_curve['epochs'], memorization_curve['retention'],
                 color=self.colors[0], linewidth=2, label='Retention')
        ax2.plot(memorization_curve['epochs'], memorization_curve['recognition'],
                 color=self.colors[1], linewidth=2, label='Recognition')
        ax2.plot(memorization_curve['epochs'], memorization_curve['recall'],
                 color=self.colors[2], linewidth=2, label='Recall')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('(b) Memorization Curve')
        ax2.legend(loc='lower right')
        ax2.set_ylim(0, 1)
        
        # Metrics bar chart (bottom left)
        ax3 = fig.add_subplot(2, 2, 3)
        metrics_to_plot = ['bleu', 'rhyme_accuracy', 'meter_consistency', 'distinct_1']
        available_metrics = {k: v for k, v in final_metrics.items() if k in metrics_to_plot}
        
        x = range(len(available_metrics))
        ax3.bar(x, list(available_metrics.values()), color=self.colors[:len(available_metrics)])
        ax3.set_xticks(x)
        ax3.set_xticklabels([m.replace('_', '\n') for m in available_metrics.keys()])
        ax3.set_ylabel('Score')
        ax3.set_title('(c) Evaluation Metrics')
        ax3.set_ylim(0, 1)
        
        # Architecture schematic (bottom right) - placeholder
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.text(0.5, 0.5, 'CNN → Hierarchical RNN → Memory & Attention\n↓\nKnowledge Base → Feedback Loop → Decoder',
                 ha='center', va='center', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('(d) Architecture Overview')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.save_dir / f"{save_name}.pdf", bbox_inches='tight')  # PDF for paper
        plt.close()
        
        print(f"Saved paper figure to {self.save_dir / save_name}.png and .pdf")


if __name__ == "__main__":
    # Test visualizations
    visualizer = ResultsVisualizer(save_dir="test_figures")
    
    # Generate sample data
    history = []
    for epoch in range(50):
        history.append({
            'epoch': epoch,
            'train': {'loss': 2.0 * np.exp(-epoch / 10) + np.random.random() * 0.1},
            'val': {
                'loss': 2.2 * np.exp(-epoch / 10) + np.random.random() * 0.1,
                'perplexity': 100 * np.exp(-epoch / 15) + np.random.random() * 5
            }
        })
    
    # Plot training curves
    visualizer.plot_training_curves(history)
    
    # Memorization curve
    curve_data = {
        'epochs': list(range(50)),
        'retention': [0.3 + 0.7 * (1 - np.exp(-e / 15)) for e in range(50)],
        'recognition': [0.4 + 0.6 * (1 - np.exp(-e / 12)) for e in range(50)],
        'recall': [0.2 + 0.8 * (1 - np.exp(-e / 18)) for e in range(50)]
    }
    visualizer.plot_memorization_curve(curve_data)
    
    # Final metrics
    final_metrics = {
        'bleu': 0.65,
        'rhyme_accuracy': 0.78,
        'meter_consistency': 0.72,
        'distinct_1': 0.45,
        'distinct_2': 0.62,
        'perplexity': 25.3
    }
    
    # Paper figure
    visualizer.create_paper_figure(history, curve_data, final_metrics)
    
    print("All test visualizations created successfully!")
