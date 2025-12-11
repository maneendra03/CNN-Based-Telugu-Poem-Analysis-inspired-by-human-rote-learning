"""
Gradio Web UI for Poem Learner
Interactive interface for poem generation, analysis, and style transfer.
"""

import gradio as gr
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.pretrained_backbone import GPT2PoemGenerator
from src.preprocessing.text_cleaner import TextCleaner
from src.models.knowledge_base import KnowledgeBase


# Global model instances
generator = None
cleaner = TextCleaner()
knowledge_base = KnowledgeBase(embed_dim=256)

# Style mapping
STYLES = {
    "Shakespearean": 0,
    "Romantic": 1,
    "Modern": 2,
    "Dickinson": 3,
    "Free Verse": 4
}


def load_models():
    """Load pre-trained models."""
    global generator
    
    if generator is None:
        print("Loading GPT-2 poem generator...")
        generator = GPT2PoemGenerator(model_name='gpt2', freeze_backbone=True)
        generator.eval()
        print("Models loaded!")
    
    return generator


def generate_poem(
    prompt: str,
    style: str,
    max_lines: int,
    temperature: float,
    creativity: float
) -> str:
    """
    Generate a poem based on prompt and style.
    """
    if not prompt.strip():
        return "Please enter a prompt to generate a poem."
    
    # Load model if needed
    gen = load_models()
    
    # Calculate parameters
    max_length = max(50, max_lines * 15)  # ~15 tokens per line
    top_p = 0.7 + (creativity * 0.25)  # Higher creativity = more diverse
    
    try:
        # Generate poem
        generated = gen.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=top_p,
            style_id=STYLES.get(style, 0)
        )
        
        # Clean up output
        lines = generated.split('\n')
        lines = [l.strip() for l in lines if l.strip()]
        
        # Limit to max_lines
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        
        return '\n'.join(lines)
    
    except Exception as e:
        return f"Error generating poem: {str(e)}"


def analyze_poem(poem_text: str) -> str:
    """
    Analyze a poem for rhyme, meter, and style.
    """
    if not poem_text.strip():
        return "Please enter a poem to analyze."
    
    lines = [l.strip() for l in poem_text.split('\n') if l.strip()]
    
    if len(lines) < 2:
        return "Please enter at least 2 lines for analysis."
    
    analysis = []
    analysis.append("## 📊 Poem Analysis\n")
    
    # Basic stats
    analysis.append(f"**Lines:** {len(lines)}")
    word_count = sum(len(l.split()) for l in lines)
    analysis.append(f"**Words:** {word_count}")
    
    # Syllables per line
    syllables = [cleaner.count_syllables(l) for l in lines]
    avg_syllables = sum(syllables) / len(syllables)
    analysis.append(f"**Avg syllables/line:** {avg_syllables:.1f}")
    
    # Rhyme analysis
    analysis.append("\n### 🎵 Rhyme Analysis")
    
    rhyme_pairs = []
    for i in range(len(lines) - 1):
        score = knowledge_base.score_rhyme(lines[i], lines[i + 1])
        if score > 0.3:
            rhyme_pairs.append((i + 1, i + 2, score))
    
    # Alternate rhymes
    for i in range(len(lines) - 2):
        score = knowledge_base.score_rhyme(lines[i], lines[i + 2])
        if score > 0.3:
            rhyme_pairs.append((i + 1, i + 3, score))
    
    if rhyme_pairs:
        for l1, l2, score in rhyme_pairs[:5]:
            analysis.append(f"- Lines {l1} & {l2}: {score:.0%} match")
    else:
        analysis.append("- No strong rhymes detected (free verse?)")
    
    # Meter consistency
    analysis.append("\n### 📏 Meter Analysis")
    variance = sum((s - avg_syllables) ** 2 for s in syllables) / len(syllables)
    consistency = max(0, 1 - (variance ** 0.5) / avg_syllables) if avg_syllables > 0 else 0
    
    analysis.append(f"**Meter consistency:** {consistency:.0%}")
    
    if consistency > 0.8:
        analysis.append("- Very consistent meter (formal poetry)")
    elif consistency > 0.5:
        analysis.append("- Moderate meter variation")
    else:
        analysis.append("- Free meter (contemporary style)")
    
    # Style guess
    analysis.append("\n### 🎭 Style Detection")
    
    if consistency > 0.8 and len(rhyme_pairs) > 2:
        analysis.append("- Likely style: **Traditional/Formal**")
    elif any("thee" in l or "thou" in l for l in lines):
        analysis.append("- Likely style: **Shakespearean/Elizabethan**")
    elif avg_syllables < 6:
        analysis.append("- Likely style: **Imagist/Minimalist**")
    else:
        analysis.append("- Likely style: **Modern/Contemporary**")
    
    return '\n'.join(analysis)


def improve_poem(poem_text: str, focus: str) -> str:
    """
    Suggest improvements for a poem.
    """
    if not poem_text.strip():
        return "Please enter a poem to improve."
    
    lines = [l.strip() for l in poem_text.split('\n') if l.strip()]
    
    suggestions = []
    suggestions.append("## ✨ Improvement Suggestions\n")
    
    if focus == "Rhyme":
        suggestions.append("### Making it Rhyme")
        
        # Find non-rhyming pairs
        for i in range(0, len(lines) - 1, 2):
            if i + 1 < len(lines):
                score = knowledge_base.score_rhyme(lines[i], lines[i + 1])
                if score < 0.3:
                    last_word = lines[i].split()[-1] if lines[i].split() else ""
                    suggestions.append(f"- Lines {i+1}-{i+2} don't rhyme. Try ending line {i+2} with a word that rhymes with '{last_word}'")
    
    elif focus == "Meter":
        suggestions.append("### Improving Meter")
        syllables = [cleaner.count_syllables(l) for l in lines]
        avg = sum(syllables) / len(syllables)
        
        for i, (line, syl) in enumerate(zip(lines, syllables)):
            diff = abs(syl - avg)
            if diff > 2:
                if syl > avg:
                    suggestions.append(f"- Line {i+1} is too long ({syl} syllables). Try removing some words.")
                else:
                    suggestions.append(f"- Line {i+1} is too short ({syl} syllables). Try adding descriptive words.")
    
    elif focus == "Imagery":
        suggestions.append("### Adding Imagery")
        suggestions.append("- Consider adding sensory details (sight, sound, smell, touch, taste)")
        suggestions.append("- Use metaphors to compare abstract ideas to concrete objects")
        suggestions.append("- Add specific color words instead of generic descriptions")
        
        # Check for weak words
        weak_words = ['nice', 'good', 'bad', 'very', 'really', 'thing']
        found_weak = []
        for word in weak_words:
            if word in poem_text.lower():
                found_weak.append(word)
        
        if found_weak:
            suggestions.append(f"\n**Weak words to replace:** {', '.join(found_weak)}")
    
    elif focus == "Flow":
        suggestions.append("### Improving Flow")
        suggestions.append("- Read the poem aloud to check natural rhythm")
        suggestions.append("- Consider enjambment (continuing sentences across lines)")
        suggestions.append("- Vary sentence length for dynamic pacing")
        
        # Check for repetition
        words = poem_text.lower().split()
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        
        repeated = [w for w, c in word_counts.items() if c > 2 and len(w) > 4]
        if repeated:
            suggestions.append(f"\n**Repeated words to vary:** {', '.join(repeated[:5])}")
    
    if len(suggestions) == 1:
        suggestions.append("Your poem looks good! Keep writing.")
    
    return '\n'.join(suggestions)


def create_ui():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(
        title="Poem Learner AI",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="pink"
        )
    ) as demo:
        gr.Markdown("""
        # 🎭 CNN-Based Poem Learning & Interpretation
        ### Inspired by Human Rote Learning
        
        This AI system combines pre-trained language models with novel cognitive-inspired 
        architecture for poem generation and understanding.
        """)
        
        with gr.Tabs():
            # Tab 1: Generate
            with gr.TabItem("✍️ Generate Poem"):
                with gr.Row():
                    with gr.Column():
                        prompt_input = gr.Textbox(
                            label="Starting Prompt",
                            placeholder="Enter a line to start your poem...",
                            lines=2
                        )
                        style_select = gr.Dropdown(
                            choices=list(STYLES.keys()),
                            value="Modern",
                            label="Poetry Style"
                        )
                        
                        with gr.Row():
                            max_lines_slider = gr.Slider(
                                minimum=4,
                                maximum=20,
                                value=8,
                                step=1,
                                label="Max Lines"
                            )
                            temp_slider = gr.Slider(
                                minimum=0.5,
                                maximum=1.5,
                                value=0.8,
                                step=0.1,
                                label="Temperature"
                            )
                        
                        creativity_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.1,
                            label="Creativity Level"
                        )
                        
                        generate_btn = gr.Button("🎨 Generate Poem", variant="primary")
                    
                    with gr.Column():
                        generated_output = gr.Textbox(
                            label="Generated Poem",
                            lines=12,
                            show_copy_button=True
                        )
                
                generate_btn.click(
                    fn=generate_poem,
                    inputs=[prompt_input, style_select, max_lines_slider, 
                           temp_slider, creativity_slider],
                    outputs=generated_output
                )
                
                # Example prompts
                gr.Examples(
                    examples=[
                        ["Roses are red,", "Romantic", 6, 0.7, 0.5],
                        ["In the silence of night,", "Modern", 8, 0.8, 0.6],
                        ["Shall I compare thee", "Shakespearean", 10, 0.7, 0.4],
                        ["The city sleeps while I", "Free Verse", 8, 0.9, 0.7],
                    ],
                    inputs=[prompt_input, style_select, max_lines_slider, 
                           temp_slider, creativity_slider]
                )
            
            # Tab 2: Analyze
            with gr.TabItem("📊 Analyze Poem"):
                with gr.Row():
                    with gr.Column():
                        analyze_input = gr.Textbox(
                            label="Paste Your Poem",
                            placeholder="Enter a poem to analyze...",
                            lines=10
                        )
                        analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                    
                    with gr.Column():
                        analysis_output = gr.Markdown(label="Analysis Results")
                
                analyze_btn.click(
                    fn=analyze_poem,
                    inputs=analyze_input,
                    outputs=analysis_output
                )
                
                gr.Examples(
                    examples=[
                        ["""Shall I compare thee to a summer's day?
Thou art more lovely and more temperate.
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date."""],
                        ["""The fog comes
on little cat feet.
It sits looking
over harbor and city
on silent haunches
and then moves on."""],
                    ],
                    inputs=analyze_input
                )
            
            # Tab 3: Improve
            with gr.TabItem("✨ Improve Poem"):
                with gr.Row():
                    with gr.Column():
                        improve_input = gr.Textbox(
                            label="Your Poem",
                            placeholder="Enter a poem to get improvement suggestions...",
                            lines=10
                        )
                        focus_select = gr.Radio(
                            choices=["Rhyme", "Meter", "Imagery", "Flow"],
                            value="Rhyme",
                            label="Improvement Focus"
                        )
                        improve_btn = gr.Button("💡 Get Suggestions", variant="primary")
                    
                    with gr.Column():
                        improvement_output = gr.Markdown(label="Suggestions")
                
                improve_btn.click(
                    fn=improve_poem,
                    inputs=[improve_input, focus_select],
                    outputs=improvement_output
                )
            
            # Tab 4: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About This Project
                
                **CNN-Based Poem Learning & Interpretation Inspired by Human Rote Learning**
                
                This project implements a novel neural architecture that mimics how humans 
                memorize and learn poetry through repetition.
                
                ### Key Innovations
                
                | Component | Description |
                |-----------|-------------|
                | **Rote Learning Memory** | LSTM cells that reinforce repeated patterns |
                | **Hierarchical Understanding** | Character, word, and line-level processing |
                | **Knowledge-Grounded Feedback** | Poetic rules integration for refinement |
                | **Memorization Curve Metric** | Novel evaluation for learning efficiency |
                
                ### Architecture
                
                ```
                Input → GPT-2/BERT Backbone → Memory & Attention
                     → Hierarchical RNN → Feedback Loop → Decoder → Output
                ```
                
                ### Technical Stack
                
                - **Pre-trained Models**: GPT-2, BERT
                - **Framework**: PyTorch
                - **UI**: Gradio
                - **Custom Modules**: Rote Learning Memory, Feedback Loop
                
                ---
                
                *Research project for publication*
                """)
        
        gr.Markdown("""
        ---
        Made with ❤️ using PyTorch, Transformers, and Gradio
        """)
    
    return demo


def main():
    """Launch the UI."""
    print("=" * 60)
    print("🎭 Starting Poem Learner UI")
    print("=" * 60)
    
    # Pre-load models
    print("\nLoading models (this may take a moment)...")
    load_models()
    
    # Create and launch UI
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
