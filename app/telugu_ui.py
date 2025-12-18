"""
Telugu Poem AI - Web Interface
English UI with Telugu poem generation and interpretation.
"""

import gradio as gr
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.telugu_backbone import create_telugu_generator
from src.preprocessing.telugu_cleaner import TeluguTextCleaner


# Global model instances
generator = None
cleaner = TeluguTextCleaner()

# Poetry styles
STYLES = {
    "Aata Veladi (ఆట వెలది)": "traditional",
    "Kandham (కందం)": "classical",
    "Utpalamala (ఉత్పలమాల)": "epic",
    "Free Verse (వచన కవిత)": "modern",
    "Folk Song (జానపద)": "folk",
    "Devotional (భక్తి)": "devotional"
}


def load_models():
    """Load Telugu pre-trained models."""
    global generator
    
    if generator is None:
        print("Loading Telugu model...")
        try:
            generator = create_telugu_generator('distilmbert', freeze_backbone=True)
            generator.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    return generator


def generate_telugu_poem(
    prompt: str,
    style: str,
    max_lines: int,
    temperature: float,
    creativity: float
) -> str:
    """Generate Telugu poem based on prompt and style."""
    
    if not prompt.strip():
        return "Please enter a starting line in Telugu"
    
    gen = load_models()
    if gen is None:
        return "Model loading failed. Please try again."
    
    max_length = max(50, max_lines * 15)
    top_p = 0.7 + (creativity * 0.25)
    
    try:
        generated = gen.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        
        # Clean up output
        lines = generated.split('\n')
        lines = [l.strip() for l in lines if l.strip()]
        
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        
        return '\n'.join(lines)
    
    except Exception as e:
        return f"Error: {str(e)}"


def analyze_telugu_poem(poem_text: str) -> str:
    """Analyze Telugu poem for praasa, meter, and style."""
    
    if not poem_text.strip():
        return "Please enter a Telugu poem to analyze"
    
    stats = cleaner.get_stats(poem_text)
    
    analysis = []
    analysis.append("## 📊 Poem Analysis\n")
    
    # Basic stats
    analysis.append(f"**Number of Lines:** {stats['num_lines']}")
    analysis.append(f"**Telugu Characters:** {stats['telugu_characters']}")
    analysis.append(f"**Telugu Ratio:** {stats['telugu_ratio']:.0%}")
    
    # Aksharas per line
    analysis.append(f"\n**Syllables per Line:** {stats['aksharas_per_line']}")
    analysis.append(f"**Average Syllables:** {stats['avg_aksharas_per_line']:.1f}")
    
    # Praasa analysis
    analysis.append("\n### 🎵 Praasa (Rhyme) Analysis")
    praasa = stats['praasa']
    
    if praasa['has_praasa']:
        analysis.append(f"✅ **Has Praasa:** Yes - '{praasa['praasa_akshara']}'")
    else:
        analysis.append(f"❌ **Has Praasa:** No")
    
    analysis.append(f"**Match Ratio:** {praasa['match_ratio']:.0%}")
    
    # Meter guess
    analysis.append("\n### 📏 Chandassu (Meter)")
    avg = stats['avg_aksharas_per_line']
    
    if 18 <= avg <= 22:
        analysis.append("- **Detected Meter:** Utpalamala / Champakamala")
    elif 8 <= avg <= 12:
        analysis.append("- **Detected Meter:** Kandham / Aataveladi")
    else:
        analysis.append("- **Detected Meter:** Free Verse")
    
    return '\n'.join(analysis)


def improve_telugu_poem(poem_text: str, focus: str) -> str:
    """Suggest improvements for Telugu poem."""
    
    if not poem_text.strip():
        return "Please enter a Telugu poem to get suggestions"
    
    stats = cleaner.get_stats(poem_text)
    suggestions = []
    
    suggestions.append("## ✨ Improvement Suggestions\n")
    
    if focus == "Praasa (Rhyme)":
        suggestions.append("### Praasa Improvement Tips")
        praasa = stats['praasa']
        
        if not praasa['has_praasa']:
            suggestions.append("- The second syllable (akshara) of each line should be the same")
            suggestions.append("- Example: చందమామ, మందిరం (both have 'ం' as 2nd syllable)")
            suggestions.append("- Try rewriting lines to match the second akshara")
        else:
            suggestions.append("- Your poem has good praasa! ✅")
    
    elif focus == "Meter (Chandassu)":
        suggestions.append("### Meter Improvement Tips")
        aksharas = stats['aksharas_per_line']
        avg = stats['avg_aksharas_per_line']
        
        suggestions.append(f"- Current average: {avg:.1f} syllables per line")
        
        for i, count in enumerate(aksharas):
            if abs(count - avg) > 3:
                if count > avg:
                    suggestions.append(f"- Line {i+1}: Too long ({count} syllables) - try shortening")
                else:
                    suggestions.append(f"- Line {i+1}: Too short ({count} syllables) - try expanding")
    
    elif focus == "Imagery (Alankaram)":
        suggestions.append("### Imagery Enhancement Tips")
        suggestions.append("- Use similes (upamana) - comparing using 'like' or 'as'")
        suggestions.append("- Use metaphors (rupaka) - direct comparison")
        suggestions.append("- Add sensory details - sight, sound, touch, smell")
        suggestions.append("- Example: 'చందమామ వంటి ముఖము' (face like the moon)")
    
    if len(suggestions) == 1:
        suggestions.append("Your poem looks good! No major improvements needed.")
    
    return '\n'.join(suggestions)


def create_ui():
    """Create Gradio interface with English UI."""
    
    with gr.Blocks(
        title="Telugu Poetry AI",
        theme=gr.themes.Soft(
            primary_hue="orange",
            secondary_hue="amber"
        )
    ) as demo:
        gr.Markdown("""
        # 🎭 Telugu Poetry AI
        ### CNN-Based Poem Learning & Interpretation Inspired by Human Rote Learning
        
        Generate, analyze, and improve Telugu poems using AI.
        """)
        
        with gr.Tabs():
            # Tab 1: Generate
            with gr.TabItem("✍️ Generate Poem"):
                with gr.Row():
                    with gr.Column():
                        prompt_input = gr.Textbox(
                            label="Starting Line (in Telugu)",
                            placeholder="చందమామ రావే...",
                            lines=2
                        )
                        style_select = gr.Dropdown(
                            choices=list(STYLES.keys()),
                            value="Free Verse (వచన కవిత)",
                            label="Poetry Style"
                        )
                        
                        with gr.Row():
                            max_lines_slider = gr.Slider(
                                minimum=4,
                                maximum=16,
                                value=8,
                                step=1,
                                label="Maximum Lines"
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
                            label="Generated Telugu Poem",
                            lines=12,
                            show_copy_button=True
                        )
                
                generate_btn.click(
                    fn=generate_telugu_poem,
                    inputs=[prompt_input, style_select, max_lines_slider, 
                           temp_slider, creativity_slider],
                    outputs=generated_output
                )
                
                gr.Examples(
                    examples=[
                        ["చందమామ రావే", "Folk Song (జానపద)", 6, 0.7, 0.5],
                        ["తెలుగు వెలుగు", "Free Verse (వచన కవిత)", 8, 0.8, 0.6],
                        ["అమ్మ అనే మాట", "Kandham (కందం)", 8, 0.7, 0.4],
                    ],
                    inputs=[prompt_input, style_select, max_lines_slider, 
                           temp_slider, creativity_slider]
                )
            
            # Tab 2: Analyze
            with gr.TabItem("📊 Analyze Poem"):
                with gr.Row():
                    with gr.Column():
                        analyze_input = gr.Textbox(
                            label="Your Telugu Poem",
                            placeholder="Enter Telugu poem here...",
                            lines=10
                        )
                        analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                    
                    with gr.Column():
                        analysis_output = gr.Markdown(label="Analysis Results")
                
                analyze_btn.click(
                    fn=analyze_telugu_poem,
                    inputs=analyze_input,
                    outputs=analysis_output
                )
                
                gr.Examples(
                    examples=[
                        ["""చందమామ రావే
జాబిల్లి రావే
నీ పాప వచ్చెను
పాలాళి తెచ్చెను"""],
                        ["""ఉప్పు కప్పురంబు నొక్కపోలికనుండు
చూడ చూడ రుచులు జాడ వేరు
పురుషులందు పుణ్య పురుషులు వేరయా
విశ్వదాభిరామ వినురవేమ"""],
                    ],
                    inputs=analyze_input
                )
            
            # Tab 3: Improve
            with gr.TabItem("✨ Improve Poem"):
                with gr.Row():
                    with gr.Column():
                        improve_input = gr.Textbox(
                            label="Your Telugu Poem",
                            placeholder="Enter Telugu poem here...",
                            lines=10
                        )
                        focus_select = gr.Radio(
                            choices=["Praasa (Rhyme)", "Meter (Chandassu)", "Imagery (Alankaram)"],
                            value="Praasa (Rhyme)",
                            label="Improvement Focus"
                        )
                        improve_btn = gr.Button("💡 Get Suggestions", variant="primary")
                    
                    with gr.Column():
                        improvement_output = gr.Markdown(label="Suggestions")
                
                improve_btn.click(
                    fn=improve_telugu_poem,
                    inputs=[improve_input, focus_select],
                    outputs=improvement_output
                )
            
            # Tab 4: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About This Project
                
                **Telugu Poetry AI** uses a novel CNN-based architecture inspired by how humans 
                memorize poetry through repetition (rote learning).
                
                ### Key Features
                
                | Feature | Description |
                |---------|-------------|
                | **Pre-trained Model** | DistilBERT Multilingual (supports Telugu) |
                | **Praasa Analysis** | Detects Telugu rhyme patterns |
                | **Meter Detection** | Identifies Chandassu based on syllable count |
                | **Memory Module** | Stores repetitive patterns (Novel) |
                | **Feedback Loop** | Iterative refinement (Novel) |
                
                ### Supported Poetry Styles
                
                - **Aata Veladi** - Used by Vemana
                - **Kandham** - Traditional meter
                - **Utpalamala / Champakamala** - Epic poetry
                - **Folk Songs** - Traditional children's songs
                - **Modern Free Verse**
                
                ### Telugu Prosody Concepts
                
                - **Praasa** - Second syllable rhyme
                - **Yati** - Pause positions
                - **Ganaalu** - Metrical feet (laghu/guru patterns)
                
                ---
                
                **Author:** Maneendra  
                **Project:** CNN-Based Poem Learning & Interpretation
                """)
        
        gr.Markdown("""
        ---
        Made with ❤️ for Telugu Poetry | తెలుగు భాష వర్ధిల్లాలి 🙏
        """)
    
    return demo


def main():
    """Launch the UI."""
    print("=" * 60)
    print("🎭 Starting Telugu Poetry AI")
    print("=" * 60)
    
    print("\nLoading models...")
    load_models()
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
