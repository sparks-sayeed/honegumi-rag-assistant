"""
Gradio interface for Honegumi RAG Assistant.

This app provides a web UI for the Honegumi RAG Assistant, allowing users to:
- Enter Bayesian optimization problem descriptions in natural language
- Select Claude models for different agents
- View generated code with syntax highlighting
- Download generated Python scripts
- Use example prompts for quick testing

Deployment: Designed for Hugging Face Spaces with Gradio SDK.
"""

import os
import sys
from pathlib import Path
import tempfile
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import gradio as gr
except ImportError:
    print("ERROR: gradio is not installed. Install with: pip install gradio")
    sys.exit(1)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

from honegumi_rag_assistant.orchestrator import run_from_text
from honegumi_rag_assistant.app_config import settings, DEFAULT_MODEL


# Claude models offered in the UI dropdowns.
MODEL_CHOICES = ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"]

# Example prompts for quick testing.  Each row supplies the three model
# dropdowns and the review checkbox alongside the prompt.
EXAMPLE_PROMPTS = [
    "Optimize temperature (50-200°C) and pressure (1-10 bar) for maximum yield in a chemical reaction.",
    "Find optimal composition ratios for a ternary alloy (Element A: 0-100%, Element B: 0-100%, Element C: 0-100%) to maximize tensile strength while minimizing cost.",
    "Optimize learning rate (1e-5 to 1e-1, log scale) and batch size (16, 32, 64, 128) for training a neural network to minimize validation loss.",
    "Optimize reaction time (1-24 hours) and catalyst concentration (0.1-5.0 mol/L) to maximize conversion efficiency in a batch reactor.",
    "Design a solar panel configuration with panel angle (0-90 degrees) and spacing (0.5-3 meters) to maximize energy output while minimizing land area.",
]

EXAMPLES = [
    [prompt, DEFAULT_MODEL, DEFAULT_MODEL, DEFAULT_MODEL, False]
    for prompt in EXAMPLE_PROMPTS
]


def generate_code(
    problem: str,
    api_key: str,
    param_selector_model: str,
    retrieval_planner_model: str,
    code_writer_model: str,
    enable_review: bool,
    progress=gr.Progress()
) -> tuple[str, str]:
    """
    Generate Bayesian optimization code from a problem description.
    
    Args:
        problem: Natural language problem description
        api_key: Anthropic API key
        param_selector_model: Model for Parameter Selector agent
        retrieval_planner_model: Model for Retrieval Planner agent
        code_writer_model: Model for Code Writer agent
        enable_review: Whether to enable the Reviewer agent
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (generated_code, status_message)
    """
    if not problem or not problem.strip():
        return "", "⚠️ Please enter a problem description."
    
    if not api_key or not api_key.strip():
        return "", "⚠️ Please enter your Anthropic API key."
    
    try:
        # Update settings with user selections
        settings.anthropic_api_key = api_key.strip()
        settings.model_name = param_selector_model
        settings.retrieval_planner_model = retrieval_planner_model
        settings.code_writer_model = code_writer_model
        settings.stream_code = not enable_review  # Enable streaming when review is disabled
        
        # Update progress
        progress(0.1, desc="Initializing pipeline...")
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            progress(0.2, desc="Analyzing problem...")
            
            # Run the pipeline
            code = run_from_text(
                problem=problem,
                output_dir=temp_dir,
                debug=False,  # Set to False for cleaner output in UI
                enable_review=enable_review
            )
            
            progress(1.0, desc="Code generation complete!")
            
            # Success message with model info
            status = f"✅ Code generated successfully using {code_writer_model}"
            if enable_review:
                status += " (with review)"
            
            return code, status
            
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\n"
        if "api_key" in str(e).lower() or "authentication" in str(e).lower():
            error_msg += "Please check that your Anthropic API key is valid."
        else:
            error_msg += "Full traceback:\n" + traceback.format_exc()
        return "", error_msg


def create_interface():
    """Create and configure the Gradio interface."""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .output-code {
        font-family: 'Monaco', 'Menlo', 'Consolas', monospace !important;
        font-size: 13px !important;
    }
    """
    
    with gr.Blocks(
        title="Honegumi RAG Assistant",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        gr.Markdown("""
        # 🤖 Honegumi RAG Assistant
        
        **Agentic Code Generation for Bayesian Optimization**
        
        Transform natural language problem descriptions into production-ready Bayesian optimization code using Meta's [Ax Platform](https://ax.dev/).
        
        This assistant uses LangGraph to orchestrate multiple specialized AI agents that:
        1. Extract optimization parameters from your description
        2. Generate skeleton code using [Honegumi](https://honegumi.readthedocs.io/)
        3. Retrieve relevant Ax documentation via RAG
        4. Generate complete, runnable Python code
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔑 Configuration")
                
                api_key = gr.Textbox(
                    label="Anthropic API Key",
                    type="password",
                    placeholder="sk-ant-...",
                    info="Your Anthropic API key (required for Claude models)"
                )

                gr.Markdown("### 🎯 Model Selection")
                gr.Markdown("*Choose models for different agents (Haiku is faster and cheaper; Opus is the most capable)*")

                param_selector_model = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=DEFAULT_MODEL,
                    label="Parameter Selector",
                    info="Extracts optimization parameters"
                )

                retrieval_planner_model = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=DEFAULT_MODEL,
                    label="Retrieval Planner",
                    info="Plans documentation queries"
                )

                code_writer_model = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=DEFAULT_MODEL,
                    label="Code Writer",
                    info=f"Generates final code (recommended: {DEFAULT_MODEL})"
                )
                
                enable_review = gr.Checkbox(
                    label="Enable Code Review",
                    value=False,
                    info="Slower but may improve quality"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Problem Description")
                
                problem = gr.Textbox(
                    label="Describe your Bayesian optimization problem",
                    placeholder="Example: Optimize temperature (50-200°C) and pressure (1-10 bar) for maximum yield in a chemical reaction.",
                    lines=6,
                    info="Describe what you want to optimize, including parameters, ranges, and objectives"
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                    generate_btn = gr.Button("🚀 Generate Code", variant="primary", scale=3)
        
        gr.Markdown("### 📋 Example Prompts")
        gr.Markdown("*Click an example to try it out:*")
        
        gr.Examples(
            examples=EXAMPLES,
            inputs=[problem, param_selector_model, retrieval_planner_model, code_writer_model, enable_review],
            label=None,
        )
        
        gr.Markdown("---")
        gr.Markdown("### 💻 Generated Code")
        
        status = gr.Textbox(
            label="Status",
            interactive=False,
            lines=2
        )
        
        code_output = gr.Code(
            label="Generated Python Script",
            language="python",
            lines=20,
            elem_classes=["output-code"]
        )
        
        download_btn = gr.DownloadButton(
            label="📥 Download Python File",
            visible=False
        )
        
        gr.Markdown("""
        ---
        ### ℹ️ About
        
        **Honegumi RAG Assistant** is an agentic AI system that generates high-quality Bayesian optimization code.
        
        - **Repository**: [github.com/hasan-sayeed/honegumi_rag_assistant](https://github.com/hasan-sayeed/honegumi_rag_assistant)
        - **Documentation**: [Honegumi Docs](https://honegumi.readthedocs.io/)
        - **Powered by**: [LangGraph](https://github.com/langchain-ai/langgraph), [Ax Platform](https://ax.dev/), Anthropic Claude models
        
        **Note**: This tool requires an Anthropic API key. Typical cost per generation: $0.08-$0.17 with Claude Sonnet 5, depending on problem complexity and how much documentation is retrieved.
        
        **Citation**:
        ```bibtex
        @software{honegumi_rag_assistant2025,
          title = {Honegumi RAG Assistant: Agentic Code Generation for Bayesian Optimization},
          author = {Sayeed, Hasan Muhammad},
          year = {2025},
          url = {https://github.com/hasan-sayeed/honegumi_rag_assistant}
        }
        ```
        """)
        
        # Event handlers
        def on_generate(problem, api_key, ps_model, rp_model, cw_model, review):
            """Handle generate button click."""
            code, status_msg = generate_code(
                problem, api_key, ps_model, rp_model, cw_model, review
            )
            
            # Show download button only if code was generated
            download_visible = bool(code and code.strip())
            
            # Save code to temporary file for download
            temp_file = None
            if download_visible:
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.py',
                    delete=False,
                    prefix='bayesian_opt_'
                )
                temp_file.write(code)
                temp_file.close()
            
            return (
                code,
                status_msg,
                gr.DownloadButton(visible=download_visible, value=temp_file.name if temp_file else None)
            )
        
        def on_clear():
            """Handle clear button click."""
            return "", "", gr.DownloadButton(visible=False)
        
        generate_btn.click(
            fn=on_generate,
            inputs=[problem, api_key, param_selector_model, retrieval_planner_model, code_writer_model, enable_review],
            outputs=[code_output, status, download_btn],
            api_name="generate"
        )
        
        clear_btn.click(
            fn=on_clear,
            outputs=[code_output, status, download_btn]
        )
    
    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    
    # Launch with configuration suitable for HF Spaces
    demo.queue()  # Enable queuing for better handling of concurrent requests
    demo.launch(
        server_name="0.0.0.0",  # Listen on all interfaces (required for HF Spaces)
        server_port=7860,  # Default port for HF Spaces
        share=False,  # Don't create a public link (HF Spaces provides its own)
    )
