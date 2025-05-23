import asyncio
import gradio as gr
from kotaemon.llms.summarization import SummarizationPipeline
from kotaemon.schema import Document

class SummarizationPage:
    def __init__(self, app):
        self._app = app
        # Define UI elements here so they can be accessed by other methods if needed
        self.doc_path_input = None
        self.target_length_slider = None
        self.batch_size_input = None
        self.summarize_button = None
        self.summary_output_textbox = None
        self.metadata_output_json = None
        
        self._build_ui()

    async def _handle_summarize(self, doc_path, target_pages, batch_size):
        if not doc_path:
            gr.Warning("Document path cannot be empty.")
            return "", None # Return empty values for outputs

        try:
            # Ensure integer types for pipeline parameters
            batch_size = int(batch_size)

            TOKENS_PER_PAGE = 333  # Conversion factor: (250 words/page * 1.33 tokens/word)
            target_pages_int = int(target_pages) # Ensure it's an integer
            target_llm_token_length = target_pages_int * TOKENS_PER_PAGE
            gr.Info(f"Target summary length: {target_pages_int} page(s) (~{target_llm_token_length} tokens)")

            # Get LLM instance from the app's LLM pool
            # In a real app, self._app.llms would be populated.
            # For standalone testing, self._app.llms[llm_choice] will be handled by MockLLMPool
            llm_instance = self._app.llms.get_default()

            dynamic_chunk_size = 2000  # Default fallback chunk size
            try:
                if hasattr(llm_instance, 'num_ctx') and isinstance(llm_instance.num_ctx, int):
                    # Use 50% of the context window for chunk size, ensure it's at least a minimum value (e.g. 512)
                    calculated_size = int(llm_instance.num_ctx * 0.5)
                    dynamic_chunk_size = max(calculated_size, 512) 
                    gr.Info(f"Using dynamic chunk size: {dynamic_chunk_size} (50% of model context: {llm_instance.num_ctx})")
                elif hasattr(llm_instance, '_obj') and hasattr(llm_instance._obj, 'num_ctx') and isinstance(llm_instance._obj.num_ctx, int):
                    # Fallback for some Langchain wrappers that might store it in _obj.num_ctx
                    calculated_size = int(llm_instance._obj.num_ctx * 0.5)
                    dynamic_chunk_size = max(calculated_size, 512)
                    gr.Info(f"Using dynamic chunk size: {dynamic_chunk_size} (50% of model context: {llm_instance._obj.num_ctx})")
                else:
                    gr.Warning(f"LLM attribute 'num_ctx' not found or not an integer. Using default chunk size: {dynamic_chunk_size}")
            except Exception as e:
                gr.Warning(f"Error accessing LLM context window: {str(e)}. Using default chunk size: {dynamic_chunk_size}")

            pipeline = SummarizationPipeline(
                llm=llm_instance,
                target_llm_token_length=target_llm_token_length,
                text_splitter_chunk_size=dynamic_chunk_size,
                consolidation_batch_size=batch_size,
            )
            
            default_llm_name = "default LLM"
            try:
                default_llm_name = self._app.llms.get_default_name()
            except Exception:
                gr.Warning("Could not determine default LLM name. Using generic name.")
            gr.Info(f"Starting summarization for: {doc_path} with LLM: {default_llm_name}")
            # Assuming pipeline.arun can take a path directly.
            # If it expects content, file loading logic would be needed here.
            # For now, let's assume it handles file paths or directory paths.
            result_doc = await pipeline.arun(path=doc_path)
            
            summary_text = result_doc.text
            metadata = result_doc.metadata
            
            gr.Info("Summarization complete.")
            return summary_text, metadata

        except FileNotFoundError:
            error_message = f"Error: Document or directory not found at '{doc_path}'."
            gr.Error(error_message)
            return error_message, None
        except Exception as e:
            error_message = f"Error during summarization: {str(e)}"
            gr.Error(error_message)
            print(f"Full exception: {e}") # For server-side logging
            return error_message, None


    def _build_ui(self):
        with gr.Blocks():
            gr.Markdown("# Document Summarization")
            
            with gr.Row():
                self.doc_path_input = gr.Textbox(
                    label="Document Path or Directory",
                    placeholder="Enter path to a document or a directory (e.g., /path/to/file.txt or /path/to/docs/)",
                    elem_id="summarization_doc_path_input"
                )

            with gr.Accordion("Configuration", open=True):
                self.target_length_slider = gr.Slider(
                    label="Target Summary Length (pages)",
                    minimum=1,
                    maximum=10,
                    value=2,
                    step=1,
                    elem_id="summarization_target_length_slider"
                )
                self.batch_size_input = gr.Number(
                    label="Consolidation Batch Size",
                    value=5, 
                    minimum=1,
                    step=1,
                    elem_id="summarization_batch_size_input",
                    precision=0 # Ensure integer input
                )

            self.summarize_button = gr.Button(
                "Summarize", 
                variant="primary",
                elem_id="summarization_summarize_button"
            )

            with gr.Group():
                gr.Markdown("### Summary Output")
                self.summary_output_textbox = gr.Textbox(
                    label="Summary",
                    interactive=False,
                    lines=15, 
                    elem_id="summarization_summary_output"
                )
            
            with gr.Accordion("Metadata", open=False):
                self.metadata_output_json = gr.JSON(
                    label="Processing Metadata",
                    elem_id="summarization_metadata_output"
                )

            # Connect button to handler
            self.summarize_button.click(
                self._handle_summarize, # Directly call _handle_summarize
                inputs=[
                    self.doc_path_input,
                    self.target_length_slider,
                    self.batch_size_input
                ],
                outputs=[
                    self.summary_output_textbox,
                    self.metadata_output_json
                ]
            )

if __name__ == "__main__":
    # Mock classes for standalone testing
    class MockLLM:
        async def arun(self, text: str, **kwargs): # Match SummarizationPipeline's LLM call if it uses 'text'
            # Simulate LLM processing for summarization
            await asyncio.sleep(0.1) # Simulate async work
            summary_content = f"Summary of '{text[:50]}...' (mock)"
            # The pipeline itself constructs the Document, so this arun is what pipeline's LLM would do
            return summary_content # For map-reduce, LLM returns string

    class MockSummarizationLLM: # This is the object ModelPool would return
        def __init__(self, name="mock_llm"):
            self.name = name
            # This mock LLM needs to be suitable for SummarizationPipeline's internal use.
            # The pipeline might call methods like .ainvoke, .arun, .agenerate etc. on it.
            # For simplicity, let's assume it needs an `ainvoke` or similar for text generation.
            # Based on SummarizationPipeline, it seems to pass the LLM instance directly to other LangChain components.
            # So, the methods called on it would depend on how map_reduce_chain_factory etc. use it.
            # Let's provide a generic `ainvoke` which is common in LangChain.
        async def ainvoke(self, prompt, **kwargs):
            await asyncio.sleep(0.05)
            # This mock is for the LLM *within* the pipeline, not the pipeline itself.
            return f"Mocked LLM response for prompt: {str(prompt)[:50]}..."

        # If the pipeline uses `agenerate` or other methods, they'd need to be mocked here.
        # For example, if it's a LangChain LLM object:
        async def agenerate_prompt(self, prompts, **kwargs):
            results = []
            for _ in prompts:
                await asyncio.sleep(0.05)
                results.append( # Assuming LLMResult structure if needed by LangChain
                    type('LLMResult', (), {'generations': [type('Generation', (), {'text': 'mock generation'})()]})()
                )
            return type('LLMResult', (), {'generations': results})()


    class MockLLMPool:
        def __init__(self):
            self._llms = {
                "ollama/mock-llama2": MockSummarizationLLM(name="ollama/mock-llama2"),
                "openai/mock-gpt-3.5-turbo": MockSummarizationLLM(name="openai/mock-gpt-3.5-turbo"),
            }
        def options(self):
            return list(self._llms.keys())
        
        def get_default_name(self):
            return self.options()[0]
        
        def __getitem__(self, key):
            print(f"[MockLLMPool] Requested LLM: {key}")
            if key in self._llms:
                return self._llms[key]
            raise KeyError(f"Mock LLM '{key}' not found.")

    class MockApp:
        def __init__(self):
            self.llms = MockLLMPool()

    # Create a dummy file for testing FileNotFoundError
    dummy_file_for_testing = "dummy_test_doc.txt"
    with open(dummy_file_for_testing, "w") as f:
        f.write("This is a dummy document for testing summarization. It has enough text to be processed by the mock pipeline.")

    print(f"Created dummy file: {dummy_file_for_testing}")
    print("To test, enter './{dummy_file_for_testing}' or an absolute path in the UI.")
    
    mock_app = MockApp()
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        SummarizationPage(mock_app)
    
    print("Launching Gradio demo for SummarizationPage...")
    # For running in environments where loop is managed (like Jupyter/Colab or other async contexts)
    # demo.launch(prevent_thread_lock=True) 
    demo.launch() # This will block if not in such an environment.

    # Clean up the dummy file (optional)
    # import os
    # os.remove(dummy_file_for_testing)
    # print(f"Cleaned up dummy file: {dummy_file_for_testing}")
