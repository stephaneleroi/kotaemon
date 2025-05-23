import asyncio
import unittest
from unittest.mock import MagicMock, patch

# Adjust import path if ktem is not directly in PYTHONPATH
# Assuming ktem is discoverable by the test runner
from ktem.pages.summarization import SummarizationPage
from kotaemon.llms.summarization import SummarizationPipeline # Ensure this is the correct import for the class being mocked

# Helper to run async tests with unittest's default runner
def async_test(f):
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

class TestSummarizationPage(unittest.TestCase):

    def setUp(self):
        self.mock_app = MagicMock()
        self.mock_app.llms.get_default_name.return_value = "mock_default_llm"
        # _build_ui is called in SummarizationPage.__init__
        # We need to patch it or ensure its dependencies are mocked if it's complex.
        # For now, assume it's simple enough or its Gradio calls will be mocked by mock_gr.
        # Alternatively, patch 'ktem.pages.summarization.SummarizationPage._build_ui'
        self.patcher_build_ui = patch('ktem.pages.summarization.SummarizationPage._build_ui', return_value=None)
        self.mock_build_ui = self.patcher_build_ui.start()
        self.addCleanup(self.patcher_build_ui.stop)


    # --- Test Cases for Dynamic Chunk Size Calculation ---

    @patch('ktem.pages.summarization.SummarizationPipeline')
    @patch('ktem.pages.summarization.gr') # Mock gradio UI calls
    @async_test
    async def test_dynamic_chunk_size_with_num_ctx(self, mock_gr, MockSummarizationPipeline):
        # Arrange
        mock_llm_instance = MagicMock()
        mock_llm_instance.num_ctx = 4096
        mock_llm_instance._obj = None # Ensure _obj.num_ctx is not used
        self.mock_app.llms.get_default.return_value = mock_llm_instance
        
        page = SummarizationPage(self.mock_app)
        
        # Act
        await page._handle_summarize(doc_path="dummy.txt", target_pages=1, batch_size=5)

        # Assert
        MockSummarizationPipeline.assert_called_once()
        _, called_kwargs = MockSummarizationPipeline.call_args
        self.assertEqual(called_kwargs.get('text_splitter_chunk_size'), 2048) # 4096 * 0.5
        mock_gr.Info.assert_any_call("Using dynamic chunk size: 2048 (50% of model context: 4096)")

    @patch('ktem.pages.summarization.SummarizationPipeline')
    @patch('ktem.pages.summarization.gr')
    @async_test
    async def test_dynamic_chunk_size_with_obj_num_ctx(self, mock_gr, MockSummarizationPipeline):
        # Arrange
        mock_llm_instance = MagicMock(num_ctx=None) # Ensure direct num_ctx is not used
        mock_llm_instance._obj = MagicMock(num_ctx=8000)
        self.mock_app.llms.get_default.return_value = mock_llm_instance
        
        page = SummarizationPage(self.mock_app)
        
        # Act
        await page._handle_summarize(doc_path="dummy.txt", target_pages=1, batch_size=5)

        # Assert
        MockSummarizationPipeline.assert_called_once()
        _, called_kwargs = MockSummarizationPipeline.call_args
        self.assertEqual(called_kwargs.get('text_splitter_chunk_size'), 4000) # 8000 * 0.5
        mock_gr.Info.assert_any_call("Using dynamic chunk size: 4000 (50% of model context: 8000)")

    @patch('ktem.pages.summarization.SummarizationPipeline')
    @patch('ktem.pages.summarization.gr')
    @async_test
    async def test_dynamic_chunk_size_clamped_to_512(self, mock_gr, MockSummarizationPipeline):
        # Arrange
        mock_llm_instance = MagicMock(num_ctx=1000) # 1000 * 0.5 = 500, which is < 512
        mock_llm_instance._obj = None
        self.mock_app.llms.get_default.return_value = mock_llm_instance
        
        page = SummarizationPage(self.mock_app)
        
        # Act
        await page._handle_summarize(doc_path="dummy.txt", target_pages=1, batch_size=5)

        # Assert
        MockSummarizationPipeline.assert_called_once()
        _, called_kwargs = MockSummarizationPipeline.call_args
        self.assertEqual(called_kwargs.get('text_splitter_chunk_size'), 512) # Clamped
        mock_gr.Info.assert_any_call("Using dynamic chunk size: 512 (50% of model context: 1000)")

    @patch('ktem.pages.summarization.SummarizationPipeline')
    @patch('ktem.pages.summarization.gr')
    @async_test
    async def test_dynamic_chunk_size_fallback_no_num_ctx(self, mock_gr, MockSummarizationPipeline):
        # Arrange
        mock_llm_instance = MagicMock()
        # Deliberately remove num_ctx and _obj or set _obj to something without num_ctx
        del mock_llm_instance.num_ctx 
        mock_llm_instance._obj = None # or MagicMock(spec=[]) to ensure no _obj.num_ctx
        self.mock_app.llms.get_default.return_value = mock_llm_instance
        
        page = SummarizationPage(self.mock_app)
        
        # Act
        await page._handle_summarize(doc_path="dummy.txt", target_pages=1, batch_size=5)

        # Assert
        MockSummarizationPipeline.assert_called_once()
        _, called_kwargs = MockSummarizationPipeline.call_args
        self.assertEqual(called_kwargs.get('text_splitter_chunk_size'), 2000) # Fallback
        mock_gr.Warning.assert_any_call("LLM attribute 'num_ctx' not found or not an integer. Using default chunk size: 2000")

    @patch('ktem.pages.summarization.SummarizationPipeline')
    @patch('ktem.pages.summarization.gr')
    @async_test
    async def test_dynamic_chunk_size_fallback_num_ctx_not_int(self, mock_gr, MockSummarizationPipeline):
        # Arrange
        mock_llm_instance = MagicMock(num_ctx="not-an-integer")
        mock_llm_instance._obj = None
        self.mock_app.llms.get_default.return_value = mock_llm_instance
        
        page = SummarizationPage(self.mock_app)
        
        # Act
        await page._handle_summarize(doc_path="dummy.txt", target_pages=1, batch_size=5)

        # Assert
        MockSummarizationPipeline.assert_called_once()
        _, called_kwargs = MockSummarizationPipeline.call_args
        self.assertEqual(called_kwargs.get('text_splitter_chunk_size'), 2000) # Fallback
        mock_gr.Warning.assert_any_call("LLM attribute 'num_ctx' not found or not an integer. Using default chunk size: 2000")

    @patch('ktem.pages.summarization.SummarizationPipeline')
    @patch('ktem.pages.summarization.gr')
    @async_test
    async def test_dynamic_chunk_size_fallback_exception_accessing_num_ctx(self, mock_gr, MockSummarizationPipeline):
        # Arrange
        mock_llm_instance = MagicMock()
        # Make num_ctx access raise an exception
        type(mock_llm_instance).num_ctx = MagicMock(side_effect=AttributeError("Test exception"))
        mock_llm_instance._obj = None # Ensure _obj.num_ctx is not the fallback path here
        self.mock_app.llms.get_default.return_value = mock_llm_instance
        
        page = SummarizationPage(self.mock_app)
        
        # Act
        await page._handle_summarize(doc_path="dummy.txt", target_pages=1, batch_size=5)

        # Assert
        MockSummarizationPipeline.assert_called_once()
        _, called_kwargs = MockSummarizationPipeline.call_args
        self.assertEqual(called_kwargs.get('text_splitter_chunk_size'), 2000) # Fallback
        mock_gr.Warning.assert_any_call("Error accessing LLM context window: Test exception. Using default chunk size: 2000")


    # --- Test Cases for Page-to-Token Conversion ---

    @patch('ktem.pages.summarization.SummarizationPipeline')
    @patch('ktem.pages.summarization.gr')
    @async_test
    async def test_page_to_token_conversion_1_page(self, mock_gr, MockSummarizationPipeline):
        # Arrange
        mock_llm_instance = MagicMock(num_ctx=4096) # Default LLM for chunk size part
        self.mock_app.llms.get_default.return_value = mock_llm_instance
        page = SummarizationPage(self.mock_app)
        
        # Act
        await page._handle_summarize(doc_path="dummy.txt", target_pages=1, batch_size=5)

        # Assert
        MockSummarizationPipeline.assert_called_once()
        _, called_kwargs = MockSummarizationPipeline.call_args
        self.assertEqual(called_kwargs.get('target_llm_token_length'), 333) # 1 * 333
        mock_gr.Info.assert_any_call("Target summary length: 1 page(s) (~333 tokens)")

    @patch('ktem.pages.summarization.SummarizationPipeline')
    @patch('ktem.pages.summarization.gr')
    @async_test
    async def test_page_to_token_conversion_3_pages(self, mock_gr, MockSummarizationPipeline):
        # Arrange
        mock_llm_instance = MagicMock(num_ctx=4096)
        self.mock_app.llms.get_default.return_value = mock_llm_instance
        page = SummarizationPage(self.mock_app)
        
        # Act
        await page._handle_summarize(doc_path="dummy.txt", target_pages=3, batch_size=5)

        # Assert
        MockSummarizationPipeline.assert_called_once()
        _, called_kwargs = MockSummarizationPipeline.call_args
        self.assertEqual(called_kwargs.get('target_llm_token_length'), 999) # 3 * 333
        mock_gr.Info.assert_any_call("Target summary length: 3 page(s) (~999 tokens)")

    @patch('ktem.pages.summarization.SummarizationPipeline')
    @patch('ktem.pages.summarization.gr')
    @async_test
    async def test_page_to_token_conversion_0_pages(self, mock_gr, MockSummarizationPipeline):
        # Arrange
        mock_llm_instance = MagicMock(num_ctx=4096)
        self.mock_app.llms.get_default.return_value = mock_llm_instance
        page = SummarizationPage(self.mock_app)
        
        # Act
        await page._handle_summarize(doc_path="dummy.txt", target_pages=0, batch_size=5)

        # Assert
        MockSummarizationPipeline.assert_called_once()
        _, called_kwargs = MockSummarizationPipeline.call_args
        self.assertEqual(called_kwargs.get('target_llm_token_length'), 0) # 0 * 333
        mock_gr.Info.assert_any_call("Target summary length: 0 page(s) (~0 tokens)")
    
    # --- Test case for FileNotFoundError ---
    @patch('ktem.pages.summarization.SummarizationPipeline')
    @patch('ktem.pages.summarization.gr')
    @async_test
    async def test_handle_summarize_file_not_found(self, mock_gr, MockSummarizationPipeline):
        # Arrange
        mock_llm_instance = MagicMock(num_ctx=4096)
        self.mock_app.llms.get_default.return_value = mock_llm_instance
        
        # Make the pipeline raise FileNotFoundError
        MockSummarizationPipeline.return_value.arun.side_effect = FileNotFoundError("File not found for test")
        
        page = SummarizationPage(self.mock_app)
        doc_path = "non_existent_path.txt"

        # Act
        summary, metadata = await page._handle_summarize(doc_path, target_pages=1, batch_size=5)

        # Assert
        mock_gr.Error.assert_called_once_with(f"Error: Document or directory not found at '{doc_path}'.")
        self.assertEqual(summary, f"Error: Document or directory not found at '{doc_path}'.")
        self.assertIsNone(metadata)

    # --- Test case for general Exception ---
    @patch('ktem.pages.summarization.SummarizationPipeline')
    @patch('ktem.pages.summarization.gr')
    @async_test
    async def test_handle_summarize_general_exception(self, mock_gr, MockSummarizationPipeline):
        # Arrange
        mock_llm_instance = MagicMock(num_ctx=4096)
        self.mock_app.llms.get_default.return_value = mock_llm_instance
        
        # Make the pipeline raise a general Exception
        error_message = "General pipeline error for test"
        MockSummarizationPipeline.return_value.arun.side_effect = Exception(error_message)
        
        page = SummarizationPage(self.mock_app)
        doc_path = "dummy_path.txt"

        # Act
        summary, metadata = await page._handle_summarize(doc_path, target_pages=1, batch_size=5)

        # Assert
        mock_gr.Error.assert_called_once_with(f"Error during summarization: {error_message}")
        self.assertEqual(summary, f"Error during summarization: {error_message}")
        self.assertIsNone(metadata)

    # --- Test case for empty doc_path ---
    @patch('ktem.pages.summarization.SummarizationPipeline')
    @patch('ktem.pages.summarization.gr')
    @async_test
    async def test_handle_summarize_empty_doc_path(self, mock_gr, MockSummarizationPipeline):
        # Arrange
        page = SummarizationPage(self.mock_app)
        
        # Act
        summary, metadata = await page._handle_summarize(doc_path="", target_pages=1, batch_size=5)

        # Assert
        mock_gr.Warning.assert_called_once_with("Document path cannot be empty.")
        self.assertEqual(summary, "")
        self.assertIsNone(metadata)
        MockSummarizationPipeline.assert_not_called() # Pipeline should not be called

if __name__ == '__main__':
    unittest.main()
