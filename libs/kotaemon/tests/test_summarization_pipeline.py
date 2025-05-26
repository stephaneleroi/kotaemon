import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kotaemon.llms.summarization import SummarizationPipeline
from kotaemon.base import Document
from kotaemon.llms.base import BaseLLM, LLMInterface
from kotaemon.indices.splitters import TokenSplitter
from kotaemon.loaders.base import BaseReader
from kotaemon.embeddings.base import BaseEmbeddings


# Mock Classes
class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content
        self.text = content  # For compatibility if .text is accessed


class MockLLM(BaseLLM):
    def __init__(self, default_summary_content: str = "mock summary"):
        super().__init__()
        self.default_summary_content = default_summary_content
        # Mock the main method that ainvoke would call internally if not _acall_unlogged
        # For this example, let's assume ainvoke directly or indirectly calls _call_async or a similar method
        # If ainvoke is directly calling an API, then ainvoke itself needs to be an AsyncMock
        # For simplicity, we'll make ainvoke an AsyncMock in tests where needed,
        # or mock the method it calls if that's more direct.
        # Let's give it an internal async method that can be mocked or overridden.
        self._internal_ainvoke = AsyncMock(return_value=MockLLMResponse(self.default_summary_content))

    async def _acall_unlogged(self, messages, **kwargs) -> LLMInterface: # type: ignore
        # This is often the method BaseLLM expects to be implemented
        return await self._internal_ainvoke(messages, **kwargs)

    async def ainvoke(self, prompt: str, **kwargs) -> MockLLMResponse: # type: ignore
        # In a real scenario, ainvoke would format the prompt into messages
        # and call _acall_unlogged or similar.
        # For testing, we can make this directly mockable or have it call the internal mock.
        return await self._internal_ainvoke(prompt, **kwargs)

    def _llm_type(self) -> str:
        return "mock_llm"


class MockReader(BaseReader):
    def __init__(self, documents: list[Document]):
        self.documents = documents

    def load_data(self, *args, **kwargs) -> list[Document]:
        return self.documents


class MockEmbeddings(BaseEmbeddings):
    def __init__(self):
        super().__init__(model_name="mock_embedding_model", model_backend="mock_backend")

    def _get_embedding(self, text: str) -> list[float]:
        return [0.1] * 768 # Dummy embedding

    async def _aget_embedding(self, text: str) -> list[float]:
        return [0.1] * 768

    def _get_texts_embedding(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]

    async def _aget_texts_embedding(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]

    def __call__(self, text_input: str) -> list[float]:
        return self._get_embedding(text_input)


class MockTokenSplitter(TokenSplitter):
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 0):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, model_name="mock_model")
        self.mock_chunk_documents = [Document(text="chunk doc")]

    def split_text(self, text: str) -> list[str]:
        # Simple split for testing
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

    def run(self, documents: list[Document], **kwargs) -> list[Document]:
        # Return a fixed list of chunked documents or implement simple logic
        all_chunks = []
        for i, doc in enumerate(documents):
            # Create a few dummy chunks per document for testing arun
            for j in range(2): # e.g., 2 chunks per doc
                all_chunks.append(Document(text=f"Chunk {j+1} of doc {i+1}", metadata={"original_doc_id": doc.id}))
        return all_chunks if all_chunks else self.mock_chunk_documents


# Test Cases
def test_pipeline_initialization():
    mock_reader = MockReader(documents=[])
    mock_splitter = MockTokenSplitter()
    mock_llm = MockLLM()
    mock_embeddings = MockEmbeddings()

    pipeline = SummarizationPipeline(
        reader=mock_reader,
        text_splitter=mock_splitter,
        llm=mock_llm,
        embedding_model=mock_embeddings,
        max_summaries_per_consolidation_batch=7,
        target_number_of_summaries_for_final_step=2,
        target_summary_length_tokens=1500,
        target_chunk_summary_tokens=200,
    )

    assert pipeline.reader == mock_reader
    assert pipeline.text_splitter == mock_splitter
    assert pipeline.llm == mock_llm
    assert pipeline.embedding_model == mock_embeddings
    assert pipeline.max_summaries_per_consolidation_batch == 7
    assert pipeline.target_number_of_summaries_for_final_step == 2
    assert pipeline.target_summary_length_tokens == 1500
    assert pipeline.target_chunk_summary_tokens == 200


@pytest.mark.asyncio
async def test_summarize_single_chunk_with_context_and_length():
    mock_llm = MockLLM(default_summary_content="Test summary with context and length")
    # We need to mock ainvoke directly on the instance for this test if it's what _asummarize_chunk calls
    mock_llm.ainvoke = AsyncMock(return_value=MockLLMResponse("Test summary with context and length"))
    
    pipeline = SummarizationPipeline(
        reader=MockReader(documents=[]),
        text_splitter=MockTokenSplitter(),
        llm=mock_llm,
        target_chunk_summary_tokens=100 # Default for pipeline if not overridden in call
    )
    
    test_chunk = Document(text="This is a test chunk.")
    previous_context = "Previous context."
    target_len = 50

    summary = await pipeline._asummarize_chunk(
        chunk=test_chunk,
        previous_chunk_summary=previous_context,
        target_tokens=target_len
    )

    assert summary == "Test summary with context and length"
    mock_llm.ainvoke.assert_called_once()
    call_args = mock_llm.ainvoke.call_args
    prompt_arg = call_args[0][0] # Prompt is the first positional argument

    assert previous_context in prompt_arg
    assert f"approximately {target_len} tokens long" in prompt_arg
    assert test_chunk.text in prompt_arg


@pytest.mark.asyncio
async def test_summarize_single_chunk_no_context_no_length():
    mock_llm = MockLLM(default_summary_content="Test summary no context/length")
    mock_llm.ainvoke = AsyncMock(return_value=MockLLMResponse("Test summary no context/length"))

    pipeline = SummarizationPipeline(
        reader=MockReader(documents=[]),
        text_splitter=MockTokenSplitter(),
        llm=mock_llm,
    )
    
    test_chunk = Document(text="This is another test chunk.")

    summary = await pipeline._asummarize_chunk(chunk=test_chunk)

    assert summary == "Test summary no context/length"
    mock_llm.ainvoke.assert_called_once()
    call_args = mock_llm.ainvoke.call_args
    prompt_arg = call_args[0][0]

    assert "Previous context" not in prompt_arg
    assert "tokens long" not in prompt_arg
    assert test_chunk.text in prompt_arg


@pytest.mark.asyncio
@patch.object(SummarizationPipeline, '_agenerate_final_summary', new_callable=AsyncMock)
@patch.object(SummarizationPipeline, '_acombine_summaries_batch', new_callable=AsyncMock)
@patch.object(SummarizationPipeline, '_asummarize_chunk', new_callable=AsyncMock)
async def test_arun_overall_pipeline_flow(
    mock_asummarize_chunk: AsyncMock,
    mock_acombine_summaries_batch: AsyncMock,
    mock_agenerate_final_summary: AsyncMock,
):
    doc1 = Document(text="First document content.", id="doc1")
    doc2 = Document(text="Second document content.", id="doc2")
    
    # MockReader setup
    mock_reader_instance = MockReader(documents=[doc1, doc2])
    mock_reader_instance.load_data = MagicMock(return_value=[doc1, doc2])

    # MockTokenSplitter setup
    # Chunks should have unique IDs or identifiable text for assertion
    chunk1_doc1 = Document(text="Chunk 1 of doc 1", id="c1d1", metadata={"original_doc_id": "doc1"})
    chunk2_doc1 = Document(text="Chunk 2 of doc 1", id="c2d1", metadata={"original_doc_id": "doc1"})
    chunk1_doc2 = Document(text="Chunk 1 of doc 2", id="c1d2", metadata={"original_doc_id": "doc2"})
    chunk2_doc2 = Document(text="Chunk 2 of doc 2", id="c2d2", metadata={"original_doc_id": "doc2"})
    all_chunks = [chunk1_doc1, chunk2_doc1, chunk1_doc2, chunk2_doc2]
    
    mock_splitter_instance = MockTokenSplitter()
    mock_splitter_instance.run = MagicMock(return_value=all_chunks)

    mock_llm_instance = MockLLM() # Not directly used if internal methods are patched

    # Embedding model (mocked but not deeply interacted with for this test)
    mock_embeddings_instance = MockEmbeddings()

    pipeline = SummarizationPipeline(
        reader=mock_reader_instance,
        text_splitter=mock_splitter_instance,
        llm=mock_llm_instance, 
        embedding_model=mock_embeddings_instance,
        max_summaries_per_consolidation_batch=2, # Causes 2 consolidation batches then 1 final
        target_number_of_summaries_for_final_step=1,
        target_summary_length_tokens=500,
        target_chunk_summary_tokens=50
    )

    # Define side effects for mocked summarization methods
    # These need to match the number of calls expected
    chunk_summaries = [f"Summary of {chunk.id}" for chunk in all_chunks]
    mock_asummarize_chunk.side_effect = chunk_summaries

    # Consolidation: 4 chunks -> 2 batches of 2 -> 2 consolidated summaries
    # Then these 2 are combined into 1 for the final step (if target_number_of_summaries_for_final_step is 1)
    # However, the loop `while len(current_summaries) > target_final_count:` will run with 2 summaries.
    # If target_final_count is 1, it will try to consolidate these 2 into 1.
    # So, 1 call to _acombine_summaries_batch if max_batch >= 2
    consolidated_summary1 = "Consolidated summary for c1d1, c2d1"
    consolidated_summary2 = "Consolidated summary for c1d2, c2d2"
    
    # Based on max_summaries_per_consolidation_batch = 2:
    # Pass 1: 
    #   Batch 1: (Summary of c1d1, Summary of c2d1) -> consolidated_summary1
    #   Batch 2: (Summary of c1d2, Summary of c2d2) -> consolidated_summary2
    # current_summaries becomes [consolidated_summary1, consolidated_summary2] (length 2)
    # If target_final_count is 1, loop continues.
    # Pass 2:
    #   Batch 1: (consolidated_summary1, consolidated_summary2) -> final_combined_summary_before_generation
    # current_summaries becomes [final_combined_summary_before_generation] (length 1)
    # Loop terminates. _agenerate_final_summary is called with this.

    final_combined_summary_before_generation = "Final combined summary from pass 2"
    mock_acombine_summaries_batch.side_effect = [
        consolidated_summary1, 
        consolidated_summary2,
        final_combined_summary_before_generation # This call might not happen depending on loop logic
    ]
    
    # Adjusting mock_acombine_summaries_batch based on a more careful trace:
    # Initial summaries: 4
    # Pass 1: batch 1 (chunk_summaries[0], chunk_summaries[1]) -> consolidated_summary1
    #         batch 2 (chunk_summaries[2], chunk_summaries[3]) -> consolidated_summary2
    # current_summaries = [consolidated_summary1, consolidated_summary2]. Length is 2.
    # If target_final_count is 1, loop continues.
    # Pass 2: batch 1 (consolidated_summary1, consolidated_summary2) -> final_combined_summary_before_generation
    # current_summaries = [final_combined_summary_before_generation]. Length is 1. Loop ends.
    # So, 3 calls to _acombine_summaries_batch if the last step is also a consolidation.
    # The prompt implies _agenerate_final_summary is called with the result of the last consolidation pass.

    # Let's assume the loop `while len(current_summaries) > target_final_count` means that
    # if `len(current_summaries)` becomes equal to `target_final_count` after consolidation,
    # then `_agenerate_final_summary` is called with those `target_final_count` summaries.
    # If `max_summaries_per_consolidation_batch` is 2 and `target_final_count` is 1:
    # 4 initial summaries.
    # Pass 1: 2 calls to _acombine_summaries_batch, resulting in 2 summaries.
    # Current summaries length = 2. Target = 1. Loop continues.
    # Pass 2: 1 call to _acombine_summaries_batch, resulting in 1 summary.
    # Current summaries length = 1. Target = 1. Loop terminates.
    # _agenerate_final_summary is called with that 1 summary.

    mock_acombine_summaries_batch.side_effect = [
        consolidated_summary1, # From (Summary of c1d1, Summary of c2d1)
        consolidated_summary2, # From (Summary of c1d2, Summary of c2d2)
        final_combined_summary_before_generation # From (consolidated_summary1, consolidated_summary2)
    ]
    # Actually, the loop structure suggests the final step might be _agenerate_final_summary, not _acombine.
    # If len(current_summaries) becomes <= target_final_count, the loop breaks and _agenerate_final_summary is called.
    # So if 4 summaries -> 2 summaries (2 calls to _acombine). Now current_summaries has length 2.
    # If target_final_count is 1, loop continues.
    # Then these 2 summaries are combined into 1. (1 call to _acombine). current_summaries has length 1.
    # Loop ends. _agenerate_final_summary is called with that single summary.
    # So, 3 calls to _acombine_summaries_batch.
    
    final_summary_text = "This is the ultimate final summary."
    mock_agenerate_final_summary.return_value = final_summary_text
    
    # Reset side_effect for mock_acombine_summaries_batch with correct number of expected calls
    # 4 initial summaries, batch_size=2 -> 2 consolidated summaries (2 calls)
    # These 2 summaries need to be consolidated if target_final_count is 1 -> 1 consolidated summary (1 call)
    # Total 3 calls to _acombine_summaries_batch
    mock_acombine_summaries_batch.side_effect = [
        "Consolidated Batch 1 (Chunks 1-2)", 
        "Consolidated Batch 2 (Chunks 3-4)",
        "Consolidated Batch 3 (Consolidated 1-2)" # This is the summary passed to _agenerate_final_summary
    ]

    # Call arun
    result_document = await pipeline.arun(documents_path="dummy/path")

    # Assertions
    mock_reader_instance.load_data.assert_called_once_with(input_dir="dummy/path") # or file="dummy/path" depending on AutoReader/DirectoryReader
    mock_splitter_instance.run.assert_called_once_with([doc1, doc2])

    # _asummarize_chunk calls
    assert mock_asummarize_chunk.call_count == len(all_chunks) # 4 chunks
    
    # Contextual Passing Check for _asummarize_chunk:
    # Call 1 (chunk1_doc1): previous_summary=None
    mock_asummarize_chunk.assert_any_call(
        chunk1_doc1, previous_chunk_summary=None, target_tokens=pipeline.target_chunk_summary_tokens
    )
    # Call 2 (chunk2_doc1): previous_summary="Summary of c1d1"
    mock_asummarize_chunk.assert_any_call(
        chunk2_doc1, previous_chunk_summary=chunk_summaries[0], target_tokens=pipeline.target_chunk_summary_tokens
    )
    # Call 3 (chunk1_doc2): previous_summary="Summary of c2d1"
    mock_asummarize_chunk.assert_any_call(
        chunk1_doc2, previous_chunk_summary=chunk_summaries[1], target_tokens=pipeline.target_chunk_summary_tokens
    )
    # Call 4 (chunk2_doc2): previous_summary="Summary of c1d2"
    mock_asummarize_chunk.assert_any_call(
        chunk2_doc2, previous_chunk_summary=chunk_summaries[2], target_tokens=pipeline.target_chunk_summary_tokens
    )
    
    # _acombine_summaries_batch calls
    # Expected calls:
    # 1. For chunk_summaries[0] and chunk_summaries[1]
    # 2. For chunk_summaries[2] and chunk_summaries[3]
    # 3. For the results of call 1 and call 2
    assert mock_acombine_summaries_batch.call_count == 3
    mock_acombine_summaries_batch.assert_any_call([chunk_summaries[0], chunk_summaries[1]])
    mock_acombine_summaries_batch.assert_any_call([chunk_summaries[2], chunk_summaries[3]])
    mock_acombine_summaries_batch.assert_any_call([
        "Consolidated Batch 1 (Chunks 1-2)", 
        "Consolidated Batch 2 (Chunks 3-4)"
    ])

    # _agenerate_final_summary call
    # It should be called with the result of the last consolidation.
    mock_agenerate_final_summary.assert_called_once_with(["Consolidated Batch 3 (Consolidated 1-2)"])
    
    # Final result check
    assert result_document.content == final_summary_text
    assert result_document.metadata["source"] == "SummarizationPipeline"
    assert result_document.metadata["total_chunks_generated"] == len(all_chunks)
    assert result_document.metadata["initial_chunk_summaries_count"] == len(all_chunks)
    assert result_document.metadata["target_summary_length_tokens"] == pipeline.target_summary_length_tokens
    
    # Check if embedding_model was "used" (passed around or an attribute accessed)
    # For now, its existence and being set is tested in initialization.
    # If a method on it was called, we'd mock that on mock_embeddings_instance and assert_called.
    # Example: if pipeline.embedding_model() was called somewhere.
    # For this test, we assume it's not directly called in a way that needs mocking beyond __init__.

    # Check that the reader call was adjusted for DirectoryReader type if it was used
    # The current mock_reader_instance is a base MockReader.
    # If pipeline.reader was an instance of DirectoryReader, load_data would be called with input_dir.
    # If it was AutoReader, it'd be file=
    # The SummarizationPipeline itself performs this check.
    # For this test, `input_dir` will be used if the mock_reader_instance was set to be a DirectoryReader.
    # Let's refine this. The pipeline receives a reader instance. It doesn't change its type.
    # The pipeline's `arun` has logic:
    # if isinstance(self.reader, DirectoryReader): docs = self.reader.load_data(input_dir=path)
    # elif isinstance(self.reader, AutoReader): docs = self.reader.load_data(file=path)
    # else: docs = self.reader.load_data(file=path)
    # Since mock_reader_instance is neither DirectoryReader nor AutoReader, it will fall to the else case.
    mock_reader_instance.load_data.assert_called_once_with(file="dummy/path")

    # To test the DirectoryReader path, we would need to make mock_reader_instance a DirectoryReader
    # Example:
    # from kotaemon.loaders import DirectoryReader
    # mock_reader_instance = DirectoryReader(input_dir="dummy_path_not_used")
    # mock_reader_instance.load_data = MagicMock(return_value=[doc1, doc2])
    # ... then pipeline init ...
    # mock_reader_instance.load_data.assert_called_once_with(input_dir="dummy/path")
    # This detail is more about testing the SummarizationPipeline's internal reader handling
    # than the overall flow of summarization logic. The current test is okay for the generic BaseReader path.

```
