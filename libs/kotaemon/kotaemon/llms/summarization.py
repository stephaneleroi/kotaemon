import asyncio
from typing import Any, AsyncGenerator, Iterator, List, Optional, Union, Dict, Tuple

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from kotaemon.base import BaseComponent, Document, LLMInterface
from kotaemon.loaders.base import BaseReader
from kotaemon.loaders import DirectoryReader, AutoReader
from kotaemon.indices.splitters import BaseSplitter, TokenSplitter, DynamicSemanticSplitter
from kotaemon.llms.base import BaseLLM
from kotaemon.llms.prompts import PromptTemplate
from kotaemon.embeddings.base import BaseEmbeddings, DocumentWithEmbedding # Added for future use


# Define a common retry strategy for LLM calls
# Retry up to 3 times, with exponential backoff starting at 1s, max 10s.
# Retry on general Exception, but one might want to be more specific
# (e.g., specific API error types from the LLM client if available).
DEFAULT_LLM_RETRY_DECORATOR = retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception) # Or a more specific LLM API error
)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    # Ensure v1 and v2 are numpy arrays and normalized for robust similarity
    vec1 = np.array(v1, dtype=float)
    vec2 = np.array(v2, dtype=float)
    
    norm_v1 = np.linalg.norm(vec1)
    norm_v2 = np.linalg.norm(vec2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0 # Or handle as an error/special case
    
    dot_product = np.dot(vec1, vec2)
    return dot_product / (norm_v1 * norm_v2)


class SummarizationPipeline(BaseComponent):
    '''
    A pipeline for summarizing a collection of documents.

    It supports contextual summarization by passing the summary of the
    previous chunk to the current one, and allows for length guidance for
    both individual chunk summaries and the final output.
    '''
    
    reader: Union[BaseReader, DirectoryReader, AutoReader]
    text_splitter: BaseSplitter # Updated type hint
    llm: BaseLLM
    embedding_model: Optional[BaseEmbeddings] = None
    # target_summary_length: str = "5 pages" # Example, will need better handling
    max_summaries_per_consolidation_batch: int
    target_number_of_summaries_for_final_step: int
    target_summary_length_tokens: int
    target_chunk_summary_tokens: int
    target_consolidated_summary_tokens: int

    def __init__(
        self,
        reader: Union[BaseReader, DirectoryReader, AutoReader],
        text_splitter: BaseSplitter, # Updated type hint
        llm: BaseLLM,
        embedding_model: Optional[BaseEmbeddings] = None,
        max_summaries_per_consolidation_batch: int = 5,
        target_number_of_summaries_for_final_step: int = 1,
        target_summary_length_tokens: int = 2000,
        target_chunk_summary_tokens: int = 250,
        target_consolidated_summary_tokens: int = 500,
        use_semantic_context_for_chunks: bool = True,
        semantic_context_top_k: int = 1,
        # target_summary_length: str = "5 pages", # Example
        **kwargs,
    ):
        """
        Initializes the SummarizationPipeline.

        Args:
            reader: Component to load documents.
            text_splitter: Component to split documents into chunks.
            llm: Language model to use for summarization.
            embedding_model: Optional embedding model for semantic context
                enhancements. Required if use_semantic_context_for_chunks is True.
            max_summaries_per_consolidation_batch: Max summaries to combine
                in one LLM call during consolidation.
            target_number_of_summaries_for_final_step: Target number of
                summaries before the final summarization step.
            target_summary_length_tokens: Target token length for the
                final summary.
            target_chunk_summary_tokens: Target token length for individual
                chunk summaries.
            target_consolidated_summary_tokens: Target token length for
                consolidated summaries during intermediate steps.
            use_semantic_context_for_chunks: Whether to use semantic similarity
                to find relevant past chunk summaries for context.
            semantic_context_top_k: How many top similar past chunk summaries
                to use for context if use_semantic_context_for_chunks is True.
            **kwargs: Additional keyword arguments for BaseComponent.
        """
        super().__init__(**kwargs)
        self.reader = reader
        self.text_splitter = text_splitter
        self.llm = llm
        self.embedding_model = embedding_model
        self.max_summaries_per_consolidation_batch = (
            max_summaries_per_consolidation_batch
        )
        self.target_number_of_summaries_for_final_step = (
            target_number_of_summaries_for_final_step
        )
        self.target_summary_length_tokens = target_summary_length_tokens
        self.target_chunk_summary_tokens = target_chunk_summary_tokens
        self.target_consolidated_summary_tokens = target_consolidated_summary_tokens
        self.use_semantic_context_for_chunks = use_semantic_context_for_chunks
        self.semantic_context_top_k = semantic_context_top_k
        # self.target_summary_length = target_summary_length # Example

    def run(self, documents_path: str, **kwargs) -> Document:
        # Placeholder for synchronous execution logic
        # 1. Load documents using self.reader
        # 2. Split documents into chunks using self.text_splitter
        # 3. Summarize chunks (iteratively if needed)
        # 4. Consolidate summaries
        # 5. Generate final summary
        # This will be built out in subsequent steps.
        raise NotImplementedError("Synchronous execution is not yet implemented.")

    async def arun(self, documents_path: str, **kwargs) -> Document:
        print(f"Starting document loading from: {documents_path}")
        # Assuming self.reader.load_data can handle a path string.
        # For DirectoryReader, load_data expects input_dir.
        # For AutoReader, load_data expects file_path.
        # This might require a check or a more unified interface in BaseReader.
        # For now, proceeding with documents_path directly.
        # If DirectoryReader is used, its load_data might need input_dir=documents_path
        # If AutoReader is used, its load_data might need file_path=documents_path
        # The task description uses self.reader.load_data(documents_path) in the example,
        # which implies the path argument name might be flexible or handled by AutoReader.
        # Let's stick to the example provided in the prompt.
        
        # Check if reader is DirectoryReader and adjust call if necessary
        if isinstance(self.reader, DirectoryReader):
            documents = self.reader.load_data(input_dir=documents_path)
        elif isinstance(self.reader, AutoReader):
            documents = self.reader.load_data(file=documents_path)
        else:
            # Assuming a generic BaseReader or other compatible reader
            documents = self.reader.load_data(file=documents_path)
        
        print(f"Loaded {len(documents)} document(s).")
        
        chunks = self.text_splitter.run(documents)
        print(f"Split documents into {len(chunks)} chunk(s).")
        
        # In future steps, these chunks will be summarized.
        # For now, just return counts.
        
        if not chunks:
            return {
                "message": "No chunks produced from documents.",
                "doc_count": len(documents),
                "chunk_count": 0,
                "chunk_summaries": []
            }

        print(f"Starting summarization of {len(chunks)} chunks...")
        list_of_initial_summaries = []
        processed_summaries_with_embeddings: List[Tuple[str, List[float]]] = [] 
        previous_summary_for_fallback = None

        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
            context_to_pass = None
            
            if self.use_semantic_context_for_chunks and self.embedding_model and i > 0 and processed_summaries_with_embeddings:
                print(f"Attempting semantic context for chunk {i+1}...")
                # 1. Get current chunk's embedding
                #Kotaemon's BaseEmbeddings ainvoke is expected to return List[DocumentWithEmbedding]
                current_chunk_embedding_docs: Optional[List[DocumentWithEmbedding]] = None
                try:
                    current_chunk_embedding_docs = await self.embedding_model.ainvoke([chunk.text])
                except Exception as e:
                    print(f"Warning: Embedding failed for current chunk {i+1} text: {e}")

                if not current_chunk_embedding_docs or not current_chunk_embedding_docs[0].embedding:
                    print(f"Warning: Could not get embedding for current chunk {i+1}. Falling back.")
                    if previous_summary_for_fallback:
                         context_to_pass = previous_summary_for_fallback
                else:
                    current_chunk_emb = current_chunk_embedding_docs[0].embedding
                    
                    # 2. Calculate similarities with past summaries' embeddings
                    similarities = []
                    for past_summary_text, past_summary_emb in processed_summaries_with_embeddings:
                        if past_summary_emb: # Ensure past summary embedding exists
                            sim = cosine_similarity(current_chunk_emb, past_summary_emb)
                            similarities.append((sim, past_summary_text))
                        else:
                            print(f"Warning: Missing embedding for a past summary. Skipping for similarity.")
                    
                    # 3. Sort by similarity (descending) and pick top_k
                    similarities.sort(key=lambda x: x[0], reverse=True)
                    top_k_summaries = [text for sim_score, text in similarities[:self.semantic_context_top_k]]
                    
                    if top_k_summaries:
                        print(f"Found {len(top_k_summaries)} relevant past summaries for chunk {i+1}.")
                        context_to_pass = "\n\n---\n\n".join(top_k_summaries)
                    elif previous_summary_for_fallback: # Fallback if no similar ones found but sequential exists
                        print(f"No semantically similar summaries found for chunk {i+1}. Using sequential fallback.")
                        context_to_pass = previous_summary_for_fallback
                    else:
                        print(f"No context found for chunk {i+1} (no semantic, no sequential).")

            elif i > 0 and previous_summary_for_fallback: # Fallback for non-semantic or if embedding model missing
                print(f"Using sequential context for chunk {i+1}.")
                context_to_pass = previous_summary_for_fallback
            else:
                print(f"No context for first chunk or no fallback available for chunk {i+1}.")


            summary_str = await self._asummarize_chunk(
                chunk,
                previous_chunk_summary=context_to_pass,
                target_tokens=self.target_chunk_summary_tokens
            )
            list_of_initial_summaries.append(summary_str)
            previous_summary_for_fallback = summary_str  # Update for next iteration's fallback

            if self.use_semantic_context_for_chunks and self.embedding_model:
                # Store summary and its embedding for future context selection
                new_summary_embedding_docs: Optional[List[DocumentWithEmbedding]] = None
                try:
                    new_summary_embedding_docs = await self.embedding_model.ainvoke([summary_str])
                except Exception as e:
                     print(f"Warning: Embedding failed for summary of chunk {i+1}: {e}")

                if new_summary_embedding_docs and new_summary_embedding_docs[0].embedding:
                    new_summary_emb = new_summary_embedding_docs[0].embedding
                    processed_summaries_with_embeddings.append((summary_str, new_summary_emb))
                    print(f"Stored summary and embedding for chunk {i+1}.")
                else:
                    print(f"Warning: Could not get embedding for summary of chunk {i+1}. It won't be used for future semantic context.")
        
        print(f"Finished summarizing {len(list_of_initial_summaries)} chunks.")

        # Hierarchical consolidation
        current_summaries = list_of_initial_summaries
        consolidation_passes = 0
        
        # Get consolidation params from self or kwargs (kwargs override for per-call flexibility)
        max_batch = kwargs.get(
            "max_summaries_per_consolidation_batch", 
            self.max_summaries_per_consolidation_batch
        )
        target_final_count = kwargs.get(
            "target_number_of_summaries_for_final_step",
            self.target_number_of_summaries_for_final_step
        )

        while len(current_summaries) > target_final_count:
            consolidation_passes += 1
            print(
                f"Consolidation pass {consolidation_passes}. "
                f"Number of summaries to process: {len(current_summaries)}"
            )

            batched_summaries_for_this_pass = []
            for i in range(0, len(current_summaries), max_batch):
                batch = current_summaries[i : i + max_batch]
                batched_summaries_for_this_pass.append(batch)

            if not batched_summaries_for_this_pass:
                # This case should ideally not be reached if len(current_summaries) > target_final_count
                print("Warning: No batches created for consolidation, breaking.")
                break

            consolidation_tasks = [
                self._acombine_summaries_batch(
                    batch,
                    target_tokens=self.target_consolidated_summary_tokens
                )
                for batch in batched_summaries_for_this_pass
            ]
            next_round_summaries = await asyncio.gather(*consolidation_tasks)
            
            print(
                f"Consolidation pass {consolidation_passes} finished. "
                f"Number of summaries now: {len(next_round_summaries)}"
            )

            # Break if consolidation isn't reducing summary count as expected
            if len(next_round_summaries) == len(batched_summaries_for_this_pass) and \
               len(next_round_summaries) > target_final_count:
                print(
                    f"Warning: Consolidation pass {consolidation_passes} did not reduce "
                    f"the number of summaries ({len(next_round_summaries)} summaries from "
                    f"{len(batched_summaries_for_this_pass)} batches). "
                    "This might be due to batch size or LLM behavior. "
                    "Breaking to avoid potential infinite loop."
                )
                current_summaries = next_round_summaries # Store results of this pass
                break 
            
            current_summaries = next_round_summaries

            if len(current_summaries) <= target_final_count:
                print(f"Reached target number of summaries ({len(current_summaries)}).")
                break
        
        final_consolidated_summaries = current_summaries
        doc_count_from_loading = len(documents)
        chunk_count_from_splitting = len(chunks)
        initial_summaries_count_from_chunking = len(list_of_initial_summaries)

        print(f"Generating final summary from {len(final_consolidated_summaries)} consolidated summary piece(s)...")
        final_summary_text = await self._agenerate_final_summary(final_consolidated_summaries)
        print("Final summary generated.")

        metadata = {
            "source": "SummarizationPipeline",
            "documents_processed_path": documents_path,
            "loaded_document_count": doc_count_from_loading,
            "total_chunks_generated": chunk_count_from_splitting,
            "initial_chunk_summaries_count": initial_summaries_count_from_chunking,
            "consolidation_passes": consolidation_passes,
            "final_consolidated_pieces_count": len(final_consolidated_summaries),
            "target_summary_length_tokens": self.target_summary_length_tokens,
        }
        
        return Document(content=final_summary_text, metadata=metadata)

    async def _asummarize_chunk(
        self,
        chunk: Document,
        previous_chunk_summary: Optional[str] = None,
        target_tokens: Optional[int] = None,
    ) -> str:
        """
        Summarize a single document chunk.

        Optionally uses context from the previous chunk's summary and can target
        a specific token length for the generated summary.

        Args:
            chunk: The document chunk to summarize.
            previous_chunk_summary: Summary of the preceding chunk, for
                contextual summarization.
            target_tokens: Target token length for this specific chunk's summary.

        Returns:
            The summary string for the chunk.
        """
        text_to_summarize = chunk.text
        if not text_to_summarize:  # Check if text is empty or None
            page_content = getattr(chunk, 'page_content', '') 
            if not page_content and hasattr(chunk, 'content'):
                 page_content = str(getattr(chunk, 'content', '')) 
            if not page_content:
                 print(f"Warning: Chunk {getattr(chunk, 'id_', 'N/A')} has no 'text', 'page_content', or 'content' attribute or it's empty. Returning empty summary.")
                 return ""
            text_to_summarize = page_content

        populate_params = {"text": text_to_summarize}

        if previous_chunk_summary:
            template = (
                "Given the previous context: {previous_summary}\n\n"
                "Summarize the following text factually, focusing on key information "
                "and its relation to the context. Ensure the summary is concise and "
                "informative"
            )
            populate_params["previous_summary"] = previous_chunk_summary
        else:
            template = "Summarize the following text factually, focusing on key information."

        if target_tokens:
            template += " The summary should be approximately {target_tokens} tokens long: {text}"
            populate_params["target_tokens"] = target_tokens
        else:
            template += ": {text}"
            
        prompt_template = PromptTemplate(template=template)
        prompt = prompt_template.populate(**populate_params)
        
        llm_response = await self._ainvoke_llm_summarize_chunk(prompt)
        
        summary = ""
        if hasattr(llm_response, 'content') and llm_response.content:
            summary = llm_response.content
        elif hasattr(llm_response, 'text') and llm_response.text: 
            summary = llm_response.text
        elif isinstance(llm_response, str): 
            summary = llm_response
        else:
            print(f"Warning: LLM response for chunk summary not as expected. Got: {type(llm_response)}. Returning empty string.")
        return summary

    async def _acombine_summaries_batch(
        self, summaries_batch: List[str], target_tokens: Optional[int] = None
    ) -> str:
        """
        Combine a batch of summaries into a single coherent summary.

        Args:
            summaries_batch: A list of summary strings to combine.
            target_tokens: Optional target token length for the consolidated summary.

        Returns:
            A single consolidated summary string.
        """
        if not summaries_batch:
            return ""
        
        text_of_combined_summaries = "\n\n---\n\n".join(summaries_batch)

        base_prompt = (
            "You are part of a hierarchical summarization process. "
            "Combine the following summaries, which cover different parts of a larger document. "
            "Create a consolidated summary that retains all key facts and distinct topics."
        )
        length_guidance = ""
        populate_params: Dict[str, Any] = {
            "text_of_combined_summaries": text_of_combined_summaries
        }

        if target_tokens:
            length_guidance = f" The consolidated summary should be approximately {target_tokens} tokens long."
            populate_params["target_tokens"] = target_tokens # Add if it's in the prompt
        
        prompt_template_str = f"{base_prompt}{length_guidance}\n\nInput summaries:\n{{text_of_combined_summaries}}"
        
        prompt_template = PromptTemplate(template=prompt_template_str)
        prompt = prompt_template.populate(**populate_params)
        
        llm_response = await self._ainvoke_llm_combine_summaries(prompt)
        
        consolidated_summary = ""
        if hasattr(llm_response, 'content') and llm_response.content:
            consolidated_summary = llm_response.content
        elif hasattr(llm_response, 'text') and llm_response.text:
            consolidated_summary = llm_response.text
        elif isinstance(llm_response, str):
            consolidated_summary = llm_response
        else:
            print(f"Warning: LLM response for consolidation not as expected. Got: {type(llm_response)}. Defaulting to joined summaries.")
            # Fallback to joined summaries if LLM fails to produce a valid response
            consolidated_summary = text_of_combined_summaries 
            
        return consolidated_summary

    # Helper methods for chunking, summarizing chunks, consolidating, etc.
    # will be added in later steps.

    async def _agenerate_final_summary(self, consolidated_summaries: List[str]) -> str:
        if not consolidated_summaries:
            return "Error: No content provided for final summarization."

        full_text_for_final_summary = "\n\n---\n\n".join(consolidated_summaries)

        prompt_template_str = (
            "Generate a comprehensive, factual summary of the following text. "
            "The summary should be approximately {target_tokens} tokens long. "
            "Analyze the content and determine the most appropriate structure for this summary "
            "(e.g., thematic, chronological, by key areas, etc.). Organize the summary according to this structure. "
            "Ensure all critical information and key points from the input are included factually and coherently. "
            "Avoid redundancy and maintain a clear, professional tone. "
            "Input text:\n{text_to_summarize}"
        )
        prompt_template = PromptTemplate(template=prompt_template_str)
        
        prompt = prompt_template.populate(
            text_to_summarize=full_text_for_final_summary,
            target_tokens=self.target_summary_length_tokens 
        )
        
        llm_response = await self._ainvoke_llm_generate_final(prompt)
        
        final_summary = ""
        if hasattr(llm_response, 'content') and llm_response.content:
            final_summary = llm_response.content
        elif hasattr(llm_response, 'text') and llm_response.text:
            final_summary = llm_response.text
        elif isinstance(llm_response, str):
            final_summary = llm_response
        else:
            print(f"Warning: LLM response structure not as expected for final summary. Got: {type(llm_response)}")
            final_summary = "Error in final summary generation." # Or provide a more descriptive error.
            
        return final_summary

    # Decorated helper methods for LLM calls with retry
    @DEFAULT_LLM_RETRY_DECORATOR
    async def _ainvoke_llm_summarize_chunk(self, prompt: Any) -> Any:
        return await self.llm.ainvoke(prompt)

    @DEFAULT_LLM_RETRY_DECORATOR
    async def _ainvoke_llm_combine_summaries(self, prompt: Any) -> Any:
        return await self.llm.ainvoke(prompt)

    @DEFAULT_LLM_RETRY_DECORATOR
    async def _ainvoke_llm_generate_final(self, prompt: Any) -> Any:
        return await self.llm.ainvoke(prompt)
