import asyncio
from typing import Any, AsyncGenerator, Iterator, List, Optional, Union, Dict

from kotaemon.base import BaseComponent, Document, LLMInterface
from kotaemon.loaders.base import BaseReader
from kotaemon.loaders import DirectoryReader, AutoReader
from kotaemon.indices.splitters import BaseSplitter, TokenSplitter
from kotaemon.llms.base import BaseLLM
from kotaemon.llms.prompts import PromptTemplate
from kotaemon.embeddings.base import BaseEmbeddings # Added for future use

class SummarizationPipeline(BaseComponent):
    '''
    A pipeline for summarizing a collection of documents.
    '''
    
    reader: Union[BaseReader, DirectoryReader, AutoReader]
    text_splitter: TokenSplitter
    llm: BaseLLM
    # embedding_model: Optional[BaseEmbeddings] = None # For future use
    # target_summary_length: str = "5 pages" # Example, will need better handling
    max_summaries_per_consolidation_batch: int
    target_number_of_summaries_for_final_step: int
    target_summary_length_tokens: int

    def __init__(
        self,
        reader: Union[BaseReader, DirectoryReader, AutoReader],
        text_splitter: TokenSplitter,
        llm: BaseLLM,
        max_summaries_per_consolidation_batch: int = 5,
        target_number_of_summaries_for_final_step: int = 1,
        target_summary_length_tokens: int = 2000,
        # embedding_model: Optional[BaseEmbeddings] = None, # For future use
        # target_summary_length: str = "5 pages", # Example
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reader = reader
        self.text_splitter = text_splitter
        self.llm = llm
        self.max_summaries_per_consolidation_batch = (
            max_summaries_per_consolidation_batch
        )
        self.target_number_of_summaries_for_final_step = (
            target_number_of_summaries_for_final_step
        )
        self.target_summary_length_tokens = target_summary_length_tokens
        # self.embedding_model = embedding_model # For future use
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
            documents = self.reader.load_data(file_path=documents_path)
        else:
            # Assuming a generic BaseReader or other compatible reader
            documents = self.reader.load_data(documents_path)
        
        print(f"Loaded {len(documents)} document(s).")
        
        chunks = self.text_splitter.split_documents(documents)
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
        chunk_summary_tasks = [self._asummarize_chunk(chunk) for chunk in chunks]
        list_of_initial_summaries = await asyncio.gather(*chunk_summary_tasks)
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
                self._acombine_summaries_batch(batch)
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

    async def _asummarize_chunk(self, chunk: Document) -> str:
        """Summarize a single document chunk."""
        text_to_summarize = chunk.text
        if not text_to_summarize: # Check if text is empty or None
            page_content = getattr(chunk, 'page_content', '') 
            if not page_content and hasattr(chunk, 'content'):
                 page_content = str(getattr(chunk, 'content', '')) 
            if not page_content:
                 print(f"Warning: Chunk {getattr(chunk, 'id_', 'N/A')} has no 'text', 'page_content', or 'content' attribute or it's empty. Returning empty summary.")
                 return ""
            text_to_summarize = page_content
        
        prompt_template = PromptTemplate(
            template="Summarize the following text factually, focusing on key information: {text}"
        )
        prompt = str(prompt_template.format(text=text_to_summarize))
        
        llm_response = await self.llm.ainvoke(prompt)
        
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

    async def _acombine_summaries_batch(self, summaries_batch: List[str]) -> str:
        """Combine a batch of summaries into a single coherent summary."""
        if not summaries_batch:
            return ""
        
        text_of_combined_summaries = "\n\n---\n\n".join(summaries_batch)

        prompt_template = PromptTemplate(
            template="The following are several summaries. Combine them into a single, "
                     "coherent summary, retaining all key facts and information. "
                     "Ensure the output is well-structured and covers all distinct topics "
                     "mentioned in the input summaries. Input summaries:\n"
                     "{text_of_combined_summaries}"
        )
        prompt = str(prompt_template.format(text_of_combined_summaries=text_of_combined_summaries))
        
        llm_response = await self.llm.ainvoke(prompt)
        
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
        
        prompt = str(prompt_template.format(
            text_to_summarize=full_text_for_final_summary,
            target_tokens=self.target_summary_length_tokens 
        ))
        
        llm_response = await self.llm.ainvoke(prompt)
        
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
