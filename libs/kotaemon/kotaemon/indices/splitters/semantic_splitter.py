from typing import Optional, Any, Dict, Type

from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.embeddings.base import BaseEmbedding as LI_BaseEmbedding

from kotaemon.embeddings.base import BaseEmbeddings as Kotaemon_BaseEmbeddings
# LCEmbeddingMixin might not be directly imported if only checked via hasattr
# from kotaemon.embeddings.langchain_based import LCEmbeddingMixin
from kotaemon.indices.splitters.base import BaseSplitter
from kotaemon.base import LlamaIndexDocTransformerMixin


class DynamicSemanticSplitter(LlamaIndexDocTransformerMixin, BaseSplitter):
    '''
    A text splitter that uses semantic similarity to determine breakpoints between chunks.
    Wraps LlamaIndex's SemanticSplitterNodeParser.
    '''
    def __init__(
        self,
        embed_model: Kotaemon_BaseEmbeddings,
        breakpoint_percentile_threshold: int = 95,
        buffer_size: int = 1,
        language_code: str = "fr", # Default to French as per user's context
        # Optional: pass a specific LlamaIndex SentenceSplitter instance
        sentence_splitter: Optional[SentenceSplitter] = None,
        **kwargs: Any, # For BaseSplitter or LlamaIndexDocTransformerMixin
    ):
        '''
        Args:
            embed_model: The Kotaemon embedding model to use.
            breakpoint_percentile_threshold: The percentile of similarity scores where
                the split occurs. Higher means larger, more inclusive chunks.
            buffer_size: Number of sentences to group together before comparing similarity.
            language_code: Language code for sentence splitting (e.g., "fr", "en").
            sentence_splitter: Optional pre-configured LlamaIndex SentenceSplitter.
                               If None, one will be created with the specified language_code.
            **kwargs: Additional parameters for BaseSplitter or LlamaIndexDocTransformerMixin.
        '''
        self._li_embed_model = self._get_li_embedding_model(embed_model)

        if sentence_splitter is None:
            # Default SentenceSplitter if not provided
            sentence_splitter = SentenceSplitter(language=language_code)

        # Parameters for LlamaIndexDocTransformerMixin to pass to SemanticSplitterNodeParser
        self._li_params: Dict[str, Any] = {
            "embed_model": self._li_embed_model,
            "breakpoint_percentile_threshold": breakpoint_percentile_threshold,
            "buffer_size": buffer_size,
            "sentence_splitter": sentence_splitter,
            # **kwargs is passed to super().__init__ and mixin might pick relevant ones
        }
        
        # Initialize BaseSplitter / LlamaIndexDocTransformerMixin / BaseComponent
        # The LlamaIndexDocTransformerMixin is expected to use _li_params and _get_li_class
        # to instantiate the Llama Index object.
        # Pass remaining kwargs to the superclass constructor.
        # BaseComponent, which is a grandparent, accepts **kwargs.
        super().__init__(**kwargs)

    def _get_li_class(self) -> Type[SemanticSplitterNodeParser]:
        return SemanticSplitterNodeParser

    def _get_li_embedding_model(self, kotaemon_embed_model: Kotaemon_BaseEmbeddings) -> LI_BaseEmbedding:
        '''
        Converts or extracts a LlamaIndex-compatible embedding model
        from the provided Kotaemon embedding model.
        '''
        # Scenario 1: Kotaemon model wraps a Langchain model which is LI-compatible (via LCEmbeddingMixin)
        # LCEmbeddingMixin often stores the Langchain object in `_obj`
        if hasattr(kotaemon_embed_model, '_obj') and isinstance(getattr(kotaemon_embed_model, '_obj'), LI_BaseEmbedding):
            return getattr(kotaemon_embed_model, '_obj')
        
        # Scenario 2: A raw LlamaIndex model was passed
        if isinstance(kotaemon_embed_model, LI_BaseEmbedding):
            return kotaemon_embed_model
        
        # Scenario 3: Kotaemon model has a specific converter to Langchain/LlamaIndex
        # (e.g., `to_langchain_format()` which returns an LI-compatible object)
        if hasattr(kotaemon_embed_model, 'to_langchain_format'):
            lc_model = kotaemon_embed_model.to_langchain_format()
            if isinstance(lc_model, LI_BaseEmbedding): # Check if the converted model is LI compatible
                return lc_model
            # Langchain's BaseEmbedding is not directly LI_BaseEmbedding.
            # However, LlamaIndex often directly accepts Langchain embedding models.
            # This check might be too strict. LlamaIndex has its own wrapper for Langchain embeddings.
            # `llama_index.core.embeddings.LangchainEmbedding`
            # For now, let's assume direct compatibility if it quacks like a Langchain model
            # that LlamaIndex can typically handle.
            # A common pattern is that LlamaIndex can use Langchain embedding models if they have `embed_query`.
            if hasattr(lc_model, 'embed_query') and hasattr(lc_model, 'embed_documents'):
                 # This is a common duck-typing check for Langchain embeddings.
                 # LlamaIndex can usually wrap these.
                 return lc_model # LlamaIndex will likely wrap this in LangchainEmbedding

        # Scenario 4: Kotaemon's BaseEmbeddings itself is directly LlamaIndex compatible
        # This is less likely given the distinct types, but good to cover if the API aligns.
        # This is implicitly covered if it passes the isinstance(kotaemon_embed_model, LI_BaseEmbedding) check.

        raise TypeError(
            f"Unsupported or incompatible embedding model type: {type(kotaemon_embed_model)}. "
            "DynamicSemanticSplitter requires a LlamaIndex-compatible embedding model "
            "or a Kotaemon model that can be converted to one (e.g., via LCEmbeddingMixin "
            "or a `to_langchain_format()` method returning a compatible Langchain model)."
        )

    # LlamaIndexDocTransformerMixin expects _li_params and _get_li_class to be defined.
    # It uses these in its `_get_li_object` method to instantiate the Llama Index node parser.
    # The `run` and other methods are inherited from BaseSplitter and LlamaIndexDocTransformerMixin.
    # BaseSplitter -> DocTransformer -> LlamaIndexDocTransformerMixin -> BaseComponent
    # DocTransformer provides the `__call__` and `transform_documents`
    # BaseSplitter provides `split_documents` (which calls transform_documents)
    # and `run` (which calls split_documents).
    # LlamaIndexDocTransformerMixin overrides `_transform` to use the Llama Index object.

```
