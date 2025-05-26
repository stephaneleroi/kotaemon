import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from llama_index.core.node_parser import SentenceSplitter as LISentenceSplitter
from llama_index.core.embeddings.base import BaseEmbedding as LI_BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser as LISemanticSplitterNodeParser

from kotaemon.indices.splitters.semantic_splitter import DynamicSemanticSplitter
from kotaemon.embeddings.base import BaseEmbeddings as Kotaemon_BaseEmbeddings
from kotaemon.embeddings.langchain_based import LCEmbeddingMixin # For testing _get_li_embedding_model
from kotaemon.base import Document # For creating Document objects


# Mock Kotaemon Embedding Model
class MockKotaemonEmbedding(Kotaemon_BaseEmbeddings):
    def __init__(self, model_name="mock_model_name_for_kotaemon_base"): # Updated model_name for clarity
        # BaseEmbeddings __init__ takes model_name and model_backend
        super().__init__(model_name=model_name, model_backend="mock_backend") 
        
        # This is the LlamaIndex compatible object we want _get_li_embedding_model to return
        # It needs to conform to the LI_BaseEmbedding interface if type checks are strict
        self.mock_li_embedding_obj = MagicMock(spec=LI_BaseEmbedding)
        
        # Mock the methods LlamaIndex SemanticSplitter might call on the LI_BaseEmbedding
        # These are usually called by the LI SemanticSplitterNodeParser internally.
        self.mock_li_embedding_obj.get_text_embedding.return_value = [0.1, 0.2, 0.3] 
        self.mock_li_embedding_obj.get_text_embedding_batch.return_value = [[0.1, 0.2, 0.3]]*2
        
        # For LCEmbeddingMixin compatibility test:
        # _get_li_embedding_model will try to access self._obj if the instance is LCEmbeddingMixin
        # So, if this mock is also used to test LCEmbeddingMixin path, it needs _obj.
        # Let's make this mock also an LCEmbeddingMixin for one of the test cases.
        # self._obj = self.mock_li_embedding_obj # This line would be for LCEmbeddingMixin simulation

    def invoke(self, text, *args, **kwargs): # Implement abstract method from BaseEmbeddings
        # This method is part of Kotaemon_BaseEmbeddings, not directly used by the splitter's LI part
        return self.mock_li_embedding_obj.get_text_embedding(text)

    async def ainvoke(self, text, *args, **kwargs): # Implement abstract method from BaseEmbeddings
        return self.mock_li_embedding_obj.get_text_embedding(text)

    # For _get_li_embedding_model Scenario 3: to_langchain_format
    def to_langchain_format(self):
        # Return an object that quacks like a Langchain embedding model
        # or is a LlamaIndex BaseEmbedding (which is what SemanticSplitterNodeParser wants)
        return self.mock_li_embedding_obj


# Test Cases
def test_dynamic_semantic_splitter_initialization():
    mock_kotaemon_embed = MockKotaemonEmbedding()
    
    # Simulate that _get_li_embedding_model returns the mock_li_embedding_obj
    with patch.object(DynamicSemanticSplitter, '_get_li_embedding_model', return_value=mock_kotaemon_embed.mock_li_embedding_obj) as mock_method:
        splitter = DynamicSemanticSplitter(
            embed_model=mock_kotaemon_embed,
            language_code="fr" 
        )
        mock_method.assert_called_once_with(mock_kotaemon_embed)

    assert splitter._li_params["embed_model"] == mock_kotaemon_embed.mock_li_embedding_obj
    assert isinstance(splitter._li_params["sentence_splitter"], LISentenceSplitter)
    assert splitter._li_params["sentence_splitter"].language == "fr"
    assert splitter._li_params["breakpoint_percentile_threshold"] == 95 # Default
    assert splitter._li_params["buffer_size"] == 1 # Default


def test_dynamic_semantic_splitter_initialization_custom_sentence_splitter():
    mock_kotaemon_embed = MockKotaemonEmbedding()
    custom_li_sentence_splitter = LISentenceSplitter(language="en", chunk_size=120)

    with patch.object(DynamicSemanticSplitter, '_get_li_embedding_model', return_value=mock_kotaemon_embed.mock_li_embedding_obj):
        splitter = DynamicSemanticSplitter(
            embed_model=mock_kotaemon_embed,
            sentence_splitter=custom_li_sentence_splitter
        )

    assert splitter._li_params["sentence_splitter"] == custom_li_sentence_splitter
    assert splitter._li_params["sentence_splitter"].language == "en" # Check custom lang
    assert splitter._li_params["sentence_splitter"].chunk_size == 120


@pytest.mark.asyncio # LlamaIndexDocTransformerMixin.get_nodes_from_documents is async
async def test_dynamic_semantic_splitter_calls_llama_index_splitter():
    mock_kotaemon_embed = MockKotaemonEmbedding()
    
    # The LlamaIndexDocTransformerMixin instantiates the LI object (_li_object)
    # using _get_li_class and _li_params. We want to mock the behavior of this _li_object.
    mock_li_splitter_instance = MagicMock(spec=LISemanticSplitterNodeParser)
    mock_li_splitter_instance.get_nodes_from_documents.return_value = ["node1", "node2"] # Mocked nodes

    docs = [Document(text="Ceci est un texte en français. Il a plusieurs phrases.")]

    # We need to patch the _get_li_object method of the LlamaIndexDocTransformerMixin
    # so it returns our mock_li_splitter_instance instead of creating a real one.
    with patch.object(DynamicSemanticSplitter, '_get_li_object', return_value=mock_li_splitter_instance) as mock_get_li_obj_method:
        # Ensure _get_li_embedding_model is also properly handled during init
        with patch.object(DynamicSemanticSplitter, '_get_li_embedding_model', return_value=mock_kotaemon_embed.mock_li_embedding_obj):
            splitter = DynamicSemanticSplitter(embed_model=mock_kotaemon_embed)
        
        # Call the method that uses the Llama Index object
        # LlamaIndexDocTransformerMixin uses get_nodes_from_documents, which calls _transform
        # which calls self._li_object.get_nodes_from_documents
        result_nodes = await splitter.get_nodes_from_documents(docs)

    mock_get_li_obj_method.assert_called_once() # Ensure our mock factory was used
    
    # Assert that the method on our mocked LlamaIndex object was called
    mock_li_splitter_instance.get_nodes_from_documents.assert_called_once_with(documents=docs)
    assert result_nodes == ["node1", "node2"]

    # To check the embed_model passed to the *actual* LI SemanticSplitterNodeParser,
    # we would need to *not* mock _get_li_object, and instead inspect splitter._li_object.
    # This test focuses on the delegation, so mocking _get_li_object is appropriate here.
    # A separate test could verify the properties of the created _li_object if needed,
    # but test_dynamic_semantic_splitter_initialization already covers _li_params.


def test_get_li_embedding_model_logic():
    # Dummy splitter instance to call the method from (doesn't need full init for this test)
    # We can mock the __init__ of the parent class or use a simplified instantiation
    # For simplicity, let's assume we can create an instance without full setup if only testing this one method.
    # Or, more cleanly, create a real instance but ensure its init doesn't fail.
    mock_kotaemon_for_init = MockKotaemonEmbedding() # For the main embed_model param
    with patch.object(DynamicSemanticSplitter, '_get_li_embedding_model', side_effect=lambda x: x): # Temp bypass for init
        splitter = DynamicSemanticSplitter(embed_model=mock_kotaemon_for_init) # Init needs an embed_model

    # Test Scenario 1: Kotaemon model is LCEmbeddingMixin
    class MockLCEmbedding(Kotaemon_BaseEmbeddings, LCEmbeddingMixin):
        def __init__(self, li_embedding_to_wrap):
            Kotaemon_BaseEmbeddings.__init__(self, model_name="lc_mixin_test", model_backend="mock")
            LCEmbeddingMixin.__init__(self) # LCEmbeddingMixin might have its own init needs
            self._obj = li_embedding_to_wrap # This is the Langchain object LlamaIndex can use

        def invoke(self, text, *args, **kwargs): return [] 
        async def ainvoke(self, text, *args, **kwargs): return []

    mock_li_embed_for_lc = MagicMock(spec=LI_BaseEmbedding) # This is what LlamaIndex expects
    mock_lc_kotaemon_embed = MockLCEmbedding(li_embedding_to_wrap=mock_li_embed_for_lc)
    
    # Temporarily unpatch _get_li_embedding_model for this specific test or call it directly
    # For this test, we are testing the actual _get_li_embedding_model method
    original_get_li_embedding_model = DynamicSemanticSplitter._get_li_embedding_model
    
    assert original_get_li_embedding_model(splitter, mock_lc_kotaemon_embed) == mock_li_embed_for_lc

    # Test Scenario 2: Raw LlamaIndex model passed
    mock_direct_li_embed = MagicMock(spec=LI_BaseEmbedding)
    assert original_get_li_embedding_model(splitter, mock_direct_li_embed) == mock_direct_li_embed

    # Test Scenario 3: Kotaemon model has to_langchain_format
    mock_kotaemon_with_converter = MockKotaemonEmbedding() # Uses the .to_langchain_format()
    # mock_kotaemon_with_converter.to_langchain_format = MagicMock(return_value=mock_kotaemon_with_converter.mock_li_embedding_obj)
    # Already implemented in MockKotaemonEmbedding
    assert original_get_li_embedding_model(splitter, mock_kotaemon_with_converter) == mock_kotaemon_with_converter.mock_li_embedding_obj
    
    # Test TypeError for incompatible model
    class IncompatibleEmbedding(Kotaemon_BaseEmbeddings):
        def __init__(self): 
            super().__init__(model_name="incompatible", model_backend="mock")
        def invoke(self, text, *args, **kwargs): return []
        async def ainvoke(self, text, *args, **kwargs): return []

    incompatible_embed = IncompatibleEmbedding()
    with pytest.raises(TypeError, match="Unsupported or incompatible embedding model type"):
        original_get_li_embedding_model(splitter, incompatible_embed)

```
