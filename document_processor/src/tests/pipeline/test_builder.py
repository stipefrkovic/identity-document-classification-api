import pytest

from document_processor.pipeline.builder import (
    DocumentProcessorPipelineBuilder,
    NeuralNetworkDocumentProcessorPipelineBuilder,
)
from document_processor.pipeline.pipeline_nodes import (
    NNDocumentIdentifierNode,
    PdfToImageConverterNode,
)


class Test_DocumentProcessorPipelineBuilder:
    def test_DocumentProcessorPipelineBuilder_not_implemented(self):
        with pytest.raises(TypeError):
            DocumentProcessorPipelineBuilder().build()


# These tests might need refactoring if/when pipeline design gets changed.
class Test_NeuralNetworkDocumentProcessorPipelineBuilder:
    def test_build_returns_pipeline_with_two_nodes(self):
        builder = NeuralNetworkDocumentProcessorPipelineBuilder()
        pipeline = builder.build()
        assert len(pipeline.processing_nodes) >= 2

    def test_build_adds_pdf_to_image_conversion_node(self):
        builder = NeuralNetworkDocumentProcessorPipelineBuilder()
        pipeline = builder.build()
        assert isinstance(pipeline.processing_nodes[0], PdfToImageConverterNode)

    def test_build_adds_nn_document_identifier_node(self):
        builder = NeuralNetworkDocumentProcessorPipelineBuilder()
        pipeline = builder.build()
        assert isinstance(pipeline.processing_nodes[1], NNDocumentIdentifierNode)
