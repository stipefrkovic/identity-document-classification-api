import pytest

from document_processor.pipeline.builder import (
    DocumentProcessorPipelineBuilder,
    EffNetDocumentProcessorPipelineBuilder,
    NeuralNetworkDocumentProcessorPipelineBuilder,
    EffDetDocumentProcessorPipelineBuilder
)
from document_processor.pipeline.pdf_to_image_converter import (
    PdfToJpgConverter
)
from document_processor.pipeline.pipeline_nodes import (
    NNDocumentIdentifierNode,
    PdfToImageConverterNode,
    EffNetDocumentClassifier,
    EffDetDocumentClassifier
)
from document_processor.pipeline.pipeline import (
    DocumentProcessorPipeline
)


class TestDocumentProcessorPipelineBuilder:
    def test_document_processor_pipeline_builder_not_implemented(self):
        with pytest.raises(TypeError):
            DocumentProcessorPipelineBuilder().build()


class TestDocumentProcessorPipelineBuilder:
    @pytest.fixture(params=[
        (EffNetDocumentProcessorPipelineBuilder, EffNetDocumentClassifier,
         "./src/document_processor/pipeline/models/effnet"),
        (EffDetDocumentProcessorPipelineBuilder, EffDetDocumentClassifier,
         "./src/document_processor/pipeline/models/effdet")
    ])
    def builder_info(self, request):
        return request.param

    @pytest.fixture
    def builder(self, builder_info):
        return builder_info[0]()

    @pytest.fixture
    def pipeline(self, builder):
        return builder.build()

    def test_initialization(self, builder):
        assert isinstance(builder, DocumentProcessorPipelineBuilder)

    def test_build_returns_pipeline(self, pipeline):
        assert isinstance(pipeline, DocumentProcessorPipeline)

    def test_build_contains_two_nodes(self, pipeline):
        nodes = pipeline.processing_nodes
        assert len(nodes) == 2

    def test_build_first_node_is_pdf_to_image(self, pipeline):
        nodes = pipeline.processing_nodes
        assert isinstance(nodes[0], PdfToImageConverterNode)

    def test_build_first_node_has_correct_converter(self, pipeline):
        nodes = pipeline.processing_nodes
        assert isinstance(nodes[0], PdfToImageConverterNode)

    def test_build_second_node_is_correct_type(self, builder_info, pipeline):
        nodes = pipeline.processing_nodes
        assert isinstance(nodes[1], builder_info[1])

    def test_build_second_node_has_correct_model_path(self, builder_info, pipeline):
        nodes = pipeline.processing_nodes
        assert isinstance(nodes[1], builder_info[1])


class TestNeuralNetworkDocumentProcessorPipelineBuilder:
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
