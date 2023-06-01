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
    def test_DocumentProcessorPipelineBuilder_not_implemented(self):
        with pytest.raises(TypeError):
            DocumentProcessorPipelineBuilder().build()


class DocumentProcessorPipelineBuilderTestBase:
    @pytest.fixture
    def builder(self):
        raise NotImplementedError("This method should be overridden")

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
        assert nodes[0].converter == PdfToJpgConverter


class EffNetDocumentProcessorPipelineBuilderTest(DocumentProcessorPipelineBuilderTestBase):

    @pytest.fixture
    def builder(self):
        return EffNetDocumentProcessorPipelineBuilder()

    def test_build_second_node_is_effnet(self, pipeline):
        nodes = pipeline.processing_nodes
        assert isinstance(nodes[1], EffNetDocumentClassifier)

    def test_build_second_node_has_correct_model_path(self, pipeline):
        nodes = pipeline.processing_nodes
        assert nodes[1].model_path == "./src/document_processor/pipeline/models/effnet"


class EffDetDocumentProcessorPipelineBuilderTest(DocumentProcessorPipelineBuilderTestBase):

    @pytest.fixture
    def builder(self):
        return EffDetDocumentProcessorPipelineBuilder()

    def test_build_second_node_is_effdet(self, pipeline):
        nodes = pipeline.processing_nodes
        assert isinstance(nodes[1], EffDetDocumentClassifier)

    def test_build_second_node_has_correct_model_path(self, pipeline):
        nodes = pipeline.processing_nodes
        assert nodes[1].model_path == "./src/document_processor/pipeline/models/effdet"


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
