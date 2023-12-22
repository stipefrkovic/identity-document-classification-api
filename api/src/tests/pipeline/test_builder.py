import pytest

from document_processor.pipeline.builder import (
    DocumentProcessorPipelineBuilder,
    EffNetDocumentProcessorPipelineBuilder,
    EffDetDocumentProcessorPipelineBuilder
)
from document_processor.pipeline.pipeline_nodes import (
    PdfToImageConverterNode,
    EffNetDocumentClassifierNode,
    EffDetDocumentClassifierNode
)
from document_processor.pipeline.pipeline import (
    DocumentProcessorPipeline
)


class TestDocumentProcessorPipelineBuilderAbstract:
    def test_document_processor_pipeline_builder_not_implemented(self):
        with pytest.raises(TypeError):
            DocumentProcessorPipelineBuilder().build(0.5)


class TestEffNetDocumentProcessorPipelineBuilder:
    @pytest.fixture(scope="class")
    def builder(self):
        return EffNetDocumentProcessorPipelineBuilder()

    @pytest.fixture(scope="class")
    def pipeline(self, builder):
        return builder.build(min_confidence=0.5, model_directory="./models/effnet")

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

    def test_build_second_node_is_correct_type(self, pipeline):
        nodes = pipeline.processing_nodes
        assert isinstance(nodes[1], EffNetDocumentClassifierNode)

    def test_build_second_node_has_correct_model_path(self, pipeline):
        nodes = pipeline.processing_nodes
        assert isinstance(nodes[1], EffNetDocumentClassifierNode)


class TestEffDetDocumentProcessorPipelineBuilder:
    @pytest.fixture(scope="class")
    def builder(self):
        return EffDetDocumentProcessorPipelineBuilder()

    @pytest.fixture(scope="class")
    def pipeline(self, builder):
        return builder.build(min_confidence=0.5,
                             model_directory="./models/effdet/saved_model/saved_model")

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

    def test_build_second_node_is_correct_type(self, pipeline):
        nodes = pipeline.processing_nodes
        assert isinstance(nodes[1], EffDetDocumentClassifierNode)

    def test_build_second_node_has_correct_model_path(self, pipeline):
        nodes = pipeline.processing_nodes
        assert isinstance(nodes[1], EffDetDocumentClassifierNode)
