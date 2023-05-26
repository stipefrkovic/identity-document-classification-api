import pytest

from document_processor.document_processor import (
    DocumentProcessor,
    NeuralNetworkDocumentProcessor,
)
from document_processor.pipeline.builder import DocumentProcessorPipelineBuilder
from document_processor.pipeline.pipeline import DocumentProcessorPipeline


class Test_DocumentProcessor:
    def test_process_document_not_implemented(self, mocker):
        with pytest.raises(TypeError):
            pipeline_builder = mocker.Mock(spec=DocumentProcessorPipelineBuilder)
            DocumentProcessor().process_document(pipeline_builder)


class Test_NeuralNetworkDocumentProcessor:
    def test_process_document(self, mocker):
        pipeline_builder = mocker.Mock(spec=DocumentProcessorPipelineBuilder)
        pipeline = mocker.Mock(spec=DocumentProcessorPipeline)
        pipeline_builder.build.return_value = pipeline

        processor = NeuralNetworkDocumentProcessor(pipeline_builder)
        document = b"PDF document contents"
        pipeline.process_document.return_value = {
            "pdf_text": "Hello, there!",
            "processed": True,
        }

        result = processor.process_document(document)

        assert result == {"pdf_text": "Hello, there!", "processed": True}
        assert pipeline.process_document.call_count == 1
        assert pipeline.process_document.call_args == mocker.call(
            {"pdf_bytes": document}
        )
