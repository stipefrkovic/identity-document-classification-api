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
            pipelineBuilder = mocker.Mock(spec=DocumentProcessorPipelineBuilder)
            DocumentProcessor().processDocument(pipelineBuilder)


class Test_NeuralNetworkDocumentProcessor:
    def test_process_document(self, mocker):
        pipelineBuilder = mocker.Mock(spec=DocumentProcessorPipelineBuilder)
        pipeline = mocker.Mock(spec=DocumentProcessorPipeline)
        pipelineBuilder.build.return_value = pipeline

        processor = NeuralNetworkDocumentProcessor(pipelineBuilder)
        document = b"PDF document contents"
        pipeline.processDocument.return_value = {
            "pdf_text": "Hello, there!",
            "processed": True,
        }

        result = processor.process_document(document)

        assert result == {"pdf_text": "Hello, there!", "processed": True}
        assert pipeline.processDocument.call_count == 1
        assert pipeline.processDocument.call_args == mocker.call(
            {"pdf_bytes": document}
        )
