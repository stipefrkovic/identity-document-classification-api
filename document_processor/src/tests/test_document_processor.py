import pytest

from document_processor.document_processor import (
    DocumentProcessor,
    PDFDocumentProcessor,
)
from document_processor.pipeline.builder import DocumentProcessorPipelineBuilder
from document_processor.pipeline.pipeline import DocumentProcessorPipeline


class TestDocumentProcessor:
    def test_process_document_not_implemented(self, mocker):
        with pytest.raises(TypeError):
            pipeline_builder = mocker.Mock(spec=DocumentProcessorPipelineBuilder)
            DocumentProcessor().process_document(pipeline_builder)


class TestNeuralNetworkDocumentProcessor:

    @pytest.fixture
    def pdf_text(self):
        return "Hello, there!"

    @pytest.fixture
    def processor_and_pipeline(self, mocker, pdf_text):
        pipeline_builder = mocker.Mock(spec=DocumentProcessorPipelineBuilder)
        pipeline = mocker.Mock(spec=DocumentProcessorPipeline)
        pipeline_builder.build.return_value = pipeline

        processor = PDFDocumentProcessor(pipeline_builder)
        document = b"PDF document contents"
        pipeline.process_document.return_value = {
            "pdf_text": pdf_text,
            "processed": True,
        }

        return processor, pipeline, document

    def test_process_document_returns_expected_result(self, processor_and_pipeline, pdf_text):
        processor, _, document = processor_and_pipeline
        result = processor.process_document(document)
        assert result == {"pdf_text": pdf_text, "processed": True}

    def test_process_document_calls_pipeline_once(self, processor_and_pipeline):
        processor, pipeline, document = processor_and_pipeline
        processor.process_document(document)
        assert pipeline.process_document.call_count == 1

    def test_process_document_calls_pipeline_with_correct_args(self, processor_and_pipeline, mocker):
        processor, pipeline, document = processor_and_pipeline
        processor.process_document(document)
        assert pipeline.process_document.call_args == mocker.call(
            {"pdf_bytes": document}
        )
