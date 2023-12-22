from abc import ABC, abstractmethod

from document_processor.pipeline.builder import DocumentProcessorPipelineBuilder
from document_processor.pipeline.pipeline import DocumentProcessorPipeline


class DocumentProcessor(ABC):
    """
    ABC for a Document Processor.
    """
    def __init__(self, pipeline_builder: DocumentProcessorPipelineBuilder, **kwargs):
        """
        Initializes DocumentProcessor.
        :param pipeline_builder: PipelineBuilder that will build the DocumentProcessorPipeline.
        :param kwargs: Args to be used by the pipeline_builder.
        """
        self.document_processing_pipeline: DocumentProcessorPipeline = (
            pipeline_builder.build(**kwargs)
        )

    @abstractmethod
    def process_document(self, document):
        """
        Processes a document with the pipeline.
        :param document: document to be processed.
        :return: dict containing data.
        """
        pass


class PDFDocumentProcessor(DocumentProcessor):
    """
    DocumentProcessor for a PDF document.
    """
    def __init__(self, pipeline_builder: DocumentProcessorPipelineBuilder, **kwargs):
        """
        Initializes PDFDocumentProcessor.
        :param pipeline_builder: PipelineBuilder that will build the DocumentProcessorPipeline.
        :param kwargs: Args to be used by the pipeline_builder.
        """
        super().__init__(pipeline_builder, **kwargs)

    def process_document(self, document):
        """
        Processes a PDF document with the pipeline.
        :param document: PDF document to be processed.
        :return: dict containing data.
        """
        data = {"pdf_bytes": document}
        data = self.document_processing_pipeline.process_document(data)
        return data
