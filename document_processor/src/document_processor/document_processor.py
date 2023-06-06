from abc import ABC, abstractmethod

from document_processor.pipeline.builder import DocumentProcessorPipelineBuilder
from document_processor.pipeline.pipeline import DocumentProcessorPipeline


class DocumentProcessor(ABC):
    def __init__(self, pipeline_builder: DocumentProcessorPipelineBuilder):
        self.document_processing_pipeline: DocumentProcessorPipeline = (
            pipeline_builder.build()
        )

    @abstractmethod
    def process_document(self, document):
        pass


class PDFDocumentProcessor(DocumentProcessor):
    def __init__(self, pipeline_builder: DocumentProcessorPipelineBuilder):
        super().__init__(pipeline_builder)

    def process_document(self, document):
        data = {"pdf_bytes": document}
        data = self.document_processing_pipeline.process_document(data)
        return data
