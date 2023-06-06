from abc import ABC, abstractmethod

from document_processor.pipeline.builder import DocumentProcessorPipelineBuilder
from document_processor.pipeline.pipeline import DocumentProcessorPipeline


class DocumentProcessor(ABC):
    def __init__(self, pipelineBuilder: DocumentProcessorPipelineBuilder, min_confidence):
        self.document_processing_pipeline: DocumentProcessorPipeline = (
            pipelineBuilder.build(min_confidence)
        )

    @abstractmethod
    def process_document(self, document):
        pass


class PDFDocumentProcessor(DocumentProcessor):
    def __init__(self, pipelineBuilder: DocumentProcessorPipelineBuilder, min_confidence):
        super().__init__(pipelineBuilder, min_confidence)

    def process_document(self, document):
        data = {"pdf_bytes": document}
        data = self.document_processing_pipeline.process_document(data)
        return data
