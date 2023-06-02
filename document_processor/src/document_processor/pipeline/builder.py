from abc import ABC, abstractmethod

import tensorflow as tf

from .pdf_to_image_converter import PdfToJpgConverter
from .pipeline import DocumentProcessorPipeline
from .pipeline_nodes import PdfToImageConverterNode, EffNetDocumentClassifierNode, EffDetDocumentClassifierNode


class DocumentProcessorPipelineBuilder(ABC):
    @abstractmethod
    def build(self):
        pass


class EffNetDocumentProcessorPipelineBuilder(DocumentProcessorPipelineBuilder):
    def build(self):
        pipeline = DocumentProcessorPipeline()

        pdf_2_image_node = PdfToImageConverterNode(PdfToJpgConverter())
        pipeline.add_processing_node(pdf_2_image_node)

        eff_net_node = EffNetDocumentClassifierNode("./src/document_processor/pipeline/models/effnet")
        pipeline.add_processing_node(eff_net_node)
        
        return pipeline


class EffDetDocumentProcessorPipelineBuilder(DocumentProcessorPipelineBuilder):
    def build(self):
        pipeline = DocumentProcessorPipeline()

        pdf_2_image_node = PdfToImageConverterNode(PdfToJpgConverter())
        pipeline.add_processing_node(pdf_2_image_node)

        eff_net_node = EffDetDocumentClassifierNode("./src/document_processor/pipeline/models/effdet")
        pipeline.add_processing_node(eff_net_node)
        
        return pipeline
