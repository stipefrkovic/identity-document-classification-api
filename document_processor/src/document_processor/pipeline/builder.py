from abc import ABC, abstractmethod

import tensorflow as tf

from .pdf_to_image_converter import PdfToJpgConverter
from .pipeline import DocumentProcessorPipeline
from .pipeline_nodes import NNDocumentIdentifierNode, PdfToImageConverterNode, EffNetDocumentClassifier


class DocumentProcessorPipelineBuilder(ABC):
    @abstractmethod
    def build(self):
        pass


class EffNetDocumentProcessorPipelineBuilder(DocumentProcessorPipelineBuilder):
    def __init__(self):
        super().__init__()

    def build(self):
        pipeline = DocumentProcessorPipeline()

        pdf_2_image_node = PdfToImageConverterNode(PdfToJpgConverter())
        pipeline.add_processing_node(pdf_2_image_node)

        eff_net_node = EffNetDocumentClassifier("./src/document_processor/pipeline/models/TODO")
        pipeline.add_processing_node(eff_net_node)
        
        return pipeline


class NeuralNetworkDocumentProcessorPipelineBuilder(DocumentProcessorPipelineBuilder):
    def __init__(self):
        super().__init__()

    def build(self):
        pipeline = DocumentProcessorPipeline()

        pdf_2_image_node = PdfToImageConverterNode(PdfToJpgConverter())
        pipeline.add_processing_node(pdf_2_image_node)

        pipeline.add_processing_node(
            NNDocumentIdentifierNode(
                tf.lite.Interpreter("./src/document_processor/pipeline/model.tflite")
            )
        )
        
        return pipeline
