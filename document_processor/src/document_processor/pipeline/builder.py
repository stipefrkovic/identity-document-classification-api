from abc import ABC, abstractmethod

import tensorflow as tf

from .pdf_to_image_converter import PdfToJpgConverter
from .pipeline import DocumentProcessorPipeline
from .pipeline_nodes import NNDocumentIdentifierNode, PdfToImageConverterNode


class DocumentProcessorPipelineBuilder(ABC):
    @abstractmethod
    def build(self):
        pass


class NeuralNetworkDocumentProcessorPipelineBuilder(DocumentProcessorPipelineBuilder):
    def __init__(self):
        super().__init__()

    def build(self):
        pipeline = DocumentProcessorPipeline()

        # Add PDF to JPG conversion node
        pdf2Image = PdfToImageConverterNode(PdfToJpgConverter())
        pipeline.addProcessingNode(pdf2Image)

        # TODO : Add node for neural network document classification
        pipeline.addProcessingNode(
            NNDocumentIdentifierNode(
                tf.lite.Interpreter("./src/document_processor/pipeline/model.tflite")
            )
        )
        return pipeline
