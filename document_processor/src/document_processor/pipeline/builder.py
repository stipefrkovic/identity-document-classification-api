from abc import ABC, abstractmethod

import tensorflow as tf

from .pdf_to_image_converter import PdfToJpgConverter
from .pipeline import DocumentProcessorPipeline
from .pipeline_nodes import (
    PdfToImageConverterNode,
    EffNetDocumentClassifierNode,
    EffDetDocumentClassifierNode,
)

from ..logger import logger


class DocumentProcessorPipelineBuilder(ABC):
    @abstractmethod
    def build(self, min_confidence):
        pass


class EffNetDocumentProcessorPipelineBuilder(DocumentProcessorPipelineBuilder):
    def build(self, *args, **kwargs):
        pipeline = DocumentProcessorPipeline()

        pdf_2_image_node = PdfToImageConverterNode(PdfToJpgConverter())
        pipeline.add_processing_node(pdf_2_image_node)

        if (min_confidence := kwargs.get("min_confidence")) is None:
            raise ValueError("min_confidence must be set for EfficientNet model")

        if (model_directory := kwargs.get("model_directory")) is None:
            raise ValueError("model_directory must be set for EfficientNet model")

        eff_net_node = EffNetDocumentClassifierNode(
            model_directory,
            min_confidence,
        )
        pipeline.add_processing_node(eff_net_node)

        return pipeline


class EffDetDocumentProcessorPipelineBuilder(DocumentProcessorPipelineBuilder):
    def build(self, *args, **kwargs):
        pipeline = DocumentProcessorPipeline()

        pdf_2_image_node = PdfToImageConverterNode(PdfToJpgConverter())
        pipeline.add_processing_node(pdf_2_image_node)

        if (model_directory := kwargs.get("model_directory")) is None:
            raise ValueError("model_directory must be set for EfficientDet model")

        if (min_confidence := kwargs.get("min_confidence")) is None:
            raise ValueError("min_confidence must be set for EfficientDet model")

        eff_det_node = EffDetDocumentClassifierNode(
            # "./src/document_processor/pipeline/models/effdet/saved_model/saved_model",
            model_directory,
            min_confidence,
        )
        pipeline.add_processing_node(eff_det_node)

        return pipeline
