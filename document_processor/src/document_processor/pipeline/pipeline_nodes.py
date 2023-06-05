from abc import ABC, abstractmethod

import numpy as np

import tensorflow as tf
from PIL import Image

from .pdf_to_image_converter import PdfToImageConverter

from ..logger import logger


class DocumentProcessingNode(ABC):
    @abstractmethod
    def process_document(self, data: dict):
        pass


class PdfToImageConverterNode(DocumentProcessingNode):
    def __init__(self, converter: PdfToImageConverter):
        self.converter = converter

    def process_document(self, data: dict):
        data["jpg_bytes"] = self.converter.convert(data["pdf_bytes"])
        return data


class MLModelDocumentClassifierNode(DocumentProcessingNode):
    document_classes = ["driving_license", "id_card", "passport"]

    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def classify_image(self, image):
        pass

    def process_document(self, data: dict):
        jpg_bytes = data["jpg_bytes"]
        pil_image = Image.open(jpg_bytes)
        classification_result, prediction_confidences = self.classify_image(pil_image)
        
        data["document_type"] = classification_result
        data["prediction_confidences"] = prediction_confidences
        
        return data


class EffNetDocumentClassifierNode(MLModelDocumentClassifierNode):
    def load_model(self, model_path):
        return tf.keras.models.load_model(model_path)

    def classify_image(self, image) -> (str, list):
        image = image.resize((224, 224))
        # Convert the image into an array
        img_array = tf.keras.utils.img_to_array(image)
        # Convert the array into a batch
        img_batch = tf.expand_dims(img_array, 0)
        # Get model predictions
        predictions = self.model.predict(img_batch)

        prediction_confidences = []
        for i, prediction in enumerate(predictions[0]):
            prediction_confidences.append((self.document_classes[i], round(prediction.item(), 2)))

        # Get the highest prediction
        prediction = np.argmax(predictions[0])
        # Get predicted class
        predicted_class = self.document_classes[prediction]

        return predicted_class, prediction_confidences


class EffDetDocumentClassifierNode(MLModelDocumentClassifierNode):
    def load_model(self, model_path):
        return tf.saved_model.load(model_path)

    def classify_image(self, image):
        (im_width, im_height) = image.size
        image_np = np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.model(input_tensor)
        highest_index = np.argmax(detections['detection_scores'][0])
        highest_class_index = detections['detection_classes'][0][highest_index].numpy().astype(np.int)
        highest_class = self.document_classes[highest_class_index - 1]

        return highest_class, None

