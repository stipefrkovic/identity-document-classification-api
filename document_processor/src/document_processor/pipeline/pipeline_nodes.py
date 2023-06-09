from abc import ABC, abstractmethod

import numpy as np

import tensorflow as tf
from PIL import Image

from .pdf_to_image_converter import PdfToImageConverter

from ..logger import logger


class DocumentProcessingNode(ABC):
    """
    ABC for a DocumentProcessingNode.
    """
    @abstractmethod
    def process_document(self, data: dict):
        """
        Process a document.
        :param data: Dictionary containing the document and other data.
        """
        pass


class PdfToImageConverterNode(DocumentProcessingNode):
    """
    DocumentProcessingNode that converts a PDF into an image.
    """
    def __init__(self, converter: PdfToImageConverter):
        """
        Initializes the PdfToImageConverterNode.
        :param converter: PdfToImageConverter that will convert the PDF into an image.
        """
        self.converter = converter

    def process_document(self, data: dict):
        """
        Converts a PDF into an image.
        :param data: Dictionary containing the PDF.
        :return: Dictionary containing the image.
        """
        data["jpg_bytes"] = self.converter.convert(data["pdf_bytes"])
        return data


class MLModelDocumentClassifierNode(DocumentProcessingNode):
    """
    DocumentProcessingNode that will classify an image with a Machine learning model.
    """

    document_classes = ["driving_license", "id_card", "passport"]

    def __init__(self, model_path, min_confidence):
        """
        Initializes a MLModelDocumentClassifierNode.
        :param model_path: Path to the Machine Learning model.
        :param min_confidence: Minimum required confidence of the classification otherwise classification is unknown.
        """
        self.model = self.load_model(model_path)
        self.min_confidence = min_confidence

    @abstractmethod
    def load_model(self, model_path):
        """
        Loads the Machine Learning model from the given path.
        :param model_path: Path to the saved model.
        :return: Machine Learning model.
        """
        pass

    @abstractmethod
    def classify_image(self, image):
        """
        Classifies an image with the Machine Learning model.
        :param image: Image to be classified.
        :return: Class.
        """
        pass

    def process_document(self, data: dict):
        """
        Classifies an image of a document.
        :param data: Dictionary containing the image of the document.
        :return: Dictionary containing document class and prediction confidences.
        """
        jpg_bytes = data["jpg_bytes"]
        pil_image = Image.open(jpg_bytes)
        classification_result, prediction_confidences = self.classify_image(pil_image)
        
        data["document_type"] = classification_result
        data["prediction_confidences"] = prediction_confidences
        
        return data


class EffNetDocumentClassifierNode(MLModelDocumentClassifierNode):
    """
    MLModelDocumentClassifierNode that uses an EffNet model.
    """
    def load_model(self, model_path):
        """
        Loads the EffNet model.
        :param model_path: Path to the EffNet model.
        :return: EffNet model.
        """
        return tf.keras.models.load_model(model_path)

    def classify_image(self, image) -> (str, list):
        """
        Classifies an image with the EffNet model.
        :param image: Image to be classified.
        :return: Class.
        """
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
        if predictions[0][prediction] < self.min_confidence:
            return None, None

        # Get predicted class
        predicted_class = self.document_classes[prediction]

        return predicted_class, prediction_confidences


class EffDetDocumentClassifierNode(MLModelDocumentClassifierNode):
    """
    MLModelDocumentClassifierNode that uses an EffDet model.
    """
    def load_model(self, model_path):
        """
        Loads the EffDet model.
        :param model_path: Path to the EffDet model.
        :return: EffDet model.
        """
        return tf.saved_model.load(model_path)

    def get_detections(self, image):
        """
        Gets all the detections of the EffDet model.
        :param image: Image to be detected.
        :return: Detections.
        """
        (im_width, im_height) = image.size
        image_np = np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.model(input_tensor)
        return detections

    def calculate_highest_index(self, detections):
        """
        Gets the index of the most confident detection.
        :param detections: Detections.
        :return: Index of the most confident detection.
        """
        return np.argmax(detections['detection_scores'][0])

    def calculate_highest_class_index(self, detections, highest_index):
        """
        Gets the class of the most confident detection.
        :param detections: Detections.
        :param highest_index: Index of the most confident detection.
        :return: Class of the most confident detection.
        """
        return detections['detection_classes'][0][highest_index].numpy().astype(np.int)

    def get_prediction_confidences(self, detections):
        """
        Gets the confidences of the detections.
        :param detections: Detections.
        :return: Confidences of the detections.
        """
        prediction_confidences = []
        for i, prediction in enumerate(detections['detection_scores'][0]):
            document_class = self.document_classes[detections['detection_classes'][0][i].numpy().astype(np.int) - 1]
            confidence = float(prediction.numpy())
            if confidence >= self.min_confidence:
                prediction_confidences.append((document_class, round(confidence, 2)))

        return prediction_confidences

    def classify_image(self, image) -> (str, list):
        """
        Classifies an image with the EffDet model.
        :param image: Image to be classified.
        :return: Class.
        """
        detections = self.get_detections(image)
        if len(detections['detection_scores'][0]) == 0:
            return None, None

        highest_index = self.calculate_highest_index(detections)
        highest_class_index = self.calculate_highest_class_index(detections, highest_index)
        highest_class = self.document_classes[highest_class_index - 1]

        if detections['detection_scores'][0][highest_index].numpy() < self.min_confidence:
            return None, None

        prediction_confidences = self.get_prediction_confidences(detections)

        return highest_class, prediction_confidences

