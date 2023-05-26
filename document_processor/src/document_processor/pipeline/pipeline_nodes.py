from abc import ABC, abstractmethod

import numpy as np

# from tflite_support.task import vision
import tensorflow as tf
from PIL import Image

from .pdf_to_image_converter import PdfToImageConverter


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


# TODO add superclass for both models


class EffNetDocumentClassifier(DocumentProcessingNode):
    # TODO put somewhere else
    document_classes = ["driving_license", "id_card", "passport"]

    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        return tf.keras.models.load_model(model_path)

    def classify_image(self, image):
        # TODO add if
        image = image.resize((224, 224))
        # Convert the image into an array
        img_array = tf.keras.utils.img_to_array(image)
        # Convert the array into a batch
        img_batch = tf.expand_dims(img_array, 0)
        # Get model predictions
        predictions = self.model.predict(img_batch)
        # Get highest prediction
        prediction = np.argmax(predictions[0])
        # Get predicted clas
        predicted_class = self.document_classes[prediction]

        return predicted_class

    def process_document(self, data: dict):
        # TODO check that data dict has this element
        jpg_bytes = data["jpg_bytes"]
        pil_image = Image.open(jpg_bytes)
        classification_result = self.classify_image(pil_image)
        data["document_type"] = classification_result
        return data

class EffDetDocumentClassifier(DocumentProcessingNode):
    # TODO put somewhere else
    document_classes = ["driving_license", "id_card", "passport"]

    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        return tf.saved_model.load(model_path)

    def classify_image(self, image):
        # image = image.convert('RGB')
        (im_width, im_height) = image.size
        image_np = np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.model(input_tensor)
        highest_index = np.argmax(detections['detection_scores'][0])
        highest_class_index = detections['detection_classes'][0][highest_index].numpy().astype(np.int)
        highest_class = self.document_classes[highest_class_index - 1]
        # highest_confidence = detections['detection_scores'][0][highest_index].numpy()
        return highest_class

    def process_document(self, data: dict):
        jpg_bytes = data["jpg_bytes"]
        pil_image = Image.open(jpg_bytes)
        highest_class = self.classify_image(pil_image)
        # Do something with the highest class and confidence
        data["document_type"] = highest_class
        return data

class NNDocumentIdentifierNode(DocumentProcessingNode):
    # TODO put somewhere else
    document_classes = ["driving_license", "id_card", "passport"]

    def __init__(self, interpreter: tf.lite.Interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()

        # self.classifier = vision.ImageClassifier.create_from_file(model_path)

    def classify_image(self, image):

        image = image.resize((224, 224))

        # Convert the image to a numpy array
        input_data = np.asarray(image)
        input_data = np.array(input_data, dtype=np.uint8)

        # Add a batch dimension
        input_data = np.expand_dims(input_data, axis=0)

        # Set the input tensor to the input data
        self.interpreter.set_tensor(
            self.interpreter.get_input_details()[0]["index"], input_data
        )

        # Invoke the interpreter
        self.interpreter.invoke()

        # Get the output predicted class
        output_details = self.interpreter.get_output_details()[0]
        output_data = self.interpreter.get_tensor(output_details["index"])

        predicted_class = np.argmax(output_data[0])
        return self.document_classes[predicted_class]

    def process_document(self, data: dict):
        jpg_bytes = data["jpg_bytes"]

        pil_image = Image.open(jpg_bytes)
        classification_result = self.classify_image(pil_image)
        data["document_type"] = classification_result
        return data
