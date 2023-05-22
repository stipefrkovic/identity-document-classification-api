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
        self.model_path = model_path
        self.model = self.load_model(self.model_path)

    def load_model(self):
        return tf.keras.models.load_model(self.model_path)

    def classify_image(self, image):
        # TODO add if
        image = image.resize((224, 224))

        # Convert the image into an array
        img_array = tf.keras.utils.img_to_array(image)
        # Convert the array into a barch
        img_batch = tf.expand_dims(img_array, 0)

        prediction = self.model_path.predict(img_batch)

        print(f"{prediction}")

        return "Hello there."

    def process_document(self, data: dict):
        # TODO check that data dict has this element
        jpg_bytes = data["jpg_bytes"]
        pil_image = Image.open(jpg_bytes)
        classification_result = self.classify_image(pil_image)
        data["document_type"] = classification_result
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
