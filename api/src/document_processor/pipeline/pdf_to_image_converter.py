import io
from abc import ABC, abstractmethod

from pdf2image import convert_from_bytes


class PdfToImageConverter(ABC):
    @staticmethod
    @abstractmethod
    def convert(pdf_bytes: bytes):
        pass


class PdfToJpgConverter(PdfToImageConverter):
    @staticmethod
    def convert(pdf_bytes: bytes):
        images = convert_from_bytes(pdf_bytes)
        image_bytes = io.BytesIO()

        for i, page in enumerate(images):
            output_image = io.BytesIO()
            page.save(output_image, format="JPEG")
            output_image.seek(0)
            image_bytes.write(output_image.read())

        image_bytes.seek(0)
        return image_bytes
