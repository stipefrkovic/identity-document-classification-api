import io

import pdf2image as p2i
import pytest
from PIL import Image

from document_processor.pipeline.pdf_to_image_converter import (
    PdfToImageConverter,
    PdfToJpgConverter,
)


class Test_PdfToImageConverter:
    def test_PdfToImageConverter_not_implemented(self):
        with pytest.raises(TypeError):
            PdfToImageConverter().convert(pdf_bytes=b"")


class Test_PdfToJpgConverter:
    @pytest.fixture
    def files_path(self):
        return "./src/tests/files/"

    def test_convert_single_page_pdf(self, files_path):
        pdf_path = files_path + "single_page.pdf"

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        converter = PdfToJpgConverter()
        result = converter.convert(pdf_bytes)
        assert isinstance(result, io.BytesIO)

        img = Image.open(result)
        assert img.format == "JPEG"

    def test_convert_multi_page_pdf(self, files_path):
        pdf_path = files_path + "multi_page.pdf"
        num_pages = len(p2i.convert_from_path(pdf_path))

        converter = PdfToJpgConverter()
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            image_bytes = converter.convert(pdf_bytes)
            assert isinstance(image_bytes, io.BytesIO)

        # Check that all pages were converted
        page_sizes = []
        for i in range(num_pages):
            bytes_offset = i * Image.open(image_bytes).tell()
            with Image.open(io.BytesIO(image_bytes.getbuffer()[bytes_offset:])) as img:
                if img.size != 0:
                    page_sizes.append(img.size)
        assert len(page_sizes) == num_pages

    def test_convert_empty_pdf(self):
        empty_pdf_bytes = b""
        converter = PdfToJpgConverter()
        with pytest.raises(p2i.exceptions.PDFPageCountError):
            converter.convert(empty_pdf_bytes)
