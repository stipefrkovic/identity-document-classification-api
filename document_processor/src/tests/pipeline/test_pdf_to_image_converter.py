import io

import pdf2image as p2i
import pytest
from PIL import Image

from document_processor.pipeline.pdf_to_image_converter import (
    PdfToImageConverter,
    PdfToJpgConverter,
)


class TestPdfToImageConverter:
    def test_PdfToImageConverter_not_implemented(self):
        with pytest.raises(TypeError):
            PdfToImageConverter().convert(pdf_bytes=b"")


class TestPdfToJpgConverter:
    @pytest.fixture
    def files_path(self):
        return "./src/tests/files/"

    @pytest.fixture(params=[1, 2])
    def pdf_and_image_bytes(self, request):
        if request.param == 1:
            pdf_bytes = b"%PDF-1.7\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 595 842]>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000018 00000 n\n0000000063 00000 n\n0000000108 00000 n\ntrailer<</Size 4/Root 1 0 R>>startxref\n142\n%%EOF\n"
        else:
            pdf_bytes = b"%PDF-1.7\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R 4 0 R]/Count 2>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 595 842]>>endobj\n4 0 obj<</Type/Page/MediaBox[0 0 595 842]>>endobj\nxref\n0 5\n0000000000 65535 f\n0000000018 00000 n\n0000000062 00000 n\n0000000106 00000 n\n0000000150 00000 n\ntrailer<</Size 5/Root 1 0 R>>startxref\n146\n%%EOF\n"

        converter = PdfToJpgConverter()
        image_bytes = converter.convert(pdf_bytes)

        return pdf_bytes, image_bytes, request.param

    def test_convert_image_bytes_not_empty(self, pdf_and_image_bytes):
        _, image_bytes, _ = pdf_and_image_bytes
        assert len(image_bytes.getvalue()) > 0

    def test_convert_results_in_BytesIO(self, pdf_and_image_bytes):
        _, image_bytes, _ = pdf_and_image_bytes
        assert isinstance(image_bytes, io.BytesIO)

    def test_convert_multi_page(self, pdf_and_image_bytes):
        _, image_bytes, len_multipage = pdf_and_image_bytes

        page_sizes = []
        for i in range(len_multipage):
            bytes_offset = i * Image.open(image_bytes).tell()
            with Image.open(io.BytesIO(image_bytes.getbuffer()[bytes_offset:])) as img:
                if img.size != 0:
                    page_sizes.append(img.size)
        assert len(page_sizes) == len_multipage

    def test_convert_out_format_JPEG(self, pdf_and_image_bytes):
        _, image_bytes, _ = pdf_and_image_bytes
        img = Image.open(image_bytes)
        assert img.format == "JPEG"

    def test_convert_invalid_pdf_throws_exception(self):
        empty_pdf_bytes = b""
        with pytest.raises(p2i.exceptions.PDFPageCountError):
            PdfToJpgConverter().convert(empty_pdf_bytes)
