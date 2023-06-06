import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import main
from document_processor.document_processor import PDFDocumentProcessor
from main import app

client = TestClient(app)
DOC_DIR = "/document/"


class TestMain:
    @pytest.fixture
    def models(self):
        return ["EFFICIENTDET", "EFFICIENTNET"]

    def test_main(self):
        response = client.get("/")
        assert response.json() == {"message": "Test"}

    def test_post_pdf_document(self):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert response.status_code == 200
            assert response.json().get("document_type") == "id_card"

    def test_post_jpg_file(self):
        # Tests that jpg files are rejected
        with open("./src/tests/files/id.jpg", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert response.status_code == 400
            assert response.json() == {"detail": "Invalid file type. File must be pdf."}

    def test_post_without_document(self):
        response = client.post(DOC_DIR)
        assert response.status_code == 422

    def test_global_app(self):
        assert isinstance(main.app, FastAPI)

    def test_model_env_var(self, models):
        assert main.model in models

    def test_global_document_processor(self):
        assert isinstance(main.document_processor, PDFDocumentProcessor)