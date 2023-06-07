import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import main
from document_processor.document_processor import PDFDocumentProcessor
from document_processor.pipeline.builder import EffNetDocumentProcessorPipelineBuilder
from main import app


model = os.getenv("MODEL")
DOC_DIR = "/document/"

main.document_processor = PDFDocumentProcessor(
            EffNetDocumentProcessorPipelineBuilder(),
            model_directory="./src/tests/test_models/effnet",
            min_confidence=0.5
        )


class TestMain:
    @pytest.fixture(scope="class")
    def client(self):
        client = TestClient(app)
        return client

    @pytest.fixture
    def models(self):
        return ["EFFICIENTDET", "EFFICIENTNET"]

    @pytest.fixture
    def mock_file(self, mocker):
        mock_file = mocker.MagicMock()
        return mock_file

    def test_read_root(self, client):
        response = client.get("/")
        assert response.json() == {"message": "Test"}

    def test_post_without_document(self, client):
        response = client.post(DOC_DIR)
        assert response.status_code == 422

    def test_global_app(self):
        assert isinstance(main.app, FastAPI)

    def test_model_env_var(self, models):
        assert model in models

    def test_post_process_document_returns_true_for_pdf(self, mock_file):
        mock_file.content_type = "application/pdf"
        assert main.check_document(mock_file)

    def test_post_process_document_returns_false_otherwise(self, mock_file):
        mock_file.content_type = "application/png"
        assert not main.check_document(mock_file)

    def test_post_process_document_calls_document_processor(self, mocker, client):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            mock_data = mocker.MagicMock()
            mocker.patch.object(mock_data, "get", return_value=None)
            mock_process_document = mocker.patch.object(main.document_processor,
                                                        "process_document",
                                                        return_value=mock_data
                                                        )
            client.post(DOC_DIR, files={"document": f})
            mock_process_document.assert_called_once()

    def test_post_pdf_process_document_accepted(self, client):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert response.status_code == 200

    def test_post_pdf_process_document_returns_correct_response_on_classified(self, client):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            json_response = response.json()
            assert json_response is not None

    def test_post_pdf_process_document_document_type_is_str(self, client):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert isinstance(response.json().get("document_type"), str)

    def test_post_pdf_process_document_returns_correct_document_type_on_classified(self, client):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert response.json().get("document_type") == "id_card"

    def test_post_pdf_process_document_prediction_confidences_is_dict(self, client):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            json_response = response.json()
            assert isinstance(json_response.get("meta"), dict)

    def test_post_pdf_process_document_returns_prediction_confidences_on_classified(self, client):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert response.json().get("meta")["prediction_confidences"] is not None

    def test_post_process_document_jpg_file_rejected(self, client):
        with open("./src/tests/files/id.jpg", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert response.status_code == 400

    def test_post_process_document_jpg_file_response_on_rejected(self, client):
        with open("./src/tests/files/id.jpg", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert response.json() == {"detail": "Invalid file type. File must be pdf."}
