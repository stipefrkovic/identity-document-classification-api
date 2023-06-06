import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import document_processor.document_processor
import main
from document_processor.document_processor import PDFDocumentProcessor
from document_processor.logger import logger
from main import app

client = TestClient(app)
DOC_DIR = "/document/"


class TestMain:
    @pytest.fixture
    def models(self):
        return ["EFFICIENTDET", "EFFICIENTNET"]

    @pytest.fixture
    def mock_file(self, mocker):
        mock_file = mocker.MagicMock()
        return mock_file

    def test_read_root(self):
        response = client.get("/")
        assert response.json() == {"message": "Test"}

    def test_post_without_document(self):
        response = client.post(DOC_DIR)
        assert response.status_code == 422

    def test_global_app(self):
        assert isinstance(main.app, FastAPI)

    def test_model_env_var(self, models):
        assert main.model in models

    def test_global_document_processor(self):
        assert isinstance(main.document_processor, PDFDocumentProcessor)

    def test_post_process_document_returns_true_for_pdf(self, mock_file):
        mock_file.content_type = "application/pdf"
        assert main.check_document(mock_file)

    def test_post_process_document_returns_false_otherwise(self, mock_file):
        mock_file.content_type = "application/png"
        assert not main.check_document(mock_file)

    def test_post_process_document_calls_document_processor(self, mocker):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            mock_data = mocker.MagicMock()
            mocker.patch.object(mock_data, "get", return_value=None)
            mock_process_document = mocker.patch.object(main.document_processor,
                                                        "process_document",
                                                        return_value=mock_data
                                                        )
            client.post(DOC_DIR, files={"document": f})
            mock_process_document.assert_called_once()

    def test_post_pdf_process_document_accepted(self):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert response.status_code == 200

    def test_post_pdf_process_document_returns_correct_response_on_classified(self):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            json_response = response.json()
            assert json_response is not None

    def test_post_pdf_process_document_document_type_is_str(self):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert isinstance(response.json().get("document_type"), str)

    def test_post_pdf_process_document_returns_correct_document_type_on_classified(self):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert response.json().get("document_type") == "id_card"

    def test_post_pdf_process_document_prediction_confidences_is_dict(self):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            json_response = response.json()
            assert isinstance(json_response.get("meta"), dict)

    def test_post_pdf_process_document_returns_prediction_confidences_on_classified(self):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert response.json().get("meta")["prediction_confidences"] is not None

    def test_post_process_document_jpg_file_rejected(self):
        with open("./src/tests/files/id.jpg", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert response.status_code == 400

    def test_post_process_document_jpg_file_response_on_rejected(self):
        with open("./src/tests/files/id.jpg", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert response.json() == {"detail": "Invalid file type. File must be pdf."}
