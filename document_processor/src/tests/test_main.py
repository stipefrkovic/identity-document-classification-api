import os
import logging
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import main
from document_processor.document_processor import PDFDocumentProcessor
from document_processor.pipeline.builder import EffNetDocumentProcessorPipelineBuilder, \
    EffDetDocumentProcessorPipelineBuilder
from main import app


model = os.getenv("MODEL")
CLASSIFY_DOC_DIR = "/classify-document/"

main.document_processor = PDFDocumentProcessor(
            EffNetDocumentProcessorPipelineBuilder(),
            model_directory="/app/models/effnet",
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

    @pytest.fixture(scope="class")
    def effnet_pipeline_builder(self):
        pipeline_builder, model_path = main.get_pipeline_builder("EFFICIENTNET")
        return pipeline_builder, model_path

    @pytest.fixture(scope="class")
    def effdet_pipeline_builder(self):
        pipeline_builder, model_path = main.get_pipeline_builder("EFFICIENTDET")
        return pipeline_builder, model_path

    def test_get_env_vars_model(self, models):
        model, _, _ = main.get_env_vars()
        assert model in models

    def test_get_env_vars_min_confidence_numeric(self):
        _, min_conf, _ = main.get_env_vars()
        assert isinstance(min_conf, (int, float))

    def test_get_pipeline_builder_effnet_builder(self, effnet_pipeline_builder):
        pipeline_builder, _ = effnet_pipeline_builder
        assert isinstance(pipeline_builder, EffNetDocumentProcessorPipelineBuilder)

    def test_get_pipeline_builder_effnet_model_path(self, effnet_pipeline_builder):
        _, model_path = effnet_pipeline_builder
        assert model_path is not None and model_path != ""

    def test_get_pipeline_builder_effdet_builder(self, effdet_pipeline_builder):
        pipeline_builder, _ = effdet_pipeline_builder
        assert isinstance(pipeline_builder, EffDetDocumentProcessorPipelineBuilder)

    def test_get_pipeline_builder_effdet_model_path(self, effdet_pipeline_builder):
        _, model_path = effdet_pipeline_builder
        assert model_path is not None and model_path != ""

    def test_get_pipeline_builder_raises_error(self):
        with pytest.raises(ValueError):
            main.get_pipeline_builder("NOTAREALMODEL")

    def test_read_root(self, client):
        response = client.get("/")
        assert response.json() == {"message": "Running"}

    def test_post_without_document(self, client):
        response = client.post(CLASSIFY_DOC_DIR)
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
        with open("/app/document_processor/src/tests/files/id_card_1.pdf", "rb") as f:
            mock_data = mocker.MagicMock()
            mocker.patch.object(mock_data, "get", return_value=None)
            mock_process_document = mocker.patch.object(main.document_processor,
                                                        "process_document",
                                                        return_value=mock_data
                                                        )
            client.post(CLASSIFY_DOC_DIR, files={"document": f})
            mock_process_document.assert_called_once()

    def test_post_pdf_process_document_accepted(self, client):
        with open("/app/document_processor/src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(CLASSIFY_DOC_DIR, files={"document": f})
            assert response.status_code == 200

    def test_post_pdf_process_document_returns_correct_response_on_classified(self, client):
        with open("/app/document_processor/src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(CLASSIFY_DOC_DIR, files={"document": f})
            json_response = response.json()
            assert json_response is not None

    def test_post_pdf_process_document_document_type_is_str(self, client):
        with open("/app/document_processor/src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(CLASSIFY_DOC_DIR, files={"document": f})
            assert isinstance(response.json().get("document_type"), str)

    def test_post_pdf_process_document_returns_correct_document_type_on_classified(self, client):
        with open("/app/document_processor/src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(CLASSIFY_DOC_DIR, files={"document": f})
            assert response.json().get("document_type") == "id_card"

    def test_post_pdf_process_document_prediction_confidences_is_dict(self, client):
        with open("/app/document_processor/src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(CLASSIFY_DOC_DIR, files={"document": f})
            json_response = response.json()
            assert isinstance(json_response.get("meta"), dict)

    def test_post_pdf_process_document_returns_prediction_confidences_on_classified(self, client):
        with open("/app/document_processor/src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(CLASSIFY_DOC_DIR, files={"document": f})
            assert response.json().get("meta")["prediction_confidences"] is not None

    def test_post_process_document_jpg_file_rejected(self, client):
        with open("/app/document_processor/src/tests/files/id.jpg", "rb") as f:
            response = client.post(CLASSIFY_DOC_DIR, files={"document": f})
            assert response.status_code == 400

    def test_post_process_document_jpg_file_response_on_rejected(self, client):
        with open("/app/document_processor/src/tests/files/id.jpg", "rb") as f:
            response = client.post(CLASSIFY_DOC_DIR, files={"document": f})
            assert response.json() == {"detail": "Invalid file type. File must be pdf."}
