import numpy as np
import pytest
import io
from PIL import Image

from document_processor.pipeline.pdf_to_image_converter import PdfToImageConverter
from document_processor.pipeline.pipeline_nodes import (
    DocumentProcessingNode,
    PdfToImageConverterNode,
    EffNetDocumentClassifierNode,
    EffDetDocumentClassifierNode,
    MLModelDocumentClassifierNode,
)

from document_processor.logger import logger


class TestDocumentProcessingNode:
    def test_process_document_not_implemented(self):
        with pytest.raises(TypeError):
            DocumentProcessingNode().process_document(data={})


class TestPdfToImageConverterNode:
    @pytest.fixture
    def converter_mock(self, mocker):
        return mocker.Mock(spec=PdfToImageConverter)

    @pytest.fixture
    def node(self, converter_mock):
        return PdfToImageConverterNode(converter=converter_mock)

    @pytest.fixture
    def data(self):
        return {"pdf_bytes": b"test pdf bytes"}

    def test_process_document_calls_convert_with_pdf_bytes(self, node, data, converter_mock):
        node.process_document(data)
        converter_mock.convert.assert_called_once_with(data["pdf_bytes"])

    def test_process_document_result_contains_jpg_bytes(self, node, data):
        result = node.process_document(data)
        assert "jpg_bytes" in result

    def test_pdf_to_jpg_conversion_returns_data(self, node, data, converter_mock):
        converter_mock.convert.return_value = b"test jpg bytes"
        result = node.process_document(data)
        assert result["jpg_bytes"] == converter_mock.convert.return_value


@pytest.fixture
def model_path():
    return "fake_path"


@pytest.fixture
def jpg_bytes():
    image = Image.new('RGB', (60, 30))
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')
    return byte_arr.getvalue()


@pytest.fixture
def input_data(jpg_bytes):
    return {"jpg_bytes": jpg_bytes}


@pytest.fixture
def res_document_type():
    return "passport"


@pytest.fixture
def res_confidences():
    return np.array([[0.1, 0.1, 0.8]])


@pytest.fixture
def mock_image():
    image = Image.new('RGB', (60, 30))
    return image


class DummyDocumentClassifierNode(MLModelDocumentClassifierNode):
    def load_model(self, model_path):
        pass

    def classify_image(self, image):
        return "passport", np.array([[0.1, 0.1, 0.8]])


class TestMLModelDocumentClassifierNode:
    @pytest.fixture
    def dummy_node(self, model_path):
        return DummyDocumentClassifierNode(model_path)

    def test_init_calls_load_model(self, mocker, model_path):
        mock_load_model = mocker.patch.object(DummyDocumentClassifierNode, 'load_model', autospec=True)
        DummyDocumentClassifierNode(model_path)
        assert mock_load_model.called

    def test_process_document_calls_image_open(self, mocker, input_data, dummy_node, mock_image):
        mock_open = mocker.patch("PIL.Image.open", return_value=mock_image)
        dummy_node.process_document(input_data)
        mock_open.assert_called_once_with(input_data["jpg_bytes"])

    def test_process_document_calls_classify_image(self, mocker, input_data, dummy_node, mock_image,
                                                   res_document_type, res_confidences):
        mocker.patch("PIL.Image.open", return_value=mock_image)
        mock_classify_image = mocker.patch.object(dummy_node,
                                                  "classify_image",
                                                  return_value=(res_document_type, res_confidences)
                                                  )
        dummy_node.process_document(input_data)
        mock_classify_image.assert_called_once_with(mock_image)

    def test_process_document_result_contains_document_type(self, mocker, input_data, dummy_node):
        mocker.patch("PIL.Image.open")
        result_data = dummy_node.process_document(input_data)
        assert "document_type" in result_data

    def test_process_document_result_contains_prediction_confidences(self, mocker, input_data, dummy_node):
        mocker.patch("PIL.Image.open")
        result_data = dummy_node.process_document(input_data)
        assert "prediction_confidences" in result_data

    def test_classify_image_not_implemented(self, model_path):
        with pytest.raises(TypeError):
            MLModelDocumentClassifierNode(model_path).classify_image(None)

    def test_load_model_not_implemented(self, model_path):
        with pytest.raises(TypeError):
            MLModelDocumentClassifierNode(model_path).load_model(None)


class TestEffNetDocumentClassifierNode:
    @pytest.fixture
    def mock_model(self, mocker, res_confidences):
        mock_model = mocker.MagicMock()
        mock_model.predict.return_value = res_confidences
        return mock_model

    @pytest.fixture
    def effnet_node(self, mocker, mock_model, model_path):
        mocker.patch("tensorflow.keras.models.load_model", return_value=mock_model)
        node = EffNetDocumentClassifierNode(model_path)
        return node

    def test_classify_image_calls_image_resize(self, mocker, effnet_node, mock_image):
        mock_resize_image = mocker.patch.object(mock_image, "resize", return_value=mock_image)

        effnet_node.classify_image(mock_image)
        mock_resize_image.assert_called_once_with((224, 224))

    def test_classify_image_calls_model_predict(self, effnet_node, mock_image, mock_model):
        effnet_node.classify_image(mock_image)
        mock_model.predict.assert_called_once()

    def test_classify_image_returns_correct_tuple(self, effnet_node, mock_image,
                                                  res_confidences, res_document_type):
        result = effnet_node.classify_image(mock_image)
        assert isinstance(result, tuple)

    def test_classify_image_returns_correct_predicted_class(self, effnet_node, mock_image, res_document_type):
        result = effnet_node.classify_image(mock_image)
        assert result[0] == res_document_type

    def test_classify_image_returns_correct_predicted_confidences(self, effnet_node, mock_image, res_confidences):
        result = effnet_node.classify_image(mock_image)
        expected_confidences = list(zip(effnet_node.document_classes, res_confidences[0]))
        assert result[1] == expected_confidences


class TestEffDetDocumentClassifierNode:
    @pytest.fixture
    def mock_model(self, mocker, res_confidences):
        mock_model = mocker.MagicMock()
        mock_model.return_value = mocker.MagicMock()
        return mock_model

    @pytest.fixture
    def effdet_node(self, mocker, mock_model, model_path):
        mocker.patch("tensorflow.saved_model.load", return_value=mock_model)
        mocker.patch("numpy.array", return_value=mocker.MagicMock())
        node = EffDetDocumentClassifierNode(model_path)
        return node

    def test_classify_image_gets_image_data(self, mocker, effdet_node, mock_image):
        mocker.patch.object(mock_image, "getdata", return_value=mocker.MagicMock())
        effdet_node.classify_image(mock_image)
        mock_image.getdata.assert_called_once()

    def test_classify_image_calls_model(self, mocker, effdet_node, mock_image, mock_model):
        mocker.patch.object(mock_image, "getdata", return_value=mocker.MagicMock())
        effdet_node.classify_image(mock_image)
        mock_model.assert_called_once()

    def test_classify_image_returns_tuple(self, mocker, effdet_node, mock_image):
        mocker.patch.object(mock_image, "getdata", return_value=mocker.MagicMock())
        result = effdet_node.classify_image(mock_image)
        assert isinstance(result, tuple)



