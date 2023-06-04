import pytest

from document_processor.pipeline.pdf_to_image_converter import PdfToImageConverter
from document_processor.pipeline.pipeline_nodes import (
    DocumentProcessingNode,
    PdfToImageConverterNode,
    EffNetDocumentClassifierNode,
    EffDetDocumentClassifierNode,
    MLModelDocumentClassifierNode,
)


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

    def test_pdf_to_jpg_conversion_calls_convert(self, node, data, converter_mock):
        node.process_document(data)
        converter_mock.convert.assert_called_once_with(data["pdf_bytes"])

    def test_pdf_to_jpg_conversion_returns_correct_data(self, node, data, converter_mock):
        converter_mock.convert.return_value = b"test jpg bytes"
        result = node.process_document(data)
        assert result["jpg_bytes"] == converter_mock.convert.return_value


@pytest.fixture
def model_path():
    return "fake_path"


@pytest.fixture
def jpg_bytes():
    return b"test jpg bytes"


@pytest.fixture
def input_data(jpg_bytes):
    return {"jpg_bytes": jpg_bytes}


@pytest.fixture
def res_document_type():
    return "passport"


class DummyDocumentClassifierNode(MLModelDocumentClassifierNode):

    def load_model(self, model_path):
        pass

    def classify_image(self, image):
        return res_document_type()


class TestMLModelDocumentClassifierNode:

    @pytest.fixture
    def dummy_node(self, mocker, model_path, res_document_type):
        node = mocker.Mock(MLModelDocumentClassifierNode)
        node.classify_image.return_value = res_document_type
        node.process_document.side_effect = lambda x: {"document_type": node.classify_image(x)}
        return node

    def test_init_calls_load_model(self, mocker, model_path):
        mock_load_model = mocker.patch.object(DummyDocumentClassifierNode, 'load_model', autospec=True)
        DummyDocumentClassifierNode(model_path)
        assert mock_load_model.called

    def test_process_document_calls_classify_image(self, input_data, dummy_node):
        dummy_node.process_document(input_data)
        dummy_node.classify_image.assert_called_once()

    def test_process_document_returns_correct_classification_result(self, res_document_type, dummy_node, input_data):
        expected_result = {"document_type" : res_document_type}
        assert dummy_node.process_document(input_data) == expected_result

    def test_classify_image_not_implemented(self, model_path):
        with pytest.raises(TypeError):
            MLModelDocumentClassifierNode(model_path).classify_image(None)

    def test_load_model_not_implemented(self, model_path):
        with pytest.raises(TypeError):
            MLModelDocumentClassifierNode(model_path).load_model(None)


class TestConcreteDocumentClassifierNode:

    @pytest.fixture(params=[EffNetDocumentClassifierNode, EffDetDocumentClassifierNode])
    def mock_node(self, mocker, request, res_document_type):
        mock_node = mocker.Mock(spec=request.param)
        mock_node.classify_image.return_value = res_document_type
        mock_node.process_document.side_effect = lambda x: {"document_type": mock_node.classify_image(x)}
        return mock_node

    def test_process_document_calls_classify_image(self, mock_node, input_data):
        mock_node.process_document(input_data)
        mock_node.classify_image.assert_called_once()

    def test_process_document_returns_classification_result(self, mock_node, input_data):
        result = mock_node.process_document(input_data)
        assert "document_type" in result

    def test_process_document_returns_correct_classification_result(self, mock_node, input_data, res_document_type):
        result = mock_node.process_document(input_data)
        assert result["document_type"] == res_document_type


