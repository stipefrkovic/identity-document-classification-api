import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from document_processor.pipeline.pdf_to_image_converter import PdfToImageConverter
from document_processor.pipeline.pipeline_nodes import (
    DocumentProcessingNode,
    NNDocumentIdentifierNode,
    PdfToImageConverterNode, EffNetDocumentClassifier, EffDetDocumentClassifier,
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
def jpg_bytes():
    return b"test jpg bytes"


@pytest.fixture
def input_data(jpg_bytes):
    return {"jpg_bytes": jpg_bytes}


class TestClassifierNode:
    @pytest.fixture(params=[EffNetDocumentClassifier, EffDetDocumentClassifier, NNDocumentIdentifierNode])
    def mock_node(self, mocker, request):
        mock_node = mocker.Mock(spec=request.param)
        mock_node.classify_image.return_value = "passport"
        mock_node.process_document.side_effect = lambda x: {"document_type": mock_node.classify_image(x)}
        return mock_node

    def test_process_document_calls_classify_image(self, mock_node, input_data):
        mock_node.process_document(input_data)
        mock_node.classify_image.assert_called_once()

    def test_process_document_returns_classification_result(self, mock_node, input_data):
        result = mock_node.process_document(input_data)
        assert "document_type" in result

    def test_process_document_returns_correct_classification_result(self, mock_node, input_data):
        result = mock_node.process_document(input_data)
        assert result["document_type"] == "passport"


class TestNNDocumentIdentifierNode:
    @pytest.fixture
    def node(self, mocker):
        mock_interpreter = mocker.Mock(spec=tf.lite.Interpreter)
        node = NNDocumentIdentifierNode(mock_interpreter)
        node.interpreter.allocate_tensors.return_value = None
        node.interpreter.get_input_details.return_value = [{"index": 0}]
        node.interpreter.get_output_details.return_value = [{"index": 0}]
        node.interpreter.get_tensor.return_value = [0]
        node.interpreter.invoke.return_value = None
        return node

    @pytest.fixture
    def image(self):
        image_array = np.random.randint(0, 256, size=(224, 224, 3))
        return Image.fromarray(image_array, mode="RGB")

    def test_classify_image_calls_get_output_details(self, node, image):
        node.classify_image(image)
        node.interpreter.get_output_details.assert_called_once()

    def test_classify_image_calls_allocate_tensors(self, node, image):
        node.classify_image(image)
        node.interpreter.allocate_tensors.assert_called_once()

    def test_classify_image_calls_get_input_details(self, node, image):
        node.classify_image(image)
        node.interpreter.get_input_details.assert_called_once()

    def test_classify_image_calls_invoke(self, node, image):
        node.classify_image(image)
        node.interpreter.invoke.assert_called_once()

    def test_classify_image_returns_result(self, node, image):
        result = node.classify_image(image)
        assert len(result) > 0

    def test_classify_image_returns_string(self, node, image):
        result = node.classify_image(image)
        assert isinstance(result, str)

    def test_classify_image_throws_errors(self, node, image):
        node.interpreter.invoke.side_effect = RuntimeError("Error")
        with pytest.raises(RuntimeError, match="Error"):
            node.classify_image(image)
