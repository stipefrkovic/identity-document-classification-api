import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from document_processor.pipeline.pdf_to_image_converter import PdfToImageConverter
from document_processor.pipeline.pipeline_nodes import (
    DocumentProcessingNode,
    NNDocumentIdentifierNode,
    PdfToImageConverterNode,
)


class TestDocumentProcessingNode:
    def test_processDocument_not_implemented(self):
        with pytest.raises(TypeError):
            DocumentProcessingNode().process_document(data={})


class TestPdfToImageConverterNode:
    def test_pdf_to_jpg_conversion(self, mocker):
        pdf_bytes = b"test pdf bytes"
        jpg_bytes = b"test jpg bytes"

        # Mock the converter
        converter_mock = mocker.Mock(spec=PdfToImageConverter)
        converter_mock.convert.return_value = jpg_bytes

        node = PdfToImageConverterNode(converter=converter_mock)
        data = {"pdf_bytes": pdf_bytes}

        # Run test
        result = node.process_document(data)

        assert converter_mock.convert.called_once_with(pdf_bytes)
        assert result["jpg_bytes"] == jpg_bytes
        assert result["pdf_bytes"] == pdf_bytes


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

    def test_classify_image_valid_output(self, node, image):

        # Run test
        result = node.classify_image(image)

        assert isinstance(result, str)
        assert len(result) > 0
        node.interpreter.get_output_details.assert_called_once()
        node.interpreter.allocate_tensors.assert_called_once()
        node.interpreter.get_input_details.assert_called_once()
        node.interpreter.invoke.assert_called_once()

    # This should be refactored once we return probability matrix.
    # def test_classify_image_probabilities_sum_to_one(self, mocker, image):
    #     model_path = "./src/document_processor/pipeline/model.tflite"
    #     node = NNDocumentIdentifierNode(model_path=model_path)

    #     # Mock the interpreter object
    #     mock_interpreter = mocker.Mock(spec=tf.lite.Interpreter)
    #     node.interpreter = mock_interpreter
    #     node.interpreter.get_output_details.return_value = [{"index": 0}]
    #     node.interpreter.get_tensor.side_effect = [np.array([[0.1, 0.3, 0.6]], dtype=np.float32)]

    #     # Run test
    #     result = node.classify_image(image)
    #     probabilities = node.interpreter.get_tensor(0)[0]

    #     assert isinstance(result, str)
    #     assert len(result) > 0
    #     assert sum(probabilities) == pytest.approx(1.0)
    #     node.interpreter.get_output_details.assert_called_once()
    #     node.interpreter.get_tensor.assert_called_once_with(0)

    def test_classify_image_throws_errors(self, node, image):
        node.interpreter.invoke.side_effect = RuntimeError("Sample error")

        with pytest.raises(RuntimeError, match="Sample error"):
            node.classify_image(image)

        node.interpreter.allocate_tensors.assert_called_once()
        node.interpreter.get_input_details.assert_called_once()
        node.interpreter.set_tensor.assert_called_once()
        node.interpreter.invoke.assert_called_once()

    @pytest.fixture
    def mock_node(self, mocker):
        mock_node = mocker.Mock(spec=NNDocumentIdentifierNode)
        mock_node.process_document.return_value = {"document_type": "passport"}

        return mock_node

    @pytest.fixture
    def jpg_bytes(self):
        jpg_bytes = b"test jpg bytes"
        return jpg_bytes

    @pytest.fixture
    def input_data(self, jpg_bytes):
        return {"jpg_bytes": jpg_bytes}

    def test_processDocument_returns_classification_result(self, mock_node, input_data):
        result = mock_node.process_document(input_data)

        assert "document_type" in result
        assert result["document_type"] == "passport"
