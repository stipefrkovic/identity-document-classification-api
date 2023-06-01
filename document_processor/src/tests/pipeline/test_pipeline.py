import pytest

from document_processor.pipeline.pipeline import DocumentProcessorPipeline
from document_processor.pipeline.pipeline_nodes import DocumentProcessingNode


class TestDocumentProcessorPipeline:
    @pytest.fixture
    def pipeline(self):
        return DocumentProcessorPipeline()

    @pytest.fixture
    def nodes(self, mocker):
        node1 = mocker.Mock(DocumentProcessingNode)
        node2 = mocker.Mock(DocumentProcessingNode)
        return node1, node2

    def test_addProcessingNode(self, pipeline, nodes):
        pipeline.add_processing_node(nodes[0])
        assert nodes[0] in pipeline.processing_nodes

    def test_processDocument(self, mocker, pipeline, nodes):
        for node in nodes:
            pipeline.add_processing_node(node)

        data = {"text": "Hello world!"}
        expected_res_data = {"text": "Better Hello world!"}

        for node in nodes:
            node.process_document.return_value = expected_res_data

        result = pipeline.process_document(data)

        nodes[0].process_document.assert_called_once_with(data)
        nodes[1].process_document.assert_called_once_with(expected_res_data)

        assert result["text"] == expected_res_data["text"]
