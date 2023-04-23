import pytest

from document_processor.pipeline.pipeline import DocumentProcessorPipeline
from document_processor.pipeline.pipeline_nodes import DocumentProcessingNode


class Test_DocumentProcessorPipeline:
    @pytest.fixture
    def pipeline(self):
        return DocumentProcessorPipeline()

    @pytest.fixture
    def nodes(self, mocker):
        node1 = mocker.Mock(DocumentProcessingNode)
        node2 = mocker.Mock(DocumentProcessingNode)
        return node1, node2

    def test_addProcessingNode(self, pipeline, nodes):
        pipeline.addProcessingNode(nodes[0])
        assert nodes[0] in pipeline.processing_nodes

    def test_processDocument(self, mocker, pipeline, nodes):
        for node in nodes:
            pipeline.addProcessingNode(node)

        data = {"text": "Hello world!"}
        expected_res_data = {"text": "Better Hello world!"}

        for node in nodes:
            node.processDocument.return_value = expected_res_data

        result = pipeline.processDocument(data)

        nodes[0].processDocument.assert_called_once_with(data)
        nodes[1].processDocument.assert_called_once_with(expected_res_data)

        assert result["text"] == expected_res_data["text"]
