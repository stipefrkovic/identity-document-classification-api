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

    @pytest.fixture
    def data(self):
        return {"text": "Hello world!"}

    def test_initial_pipeline_is_empty(self, pipeline):
        assert not pipeline.processing_nodes

    def test_process_document_no_processing_for_empty_pipeline(self, pipeline, data):
        result = pipeline.process_document(data)
        assert result == data

    def test_add_processing_node(self, pipeline, nodes):
        pipeline.add_processing_node(nodes[0])
        assert nodes[0] in pipeline.processing_nodes

    def test_process_document(self, pipeline, nodes, data):
        for node in nodes:
            pipeline.add_processing_node(node)

        expected_res_data = {"text": "Better Hello world!"}

        for node in nodes:
            node.process_document.return_value = expected_res_data

        result = pipeline.process_document(data)

        nodes[0].process_document.assert_called_once_with(data)
        nodes[1].process_document.assert_called_once_with(expected_res_data)

        assert result["text"] == expected_res_data["text"]

    def test_process_document_handles_node_errors(self, pipeline, nodes, data):
        pipeline.add_processing_node(nodes[0])

        error_message = "An error occurred during processing"
        nodes[0].process_document.side_effect = Exception(error_message)

        with pytest.raises(Exception, match=error_message):
            pipeline.process_document(data)