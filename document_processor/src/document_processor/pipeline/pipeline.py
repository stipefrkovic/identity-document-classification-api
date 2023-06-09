from .pipeline_nodes import DocumentProcessingNode


class DocumentProcessorPipeline:
    """
    DocumentProcessorPipeline that will process the document by iterating through its nodes.
    """
    def __init__(self):
        """
        Initializes the DocumentProcessorPipeline.
        """
        self.processing_nodes: list[DocumentProcessingNode] = []

    def add_processing_node(self, node: DocumentProcessingNode):
        """
        Adds a DocumentProcessingNode to the pipeline.
        :param node: DocumentProcessingNode to be added.
        """
        self.processing_nodes.append(node)

    def process_document(self, data: dict):
        """
        Processes a document by iterating through its nodes.
        :param data: Dictionary that contains the document and will be processed when iterating through the pipeline.
        :return: Dictionary that was processed after iterated through the pipeline.
        """
        for node in self.processing_nodes:
            data = node.process_document(data)
        return data
