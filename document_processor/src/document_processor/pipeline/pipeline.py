from abc import ABC

from .pipeline_nodes import DocumentProcessingNode


class DocumentProcessorPipeline(ABC):
    def __init__(self):
        self.processing_nodes: list[DocumentProcessingNode] = []

    def add_processing_node(self, node: DocumentProcessingNode):
        self.processing_nodes.append(node)

    def process_document(self, data: dict):
        for node in self.processing_nodes:
            data = node.process_document(data)
        return data
