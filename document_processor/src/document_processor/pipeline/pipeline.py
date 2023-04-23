from abc import ABC

from .pipeline_nodes import DocumentProcessingNode


class DocumentProcessorPipeline(ABC):
    def __init__(self):
        self.processing_nodes: list[DocumentProcessingNode] = []

    def addProcessingNode(self, node: DocumentProcessingNode):
        self.processing_nodes.append(node)

    def processDocument(self, data: dict):
        for node in self.processing_nodes:
            data = node.processDocument(data)
        return data
