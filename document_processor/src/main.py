from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from document_processor.document_processor import MLModelDocumentProcessor
from document_processor.pipeline.builder import (
    EffNetDocumentProcessorPipelineBuilder,
    EffDetDocumentProcessorPipelineBuilder
)

app = FastAPI()

pipeline_builder = EffNetDocumentProcessorPipelineBuilder()
document_processor = MLModelDocumentProcessor(pipeline_builder)


@app.get("/")
def read_root():
    return {"message": "Test"}


class DocumentTypeResponse(BaseModel):
    document_type: str
    meta: dict


def check_document(document: File):
    if document.content_type == "application/pdf":
        return True
    return False


@app.post("/document/")
async def process_document(document: UploadFile):
    if not check_document(document):
        raise HTTPException(
            status_code=400, detail="Invalid file type. File must be pdf."
        )

    byte_file = await document.read()

    data = document_processor.process_document(byte_file)

    if data.get("document_type", None) is not None:
        return {
            "document_type": data.get("document_type"),
            "meta": {"filename": document.filename},
        }
    else:
        return {"document_type": "unknown", "meta": {"filename": document.filename}}
