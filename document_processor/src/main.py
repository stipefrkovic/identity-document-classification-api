import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from document_processor.document_processor import PDFDocumentProcessor
from document_processor.pipeline.builder import (
    EffNetDocumentProcessorPipelineBuilder,
    EffDetDocumentProcessorPipelineBuilder,
)

app = FastAPI()

# Model to use is loaded from environment variable
model = os.getenv("MODEL")

# read min confidence from environment variable as int

min_confidence = os.getenv("MIN_CONFIDENCE")

if min_confidence is None:
    min_confidence = 0.5
else:
    min_confidence = float(min_confidence)

if model == "EFFICIENTNET":
    pipeline_builder = EffNetDocumentProcessorPipelineBuilder()
elif model == "EFFICIENTDET":
    pipeline_builder = EffDetDocumentProcessorPipelineBuilder()

document_processor = PDFDocumentProcessor(pipeline_builder, min_confidence)


@app.get("/")
def read_root():
    return {"message": "Test"}


class DocumentTypeResponse(BaseModel):
    document_type: str
    meta: dict


def check_document(document: File):
    return document.content_type == "application/pdf"


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
            "meta": {
                "filename": document.filename,
                "prediction_confidences": data.get("prediction_confidences", None),
            },
        }
    else:
        return {"document_type": "unknown", "meta": {"filename": document.filename}}
