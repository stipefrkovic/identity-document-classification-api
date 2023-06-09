import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from contextlib import asynccontextmanager

from document_processor.logger import logger
from document_processor.document_processor import PDFDocumentProcessor
from document_processor.pipeline.builder import (
    EffNetDocumentProcessorPipelineBuilder,
    EffDetDocumentProcessorPipelineBuilder,
)

DEFAULT_MIN_CONFIDENCE = 0.5
app = FastAPI()
document_processor = None


def get_env_vars():
    # Model to use is loaded from environment variable
    model = os.getenv("MODEL")

    # read min confidence from environment variable as int
    min_confidence = float(os.getenv("MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE))

    return model, min_confidence


def get_pipeline_builder(model):
    if model == "EFFICIENTNET":
        pipeline_builder = EffNetDocumentProcessorPipelineBuilder()
        model_directory = "./src/document_processor/pipeline/models/effnet"
    elif model == "EFFICIENTDET":
        pipeline_builder = EffDetDocumentProcessorPipelineBuilder()
        model_directory = (
            "./src/document_processor/pipeline/models/effdet/saved_model/saved_model"
        )
    else:
        raise ValueError("Invalid model specified in environment variable MODEL")

    return pipeline_builder, model_directory


if "TESTING" not in os.environ:
    model, min_confidence = get_env_vars()
    logger.info("Starting the API")
    pipeline_builder, model_directory = get_pipeline_builder(model)

    document_processor = PDFDocumentProcessor(
        pipeline_builder, model_directory=model_directory, min_confidence=min_confidence
    )


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
