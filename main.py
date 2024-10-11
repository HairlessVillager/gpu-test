from time import time
from typing import Union
from typing_extensions import Annotated, List
from logging import getLogger
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Body, Header
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
logger = getLogger("app")
pipe = pipeline(
    "text-classification", model="TrustSafeAI/RADAR-Vicuna-7B", device="cuda:0"
)


class DocumentResult(BaseModel):
    label: str
    score: float


class AIDetectionResponseModel(BaseModel):
    version: str
    scanId: str
    documents: List[DocumentResult]


def analyze_and_classify(text: str):
    return pipe(text)[0]


def split_document_by_length(document: str, segment_length: int) -> List[str]:
    return [
        document[i : i + segment_length]
        for i in range(0, len(document), segment_length)
    ]


@app.post("/predit/text", response_model=AIDetectionResponseModel)
def ai_detection_on_single_string(
    document: Annotated[str, Body()],
    api_key: Annotated[Union[str, None], Header(alias="x-api-key")] = None,
    version: Annotated[str, Body()] = "no-version",
    multilingual: Annotated[bool, Body()] = False,
):
    start = time()
    documents = []
    for doc in split_document_by_length(document, 300):
        result = analyze_and_classify(doc)
        documents.append(
            {
                **result,
                "original_paragraph": doc,
            }
        )
    end = time()
    cost = end - start
    length = len(document)
    logger.debug(f"{cost:.6f} / {length} = {cost / length :.6f}")
    return {
        "version": "no-version",
        "scanId": uuid4().hex,
        "documents": documents,
    }
