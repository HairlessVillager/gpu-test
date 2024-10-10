from time import time
from typing import Annotated, List
from logging import getLogger
from uuid import uuid4
import asyncio
import os

from fastapi import FastAPI, Body, Header
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
logger = getLogger("app")
BATCH_TIMEOUT = os.getenv("BATCH_TIMEOUT")
BATCH_SIZE_MAX = os.getenv("BATCH_SIZE_MAX")


class DebugContext:

    def __init__(self, uuid: str, start: float):
        self.uuid = uuid
        self.start = start
        self.tags = {}

    def debug(self, msg: str):
        logger.debug(f"{self.uuid} after {time()-self.start:.6f}, {self.tags}: {msg}")

    def tag(self, key: str, value):
        new_ctx = DebugContext(uuid=self.uuid, start=self.start)
        new_ctx.tags[key] = value
        return new_ctx


async def server_loop(mq: asyncio.Queue):
    logger.info(f"{BATCH_TIMEOUT=}")
    logger.info(f"{BATCH_SIZE_MAX=}")
    logger.debug("loadding pipeline...")
    pipe = pipeline(
        "text-classification", model="TrustSafeAI/RADAR-Vicuna-7B", device="cuda:0"
    )
    logger.debug("loadding pipeline...done")
    while True:
        texts = []
        rqs = []
        try:
            async with asyncio.timeout(BATCH_TIMEOUT):
                for _ in range(BATCH_SIZE_MAX):
                    text, rq, ctx = await mq.get()
                    texts.append(text)
                    rqs.append(rq)
                    ctx.debug("received from mq")
        except asyncio.TimeoutError:
            pass

        if texts:
            batch_size = len(texts)
            logger.debug("inferenceing...")
            logger.info(f"{batch_size=}")
            results = pipe(texts, batch_size=batch_size)
            logger.debug("inferenceing...done")

            for result, rq in zip(results, rqs):
                rq.put_nowait(result)


@app.on_event("startup")
async def startup():
    mq = asyncio.Queue()
    app.mq = mq
    asyncio.create_task(server_loop(mq))


class DocumentResult(BaseModel):
    label: str
    score: float


class AIDetectionResponseModel(BaseModel):
    version: str
    scanId: str
    documents: List[DocumentResult]


async def analyze_and_classify(text: str, ctx: DebugContext):
    rq = asyncio.Queue()
    ctx.debug("wait to put")
    await app.mq.put((text, rq, ctx))
    ctx.debug("put, wait to get")
    output = await rq.get()
    ctx.debug("got")
    return output


def split_document_by_length(document: str, segment_length: int) -> List[str]:
    return [
        document[i : i + segment_length]
        for i in range(0, len(document), segment_length)
    ]


@app.post("/predit/text", response_model=AIDetectionResponseModel)
async def ai_detection_on_single_string(
    document: Annotated[str, Body()],
    api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
    version: Annotated[str, Body()] = "no-version",
    multilingual: Annotated[bool, Body()] = False,
):
    scanId = uuid4().hex
    start = time()
    ctx = DebugContext(scanId, start)
    ctx.debug(f"received request, length={len(document)}")

    tasks = []
    segments = split_document_by_length(document, 300)
    for i, segment in enumerate(segments):
        task = analyze_and_classify(segment, ctx.tag("id", i))
        tasks.append(task)
    results = await asyncio.gather(*tasks)

    documents = [
        {**result, "original_paragraph": segment}
        for segment, result in zip(segments, results)
    ]
    ctx.debug("completed")
    return {
        "version": "no-version",
        "scanId": scanId,
        "documents": documents,
    }
