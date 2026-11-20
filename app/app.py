"""FastAPI entry point for the Databricks Car Manual RAG App."""

from pathlib import Path
from typing import Literal

import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag_core import DOCUMENTS
from rag_service import ask_manual

app = FastAPI(
    title="Databricks Car Manual RAG",
    description="Grounded answers from Peugeot 208 and Hyundai HB20 manuals.",
    version="1.0.0",
)

INDEX_HTML = Path(__file__).with_name("index.html")


class AskRequest(BaseModel):
    vehicle: Literal["peugeot", "hyundai"]
    question: str = Field(min_length=3, max_length=1000)


class SourceResponse(BaseModel):
    number: int
    document_id: str
    manufacturer: str
    model: str
    model_year: int
    section_title: str | None
    pdf_page_numbers: list[float] | list[int] | str
    text_preview: str


class AskResponse(BaseModel):
    vehicle: str
    answer: str
    sources: list[SourceResponse]
    trace_id: str | None


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/vehicles")
def vehicles() -> dict:
    return DOCUMENTS


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        result = await run_in_threadpool(
            ask_manual,
            request.question,
            request.vehicle,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="The RAG request failed. Check the Databricks App logs.",
        ) from error

    sources = [
        SourceResponse(
            number=number,
            document_id=source["document_id"],
            manufacturer=source["manufacturer"],
            model=source["model"],
            model_year=int(source["model_year"]),
            section_title=source.get("section_title"),
            pdf_page_numbers=source["pdf_page_numbers"],
            text_preview=source["chunk_text"][:600],
        )
        for number, source in enumerate(result["sources"], start=1)
    ]

    return AskResponse(
        vehicle=DOCUMENTS[request.vehicle]["label"],
        answer=result["answer"],
        sources=sources,
        trace_id=mlflow.get_last_active_trace_id(),
    )
