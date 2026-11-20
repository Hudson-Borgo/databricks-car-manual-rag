"""Databricks integrations for retrieval, generation, and MLflow tracing."""

import os
from functools import lru_cache

import mlflow
from databricks.ai_search.client import AISearchClient
from databricks_openai import DatabricksOpenAI
from mlflow.entities import Document, SpanType

from rag_core import (
    DOCUMENTS,
    SYSTEM_PROMPT,
    build_user_prompt,
    search_result_rows,
    validate_vehicle,
)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required Databricks App resource {name!r} is missing.")
    return value


def _workspace_url() -> str:
    host = _required_environment("DATABRICKS_HOST")
    if not host.startswith(("https://", "http://")):
        host = f"https://{host}"
    return host


def configure_mlflow() -> None:
    """Route application traces to the resource-bound MLflow experiment."""
    mlflow.set_tracking_uri("databricks")
    experiment_id = _required_environment("MLFLOW_EXPERIMENT_ID")
    mlflow.set_experiment(experiment_id=experiment_id)
    mlflow.openai.autolog()


@lru_cache(maxsize=1)
def _search_index():
    client = AISearchClient(
        workspace_url=_workspace_url(),
        service_principal_client_id=_required_environment("DATABRICKS_CLIENT_ID"),
        service_principal_client_secret=_required_environment(
            "DATABRICKS_CLIENT_SECRET"
        ),
    )
    return client.get_index(index_name=_required_environment("VECTOR_SEARCH_INDEX"))


@lru_cache(maxsize=1)
def _llm_client() -> DatabricksOpenAI:
    return DatabricksOpenAI()


configure_mlflow()


@mlflow.trace(name="retrieve_chunks", span_type=SpanType.RETRIEVER)
def retrieve(question: str, vehicle: str, *, k: int = 5) -> list[dict]:
    """Run metadata-filtered hybrid retrieval for one vehicle manual."""
    selected_vehicle = validate_vehicle(vehicle)
    expected_document_id = DOCUMENTS[selected_vehicle]["document_id"]

    results = _search_index().similarity_search(
        query_text=question,
        columns=[
            "chunk_id",
            "chunk_text",
            "document_id",
            "manufacturer",
            "model",
            "model_year",
            "section_title",
            "pdf_page_numbers",
        ],
        filters={"document_id": expected_document_id},
        num_results=k,
        query_type="hybrid",
    )

    chunks = search_result_rows(results)
    if not all(chunk["document_id"] == expected_document_id for chunk in chunks):
        raise RuntimeError("AI Search returned evidence from the wrong vehicle.")

    documents = [
        Document(
            id=str(chunk["chunk_id"]),
            page_content=chunk["chunk_text"],
            metadata={
                "doc_uri": (
                    f"{chunk['document_id']}#pdf-pages={chunk['pdf_page_numbers']}"
                ),
                "document_id": chunk["document_id"],
                "section_title": chunk.get("section_title"),
                "pdf_page_numbers": chunk["pdf_page_numbers"],
            },
        )
        for chunk in chunks
    ]
    mlflow.get_current_active_span().set_outputs(documents)
    return chunks


@mlflow.trace(name="rag_chain", span_type=SpanType.CHAIN)
def ask_manual(question: str, vehicle: str) -> dict:
    """Retrieve evidence and return a grounded, cited answer."""
    selected_vehicle = validate_vehicle(vehicle)
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("Question must not be empty.")

    chunks = retrieve(clean_question, selected_vehicle, k=5)
    if not chunks:
        return {
            "answer": "Não encontrei informações no manual selecionado.",
            "sources": [],
        }

    response = _llm_client().chat.completions.create(
        model=_required_environment("SERVING_ENDPOINT"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    clean_question,
                    selected_vehicle,
                    chunks,
                ),
            },
        ],
        temperature=0,
        max_tokens=500,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": chunks,
    }
