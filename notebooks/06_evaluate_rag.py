# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 06 — Evaluate the baseline RAG application
# MAGIC
# MAGIC This notebook turns the latest target and unsupported-question traces into
# MAGIC a versioned MLflow evaluation dataset in Unity Catalog and applies built-in
# MAGIC RAG judges plus project-specific guidelines.
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC - Distinguish retrieval relevance, retrieval sufficiency, groundedness,
# MAGIC   answer relevance, and deterministic business rules.
# MAGIC - Build a reusable evaluation dataset from production-shaped traces.
# MAGIC - Apply built-in MLflow LLM judges and custom guidelines.
# MAGIC - Inspect aggregate metrics and per-example rationales.
# MAGIC
# MAGIC **Expected input:** two recent `rag_chain` traces created by rerunning
# MAGIC Notebook 05 after its retriever-span update.
# MAGIC
# MAGIC **Expected output:** a Unity Catalog evaluation dataset, an MLflow
# MAGIC evaluation run, aggregate pass rates, and per-trace feedback.
# MAGIC
# MAGIC **Exam alignment:** Governance and Evaluation and Monitoring.

# COMMAND ----------

# MAGIC %pip install -U "mlflow[databricks]>=3.10"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from mlflow.genai.datasets import create_dataset, get_dataset
from mlflow.genai.scorers import (
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    Safety,
)

EVALUATION_DATASET = "workspace.car_manual_rag.rag_evaluation_dataset"
MLFLOW_EXPERIMENT = "/Users/hudsonborgo98@gmail.com/car-manual-llm"
EXPECTED_TRACE_COUNT = 2

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load the latest successful RAG traces
# MAGIC
# MAGIC We evaluate completed application traces instead of isolated answer text.
# MAGIC This gives the RAG judges access to the retriever span and its documents.

# COMMAND ----------

latest_traces = mlflow.search_traces(
    filter_string=(
        "attributes.status = 'OK' AND "
        "tags.`mlflow.traceName` = 'rag_chain'"
    ),
    order_by=["attributes.timestamp_ms DESC"],
    max_results=EXPECTED_TRACE_COUNT,
    return_type="list",
)

assert len(latest_traces) == EXPECTED_TRACE_COUNT, (
    "Expected two recent rag_chain traces. Pull the latest repository changes "
    "and rerun Notebook 05 before continuing."
)

for trace in latest_traces:
    print(
        f"Trace ID: {trace.info.trace_id} | "
        f"status: {trace.info.status} | "
        f"request: {trace.data.request}"
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Validate the trace schema
# MAGIC
# MAGIC Retrieval-specific judges require at least one span with type
# MAGIC `RETRIEVER`. This validation prevents an evaluation run that silently lacks
# MAGIC retrieval context.

# COMMAND ----------

trace_schema_rows = []

for trace in latest_traces:
    retriever_spans = trace.search_spans(span_type="RETRIEVER")
    retriever_document_count = sum(
        len(span.outputs or [])
        for span in retriever_spans
    )
    schema_status = (
        "PASS"
        if retriever_spans and retriever_document_count > 0
        else "FAIL"
    )
    trace_schema_rows.append(
        (
            trace.info.trace_id,
            len(retriever_spans),
            retriever_document_count,
            schema_status,
        )
    )

assert all(row[-1] == "PASS" for row in trace_schema_rows), (
    "The latest traces do not contain evaluation-compatible retrieval output."
)

display(
    spark.createDataFrame(
        trace_schema_rows,
        """
        trace_id STRING,
        retriever_span_count INT,
        retrieved_document_count INT,
        status STRING
        """,
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Create or update a Unity Catalog evaluation dataset
# MAGIC
# MAGIC Evaluation datasets make test cases versioned, reusable, and governed.
# MAGIC Merging the traces preserves the application inputs, outputs, and span data
# MAGIC required by the RAG judges.

# COMMAND ----------

try:
    evaluation_dataset = get_dataset(name=EVALUATION_DATASET)
    dataset_action = "loaded"
except Exception:
    evaluation_dataset = create_dataset(name=EVALUATION_DATASET)
    dataset_action = "created"

evaluation_dataset = evaluation_dataset.merge_records(latest_traces)
evaluation_records = evaluation_dataset.to_df()

print(f"Dataset action: {dataset_action}")
print(f"Dataset name: {evaluation_dataset.name}")
print(f"Dataset version: {evaluation_dataset.version}")
print(f"Record count: {len(evaluation_records)}")
display(evaluation_records)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Define the quality dimensions
# MAGIC
# MAGIC Built-in judges cover common RAG dimensions. Two project-specific
# MAGIC guidelines target issues observed in the baseline output:
# MAGIC
# MAGIC - Unsupported answers should not cite unrelated chunks as evidence.
# MAGIC - Actionable instructions should not leave unexplained references such as
# MAGIC   “perform action (2).”

# COMMAND ----------

scorers = [
    RetrievalGroundedness(),
    RetrievalRelevance(),
    RelevanceToQuery(),
    Safety(),
    Guidelines(
        name="citation_quality",
        guidelines=(
            "Factual vehicle-manual claims must cite source identifiers. "
            "If the answer says the requested information is unavailable, it "
            "must not present unrelated retrieved chunks as supporting evidence."
        ),
    ),
    Guidelines(
        name="actionable_answer",
        guidelines=(
            "The answer must be understandable and actionable on its own. It "
            "must not leave unexplained cross-references such as '(1)', '(2)', "
            "or 'the indicated action'."
        ),
    ),
]

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Run the evaluation
# MAGIC
# MAGIC Each built-in judge attaches feedback and a rationale to every evaluated
# MAGIC trace. The result also records aggregate metrics for later comparisons.

# COMMAND ----------

evaluation_result = mlflow.genai.evaluate(
    data=evaluation_dataset,
    scorers=scorers,
)

print("Aggregate metrics:")
for metric_name, metric_value in sorted(evaluation_result.metrics.items()):
    print(f"- {metric_name}: {metric_value}")

display(evaluation_result.result_df)

# COMMAND ----------
# MAGIC %md
# MAGIC ## How to interpret the baseline
# MAGIC
# MAGIC - **Retrieval relevance:** Were the returned chunks useful for the query?
# MAGIC - **Retrieval groundedness:** Are answer claims supported by those chunks?
# MAGIC - **Relevance to query:** Did the answer address what the user asked?
# MAGIC - **Safety:** Did the answer avoid harmful or inappropriate content?
# MAGIC - **Citation quality:** Were citations meaningful rather than decorative?
# MAGIC - **Actionable answer:** Did the response resolve manual cross-references?
# MAGIC
# MAGIC A failed scorer is a diagnostic signal, not a reason to hide the test case.
# MAGIC We will use the rationales to choose the smallest justified improvement.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Knowledge check
# MAGIC
# MAGIC Which evaluation result most strongly suggests a **retrieval** problem
# MAGIC rather than a generation/prompt problem?
# MAGIC
# MAGIC A. Retrieval relevance fails because top-ranked chunks do not address the
# MAGIC question.
# MAGIC
# MAGIC B. Safety fails even though all retrieved chunks are relevant.
# MAGIC
# MAGIC C. Citation formatting is inconsistent while retrieved evidence is
# MAGIC sufficient.
# MAGIC
# MAGIC D. The answer uses a long sentence that is fully supported by the context.
