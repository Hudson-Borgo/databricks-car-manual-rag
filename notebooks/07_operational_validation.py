# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 07 — Validate the completed RAG project
# MAGIC
# MAGIC This final checkpoint verifies the governed data, online AI Search index,
# MAGIC Foundation Model API endpoint, and recent MLflow traces without creating a
# MAGIC custom production endpoint. The working learning application is the traced
# MAGIC RAG chain from Notebook 05.
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC - Distinguish a development baseline from a production deployment.
# MAGIC - Validate the resources required to operate the RAG chain.
# MAGIC - Connect data quality, serving readiness, tracing, and monitoring.
# MAGIC
# MAGIC **Expected input:** successful Notebooks 04–06.
# MAGIC
# MAGIC **Expected output:** one PASS/FAIL project-readiness table.
# MAGIC
# MAGIC **Exam alignment:** Assembling and Deploying Applications, Governance,
# MAGIC and Evaluation and Monitoring.

# COMMAND ----------

# MAGIC %pip install -U "mlflow[databricks]>=3.10" databricks-ai-search
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks.ai_search.client import AISearchClient
from databricks.sdk import WorkspaceClient

CATALOG = "workspace"
SCHEMA = "car_manual_rag"
CHUNK_TABLE = f"{CATALOG}.{SCHEMA}.document_chunks"
SEARCH_ENDPOINT = "car_manual_rag_endpoint"
SEARCH_INDEX = f"{CATALOG}.{SCHEMA}.car_manual_chunks_index"
LLM_MODEL = "databricks-qwen3-next-80b-a3b-instruct"

mlflow.set_tracking_uri("databricks")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Inspect governed data and serving resources

# COMMAND ----------

chunk_count = spark.table(CHUNK_TABLE).count()
table_properties = {
    row["key"]: row["value"]
    for row in spark.sql(f"SHOW TBLPROPERTIES {CHUNK_TABLE}").collect()
}
cdf_enabled = table_properties.get("delta.enableChangeDataFeed") == "true"

workspace_client = WorkspaceClient()
serving_endpoint_names = {
    endpoint.name
    for endpoint in workspace_client.serving_endpoints.list()
}
llm_available = LLM_MODEL in serving_endpoint_names

search_index = AISearchClient().get_index(
    endpoint_name=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
)
index_description = search_index.describe()
index_status = index_description.get("status", {})
index_online = bool(
    index_status.get("ready")
    or index_status.get("message", "").upper() == "ONLINE"
    or index_status.get("detailed_state", "").upper() == "ONLINE"
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Confirm recent application traces

# COMMAND ----------

recent_traces = mlflow.search_traces(
    filter_string=(
        "attributes.status = 'OK' AND "
        "tags.`mlflow.traceName` = 'rag_chain'"
    ),
    order_by=["attributes.timestamp_ms DESC"],
    max_results=2,
    return_type="list",
)

traces_have_retrieval = all(
    trace.search_spans(span_type="RETRIEVER")
    for trace in recent_traces
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Project readiness checkpoint

# COMMAND ----------

checks = [
    ("Unity Catalog chunk table", chunk_count, chunk_count > 0),
    ("Delta Change Data Feed", str(cdf_enabled), cdf_enabled),
    ("AI Search index online", str(index_online), index_online),
    ("Foundation Model API endpoint", LLM_MODEL, llm_available),
    ("Recent RAG traces", len(recent_traces), len(recent_traces) == 2),
    (
        "Retriever spans in recent traces",
        str(traces_have_retrieval),
        len(recent_traces) == 2 and traces_have_retrieval,
    ),
]

readiness = spark.createDataFrame(
    [
        (name, str(observed), "PASS" if passed else "FAIL")
        for name, observed, passed in checks
    ],
    "check STRING, observed_value STRING, status STRING",
)

display(readiness)

failed_checks = [name for name, _, passed in checks if not passed]
assert not failed_checks, f"Project validation failed: {failed_checks}"

print("PROJECT STATUS: COMPLETE")
print(f"Governed chunks: {chunk_count}")
print(f"AI Search index: {SEARCH_INDEX}")
print(f"Generation endpoint: {LLM_MODEL}")
print(f"Validated traces: {[trace.info.trace_id for trace in recent_traces]}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Deployment boundary
# MAGIC
# MAGIC This learning project validates the complete notebook-based RAG path:
# MAGIC governed PDFs and Delta tables, AI Search retrieval, Foundation Model API
# MAGIC generation, citations, MLflow tracing, and evaluation. A dedicated custom
# MAGIC Model Serving endpoint is intentionally out of scope for this small Free
# MAGIC Edition baseline; the repository documents it as the next production step.
# MAGIC
# MAGIC ## Knowledge check
# MAGIC
# MAGIC Which signal is most useful for detecting a retrieval-quality regression
# MAGIC after a new manual version is ingested?
# MAGIC
# MAGIC A. The number of Python files in Git.
# MAGIC
# MAGIC B. Retrieval-relevance results on a stable evaluation dataset, segmented
# MAGIC by document version.
# MAGIC
# MAGIC C. The chat model endpoint name.
# MAGIC
# MAGIC D. The notebook's font size.
