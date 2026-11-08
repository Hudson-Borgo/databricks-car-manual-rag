# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 04 — Create a Delta Sync AI Search index
# MAGIC
# MAGIC This notebook validates the chunk table and prepares it for a Databricks
# MAGIC AI Search index. The endpoint and index are then created in the Databricks
# MAGIC UI so that each managed component is visible and inspectable.
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC - Distinguish an AI Search endpoint from an AI Search index.
# MAGIC - Explain Delta Sync versus Direct Vector Access indexes.
# MAGIC - Use a Databricks-managed embedding model.
# MAGIC - Enable incremental synchronization with Delta Change Data Feed.
# MAGIC - Configure hybrid retrieval and metadata columns.
# MAGIC
# MAGIC **Exam alignment:** Data Preparation, Application Development, and
# MAGIC Assembling and Deploying Applications.

# COMMAND ----------

from pyspark.sql import functions as F

CATALOG = "workspace"
SCHEMA = "car_manual_rag"
CHUNK_TABLE = "document_chunks"

AI_SEARCH_ENDPOINT = "car_manual_rag_endpoint"
AI_SEARCH_INDEX = "car_manual_chunks_index"

FULL_CHUNK_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{CHUNK_TABLE}"
FULL_AI_SEARCH_INDEX_NAME = f"{CATALOG}.{SCHEMA}.{AI_SEARCH_INDEX}"

print(f"Source table: {FULL_CHUNK_TABLE_NAME}")
print(f"Planned endpoint: {AI_SEARCH_ENDPOINT}")
print(f"Planned index: {FULL_AI_SEARCH_INDEX_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Validate the index source
# MAGIC
# MAGIC `chunk_id` is the AI Search primary key, and `chunk_text` is the embedding
# MAGIC source. Vehicle and page metadata will support filters, result inspection,
# MAGIC and citations.

# COMMAND ----------

chunks = spark.table(FULL_CHUNK_TABLE_NAME)

chunk_count = chunks.count()
distinct_primary_key_count = chunks.select("chunk_id").distinct().count()
null_primary_key_count = chunks.filter(F.col("chunk_id").isNull()).count()
empty_embedding_source_count = chunks.filter(
    F.col("chunk_text").isNull() | (F.trim("chunk_text") == "")
).count()
missing_filter_metadata_count = chunks.filter(
    F.col("document_id").isNull()
    | F.col("manufacturer").isNull()
    | F.col("model").isNull()
    | F.col("model_year").isNull()
).count()
missing_citation_metadata_count = chunks.filter(
    F.col("pdf_page_numbers").isNull()
    | (F.size("pdf_page_numbers") == 0)
).count()

assert chunk_count > 0, "The source table must contain chunks."
assert chunk_count == distinct_primary_key_count, "chunk_id must be unique."
assert null_primary_key_count == 0, "chunk_id must not be null."
assert empty_embedding_source_count == 0, "chunk_text must not be empty."
assert missing_filter_metadata_count == 0, "Vehicle metadata is incomplete."
assert missing_citation_metadata_count == 0, "PDF page metadata is incomplete."

source_validation = spark.createDataFrame(
    [
        ("chunk_count", chunk_count, "PASS"),
        (
            "distinct_primary_key_count",
            distinct_primary_key_count,
            "PASS",
        ),
        ("null_primary_key_count", null_primary_key_count, "PASS"),
        (
            "empty_embedding_source_count",
            empty_embedding_source_count,
            "PASS",
        ),
        (
            "missing_filter_metadata_count",
            missing_filter_metadata_count,
            "PASS",
        ),
        (
            "missing_citation_metadata_count",
            missing_citation_metadata_count,
            "PASS",
        ),
    ],
    ["check", "observed_value", "status"],
)

display(source_validation)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Enable Delta Change Data Feed
# MAGIC
# MAGIC A Delta Sync index on a Standard endpoint requires a source change data
# MAGIC feed. It lets later syncs process inserts, updates, and deletes
# MAGIC incrementally instead of rebuilding our application-side index logic.

# COMMAND ----------

spark.sql(
    f"""
    ALTER TABLE {FULL_CHUNK_TABLE_NAME}
    SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    """
)

cdf_property = spark.sql(
    f"""
    SHOW TBLPROPERTIES {FULL_CHUNK_TABLE_NAME}
    ('delta.enableChangeDataFeed')
    """
).first()

cdf_enabled = str(cdf_property["value"]).lower() == "true"
assert cdf_enabled, "Delta Change Data Feed was not enabled."

table_detail = (
    spark.sql(f"DESCRIBE DETAIL {FULL_CHUNK_TABLE_NAME}")
    .select("format", "numFiles", "sizeInBytes")
    .first()
)

display(
    spark.createDataFrame(
        [
            ("table_format", table_detail["format"], "PASS"),
            ("change_data_feed", cdf_property["value"], "PASS"),
            ("source_row_count", str(chunk_count), "PASS"),
        ],
        ["check", "observed_value", "status"],
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create the Standard AI Search endpoint
# MAGIC
# MAGIC In the Databricks workspace:
# MAGIC
# MAGIC 1. Open **Compute** from the left sidebar.
# MAGIC 2. Open the **AI Search** tab and select **Create endpoint**.
# MAGIC 3. Enter `car_manual_rag_endpoint`.
# MAGIC 4. Select endpoint type **Standard** and confirm.
# MAGIC 5. Wait until the endpoint is ready before creating the index.
# MAGIC
# MAGIC The endpoint is serving compute. It can host indexes, but it does not
# MAGIC itself contain the chunk records.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create the Delta Sync index
# MAGIC
# MAGIC In **Catalog Explorer**, open
# MAGIC `workspace.car_manual_rag.document_chunks`, select **Create**, and then
# MAGIC **Vector search index**. Use this configuration:
# MAGIC
# MAGIC | Setting | Value |
# MAGIC |---|---|
# MAGIC | Name | `workspace.car_manual_rag.car_manual_chunks_index` |
# MAGIC | Index type | **Hybrid** |
# MAGIC | Primary key | `chunk_id` |
# MAGIC | Embedding source | **Compute embeddings** |
# MAGIC | Text column | `chunk_text` |
# MAGIC | Embedding model | Keep the available Databricks-managed default |
# MAGIC | Vector Search endpoint | `car_manual_rag_endpoint` |
# MAGIC | Sync mode | **Triggered** |
# MAGIC
# MAGIC Under **Advanced settings**, index these metadata columns in addition to
# MAGIC the primary key and embedding source:
# MAGIC
# MAGIC ```text
# MAGIC document_id, manufacturer, model, model_year, language, chunk_index,
# MAGIC document_title, section_title, pdf_page_numbers, character_count,
# MAGIC chunking_strategy
# MAGIC ```
# MAGIC
# MAGIC Leave **Save computed embeddings** off for this baseline. Create the
# MAGIC index, open its Overview page, and click **Sync now** if the first
# MAGIC triggered sync has not started automatically. Wait for the index and
# MAGIC data ingest status to become ready.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why this configuration?
# MAGIC
# MAGIC - **Delta Sync** keeps the index connected to our governed Delta source.
# MAGIC - **Managed embeddings** avoid a separate batch embedding pipeline while
# MAGIC   still teaching the embedding model's role.
# MAGIC - **Hybrid** allows later comparison of semantic similarity with combined
# MAGIC   semantic and keyword retrieval.
# MAGIC - **Triggered** gives us an explicit synchronization checkpoint and is
# MAGIC   sufficient for a learning corpus that changes infrequently.
# MAGIC - **Metadata columns** let us constrain retrieval to the selected vehicle
# MAGIC   and return page citations with each result.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Knowledge check
# MAGIC
# MAGIC Why does this project use a Delta Sync index rather than Direct Vector
# MAGIC Access?
# MAGIC
# MAGIC A. Delta Sync supports only keyword search, which is enough for manuals.
# MAGIC
# MAGIC B. Direct Vector Access automatically tracks every Delta table change.
# MAGIC
# MAGIC C. Delta Sync incrementally synchronizes the governed source table, while
# MAGIC Direct Vector Access requires the application to manage vector writes and
# MAGIC is unavailable in Databricks Free Edition.
# MAGIC
# MAGIC D. Direct Vector Access cannot store metadata fields.