# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Section-aware document chunking
# MAGIC
# MAGIC This notebook converts cleaned parser elements into retrieval units. It
# MAGIC compares two chunk-size configurations, persists one practical baseline,
# MAGIC and preserves vehicle, section, element, and PDF-page metadata.
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC - Explain the trade-off between small and large chunks.
# MAGIC - Preserve semantic section context and source citations.
# MAGIC - Apply bounded overlap without crossing section boundaries.
# MAGIC - Split oversized elements such as long tables.
# MAGIC - Write retrieval-ready chunks to a Delta table in Unity Catalog.
# MAGIC
# MAGIC **Exam alignment:** Data Preparation, Application Development, and
# MAGIC Assembling and Deploying Applications.

# COMMAND ----------

import statistics
import sys
from pathlib import Path

from pyspark.sql import functions as F

CATALOG = "workspace"
SCHEMA = "car_manual_rag"
SOURCE_TABLE = "source_documents"
CLEAN_ELEMENT_TABLE = "document_elements_clean"
CHUNK_TABLE = "document_chunks"

FULL_SOURCE_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{SOURCE_TABLE}"
FULL_CLEAN_ELEMENT_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{CLEAN_ELEMENT_TABLE}"
FULL_CHUNK_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{CHUNK_TABLE}"

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Import the tested chunking module

# COMMAND ----------

repository_roots = [
    Path(path)
    for path in sys.path
    if path and (Path(path) / "pyproject.toml").exists()
]

assert repository_roots, "Could not locate the project Git folder root."

source_directory = repository_roots[0] / "src"
if str(source_directory) not in sys.path:
    sys.path.insert(0, str(source_directory))

from car_manual_rag.chunking import (  # noqa: E402
    CHUNKING_STRATEGY,
    build_section_aware_chunks,
)

print(f"Loaded chunking strategy: {CHUNKING_STRATEGY}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Load governed inputs
# MAGIC
# MAGIC Only elements marked `retrieval_eligible` contribute text. Structural
# MAGIC elements become context, body elements become chunk content, and page
# MAGIC furniture plus visual elements remain available in their upstream tables.

# COMMAND ----------

source_documents = spark.table(FULL_SOURCE_TABLE_NAME)
eligible_elements = (
    spark.table(FULL_CLEAN_ELEMENT_TABLE_NAME)
    .filter("retrieval_eligible")
    .select(
        "document_id",
        "element_id",
        "element_type",
        "cleaned_content",
        "pdf_page_numbers",
    )
)

display(
    eligible_elements
    .groupBy("document_id", "element_type")
    .count()
    .orderBy("document_id", "element_type")
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Compare two chunk-size configurations
# MAGIC
# MAGIC This is a compact engineering experiment rather than a claim that one size
# MAGIC is universally optimal:
# MAGIC
# MAGIC - **1,000 characters:** more precise units and more index records.
# MAGIC - **1,600 characters:** more surrounding context and fewer index records.
# MAGIC
# MAGIC Both use up to 200 characters of element-boundary overlap. Character-based
# MAGIC token estimates are approximate; the selected embedding model's tokenizer
# MAGIC and context limit remain the final constraint.

# COMMAND ----------

document_metadata = {
    row.document_id: row.asDict()
    for row in source_documents.collect()
}

elements_by_document = {}
for document_id in document_metadata:
    elements_by_document[document_id] = [
        row.asDict()
        for row in (
            eligible_elements
            .filter(F.col("document_id") == document_id)
            .orderBy("element_id")
            .collect()
        )
    ]

comparison_rows = []
chunks_by_configuration = {}

for max_chars in (1000, 1600):
    configuration_chunks = []
    for document_id, elements in elements_by_document.items():
        configuration_chunks.extend(
            {
                "document_id": document_id,
                **chunk,
            }
            for chunk in build_section_aware_chunks(
                document_id,
                elements,
                max_chars=max_chars,
                overlap_chars=200,
            )
        )

    chunks_by_configuration[max_chars] = configuration_chunks
    character_counts = [
        int(chunk["character_count"])
        for chunk in configuration_chunks
    ]
    token_estimates = [
        int(chunk["estimated_token_count"])
        for chunk in configuration_chunks
    ]
    sorted_character_counts = sorted(character_counts)
    p95_index = min(
        len(sorted_character_counts) - 1,
        int(len(sorted_character_counts) * 0.95),
    )

    comparison_rows.append(
        (
            max_chars,
            200,
            len(configuration_chunks),
            round(statistics.mean(character_counts), 1),
            sorted_character_counts[p95_index],
            max(character_counts),
            round(statistics.mean(token_estimates), 1),
        )
    )

chunk_comparison = spark.createDataFrame(
    comparison_rows,
    """
    max_characters INT,
    overlap_characters INT,
    chunk_count INT,
    average_characters DOUBLE,
    p95_characters INT,
    maximum_characters INT,
    average_estimated_tokens DOUBLE
    """,
)

display(chunk_comparison.orderBy("max_characters"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Persist the 1,600-character baseline
# MAGIC
# MAGIC We select 1,600 characters as a pragmatic starting point for these manuals.
# MAGIC It retains more procedural context while remaining comfortably below
# MAGIC typical embedding-model limits. Retrieval evaluation can later justify a
# MAGIC smaller or larger value.

# COMMAND ----------

SELECTED_MAX_CHARS = 1600
selected_chunks = chunks_by_configuration[SELECTED_MAX_CHARS]

chunk_rows = []
for chunk in selected_chunks:
    metadata = document_metadata[chunk["document_id"]]
    chunk_rows.append(
        (
            chunk["chunk_id"],
            chunk["document_id"],
            metadata["manufacturer"],
            metadata["model"],
            metadata["model_year"],
            metadata["language"],
            chunk["chunk_index"],
            chunk["document_title"],
            chunk["section_title"],
            chunk["start_element_id"],
            chunk["end_element_id"],
            chunk["element_ids"],
            chunk["pdf_page_numbers"],
            chunk["chunk_text"],
            chunk["character_count"],
            chunk["estimated_token_count"],
            chunk["chunking_strategy"],
        )
    )

document_chunks = (
    spark.createDataFrame(
        chunk_rows,
        """
        chunk_id STRING,
        document_id STRING,
        manufacturer STRING,
        model STRING,
        model_year INT,
        language STRING,
        chunk_index INT,
        document_title STRING,
        section_title STRING,
        start_element_id INT,
        end_element_id INT,
        element_ids ARRAY<INT>,
        pdf_page_numbers ARRAY<INT>,
        chunk_text STRING,
        character_count INT,
        estimated_token_count INT,
        chunking_strategy STRING
        """,
    )
    .withColumn("chunked_at", F.current_timestamp())
)

(
    document_chunks.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FULL_CHUNK_TABLE_NAME)
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Validate retrieval-ready chunks

# COMMAND ----------

persisted_chunks = spark.table(FULL_CHUNK_TABLE_NAME)

chunk_count = persisted_chunks.count()
unique_chunk_count = persisted_chunks.select("chunk_id").distinct().count()
empty_chunk_count = persisted_chunks.filter(
    F.col("chunk_text").isNull() | (F.trim("chunk_text") == "")
).count()
oversized_chunk_count = persisted_chunks.filter(
    F.col("character_count") > SELECTED_MAX_CHARS
).count()
missing_page_metadata_count = persisted_chunks.filter(
    F.size("pdf_page_numbers") == 0
).count()

assert chunk_count == unique_chunk_count, "Chunk IDs must be unique."
assert empty_chunk_count == 0, "Chunks must contain text."
assert oversized_chunk_count == 0, "Chunks exceed the selected size."
assert missing_page_metadata_count == 0, "Chunks must retain PDF page metadata."

chunk_validation = spark.createDataFrame(
    [
        ("chunk_count", chunk_count, "PASS"),
        ("unique_chunk_count", unique_chunk_count, "PASS"),
        ("empty_chunk_count", empty_chunk_count, "PASS"),
        ("oversized_chunk_count", oversized_chunk_count, "PASS"),
        ("missing_page_metadata_count", missing_page_metadata_count, "PASS"),
    ],
    ["check", "observed_value", "status"],
)

display(chunk_validation)

# COMMAND ----------

display(
    persisted_chunks
    .groupBy("manufacturer", "model", "model_year")
    .agg(
        F.count("*").alias("chunk_count"),
        F.round(F.avg("character_count"), 1).alias("average_characters"),
        F.max("character_count").alias("maximum_characters"),
        F.round(F.avg("estimated_token_count"), 1).alias(
            "average_estimated_tokens"
        ),
    )
    .orderBy("manufacturer", "model")
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Inspect chunks related to temperature warnings

# COMMAND ----------

display(
    persisted_chunks
    .filter(F.lower("chunk_text").contains("temperatura"))
    .select(
        "manufacturer",
        "model",
        "section_title",
        "pdf_page_numbers",
        "character_count",
        "chunk_text",
    )
    .orderBy("manufacturer", "model", "chunk_index")
    .limit(20)
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Knowledge check
# MAGIC
# MAGIC What is the most important reason to evaluate chunk size with retrieval
# MAGIC results instead of always choosing the smallest possible chunks?
# MAGIC
# MAGIC A. Smaller chunks always require a larger embedding model.
# MAGIC
# MAGIC B. Small chunks can improve precision but lose context needed to answer the
# MAGIC question correctly.
# MAGIC
# MAGIC C. AI Search cannot index chunks below 1,000 characters.
# MAGIC
# MAGIC D. Unity Catalog requires every chunk to span multiple pages.
