# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — PDF extraction experiment
# MAGIC
# MAGIC Before building a production extraction pipeline, this notebook profiles
# MAGIC both manuals with a reproducible baseline. We measure whether the PDFs
# MAGIC contain embedded text, inspect representative pages, and check whether the
# MAGIC Databricks managed document parser is visible in this workspace.
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC - Choose an extraction approach from document evidence, not file extension.
# MAGIC - Distinguish embedded-text extraction from OCR.
# MAGIC - Compare a Python-library baseline with Databricks Document Parsing.
# MAGIC - Detect empty, image-only, or extraction-error pages.
# MAGIC - Preserve document and page identity for future citations.
# MAGIC
# MAGIC **Exam alignment:** Data Preparation and Application Development.

# COMMAND ----------
# MAGIC %pip install PyPDF2==3.0.1

# COMMAND ----------

import io
import json
import statistics
from collections import Counter

from PyPDF2 import PdfReader

CATALOG = "workspace"
SCHEMA = "car_manual_rag"
VOLUME = "manuals"
SOURCE_TABLE = "source_documents"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
FULL_SOURCE_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{SOURCE_TABLE}"

print(f"Source manifest: {FULL_SOURCE_TABLE_NAME}")
print(f"PDF volume:      {VOLUME_PATH}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Confirm governed inputs
# MAGIC
# MAGIC The Delta manifest supplies stable document identity and vehicle metadata.
# MAGIC The volume supplies the PDF bytes. We expect exactly two registered files.

# COMMAND ----------

source_documents = spark.table(FULL_SOURCE_TABLE_NAME)
display(source_documents.orderBy("manufacturer", "model"))

assert source_documents.count() == 2, (
    "Expected two source-document records. Rerun 01_ingest_pdf first."
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Read PDFs as binary files
# MAGIC
# MAGIC Spark's `binaryFile` data source returns one row per file, including the
# MAGIC file path, modification time, length, and binary content. The two manuals
# MAGIC total less than 30 MB, so collecting them for this controlled experiment
# MAGIC is acceptable. A larger production corpus would use distributed parsing.

# COMMAND ----------

binary_files = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.pdf")
    .load(VOLUME_PATH)
)

display(binary_files.select("path", "length", "modificationTime"))

assert binary_files.count() == 2, (
    "Expected two PDFs in the managed volume. Rerun 01_ingest_pdf first."
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Profile embedded-text extraction with PyPDF2
# MAGIC
# MAGIC PyPDF2 reads the PDF text layer. It does not perform optical character
# MAGIC recognition (OCR). Therefore, empty results can indicate scanned pages,
# MAGIC image-only diagrams, intentionally blank pages, or extraction problems.
# MAGIC
# MAGIC We retain one-based page numbers because those are meaningful to users and
# MAGIC will later become citation metadata.

# COMMAND ----------

def canonical_volume_path(path: str) -> str:
    """Normalize Spark paths to the canonical /Volumes form."""
    for prefix in ("dbfs:", "file:"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


metadata_by_path = {
    row.file_path: row.asDict()
    for row in source_documents.collect()
}

profile_rows = []
sample_rows = []

for binary_file in binary_files.collect():
    file_path = canonical_volume_path(binary_file.path)
    metadata = metadata_by_path.get(file_path)

    assert metadata is not None, (
        f"No source-document metadata found for {file_path}. "
        "Check path normalization and rerun 01_ingest_pdf."
    )

    reader = PdfReader(io.BytesIO(binary_file.content))
    page_count = len(reader.pages)
    encrypted = bool(reader.is_encrypted)
    page_character_counts = []
    extraction_errors = 0

    representative_pages = {
        1,
        max(1, page_count // 2),
        page_count,
    }

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            raw_text = ""
            extraction_errors += 1
            page_character_counts.append(0)
            if page_number in representative_pages:
                sample_rows.append(
                    (
                        metadata["document_id"],
                        metadata["manufacturer"],
                        metadata["model"],
                        page_number,
                        0,
                        f"EXTRACTION ERROR: {type(exc).__name__}: {exc}",
                    )
                )
            continue

        character_count = len(raw_text.strip())
        page_character_counts.append(character_count)

        if page_number in representative_pages:
            sample_rows.append(
                (
                    metadata["document_id"],
                    metadata["manufacturer"],
                    metadata["model"],
                    page_number,
                    character_count,
                    raw_text.strip()[:2000],
                )
            )

    pages_with_text = sum(count > 0 for count in page_character_counts)
    empty_pages = page_count - pages_with_text
    text_coverage_pct = round(100 * pages_with_text / page_count, 2)
    total_characters = sum(page_character_counts)
    median_characters = int(statistics.median(page_character_counts or [0]))

    if extraction_errors:
        recommendation = "REVIEW_EXTRACTION_ERRORS"
    elif text_coverage_pct < 80:
        recommendation = "EVALUATE_OCR_OR_LAYOUT_PARSER"
    else:
        recommendation = "EMBEDDED_TEXT_BASELINE_IS_VIABLE"

    profile_rows.append(
        (
            metadata["document_id"],
            metadata["manufacturer"],
            metadata["model"],
            metadata["model_year"],
            page_count,
            pages_with_text,
            empty_pages,
            text_coverage_pct,
            total_characters,
            median_characters,
            extraction_errors,
            encrypted,
            recommendation,
        )
    )

profile_columns = [
    "document_id",
    "manufacturer",
    "model",
    "model_year",
    "page_count",
    "pages_with_text",
    "empty_pages",
    "text_coverage_pct",
    "total_characters",
    "median_characters_per_page",
    "extraction_errors",
    "encrypted",
    "recommendation",
]

profile_df = spark.createDataFrame(profile_rows, profile_columns)
display(profile_df.orderBy("manufacturer", "model"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Inspect representative pages
# MAGIC
# MAGIC Metrics can reveal missing text but cannot prove that reading order,
# MAGIC headings, tables, warnings, or multi-column layouts were preserved. Inspect
# MAGIC the previews below for broken words, scrambled ordering, repeated headers,
# MAGIC page numbers, and useful section boundaries.

# COMMAND ----------

sample_columns = [
    "document_id",
    "manufacturer",
    "model",
    "page_number",
    "character_count",
    "text_preview",
]

samples_df = spark.createDataFrame(sample_rows, sample_columns)
display(samples_df.orderBy("manufacturer", "model", "page_number"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Check the managed parser candidate
# MAGIC
# MAGIC `ai_parse_document` can preserve structured elements such as paragraphs,
# MAGIC tables, figures, and layout. Function visibility is only a capability
# MAGIC check; it does not prove that the account has sufficient entitlement,
# MAGIC quota, or serverless environment configuration to execute it.

# COMMAND ----------

try:
    managed_parser_functions = spark.sql(
        "SHOW FUNCTIONS LIKE 'ai_parse_document'"
    )
    managed_parser_visible = managed_parser_functions.count() > 0
except Exception as exc:
    managed_parser_visible = False
    print(f"Managed parser capability check failed: {type(exc).__name__}: {exc}")

print(f"ai_parse_document visible: {managed_parser_visible}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Run a controlled managed-parser comparison
# MAGIC
# MAGIC To limit cost and latency, this experiment parses only the same first,
# MAGIC middle, and final pages inspected in the PyPDF2 baseline. Figure
# MAGIC descriptions are disabled because they are not needed for this text and
# MAGIC layout comparison.
# MAGIC
# MAGIC `pageRange` uses one-based page numbers, matching our citation convention.

# COMMAND ----------

MANAGED_PARSER_SAMPLES = {
    "hyundai_hb20_2022_pt_br": "1,247,494",
    "peugeot_208_2023_pt_br": "1,142,284",
}

managed_summary_rows = []
managed_element_rows = []
managed_parser_execution_status = "NOT_RUN"

if managed_parser_visible:
    try:
        for document in source_documents.collect():
            page_range = MANAGED_PARSER_SAMPLES[document.document_id]
            escaped_path = document.file_path.replace("'", "''")

            parsed_row = spark.sql(
                f"""
                SELECT
                  path,
                  CAST(
                    ai_parse_document(
                      content,
                      map(
                        'version', '2.0',
                        'descriptionElementTypes', '',
                        'pageRange', '{page_range}'
                      )
                    ) AS STRING
                  ) AS parsed_json
                FROM read_files(
                  '{escaped_path}',
                  format => 'binaryFile'
                )
                """
            ).first()

            parsed_payload = json.loads(parsed_row.parsed_json)
            elements = parsed_payload.get("document", {}).get("elements", [])
            errors = parsed_payload.get("error_status", []) or []
            element_types = Counter(
                element.get("type", "unknown")
                for element in elements
            )

            managed_summary_rows.append(
                (
                    document.document_id,
                    page_range,
                    len(elements),
                    len(errors),
                    json.dumps(dict(sorted(element_types.items()))),
                )
            )

            for element in elements:
                page_numbers = sorted(
                    {
                        bbox.get("page_id") + 1
                        for bbox in (element.get("bbox") or [])
                        if bbox.get("page_id") is not None
                    }
                )

                managed_element_rows.append(
                    (
                        document.document_id,
                        page_range,
                        element.get("id"),
                        ",".join(str(page) for page in page_numbers),
                        element.get("type"),
                        element.get("confidence"),
                        (element.get("content") or "")[:2000],
                    )
                )

        managed_parser_execution_status = "PASS"
    except Exception as exc:
        managed_parser_execution_status = "UNAVAILABLE_OR_FAILED"
        print(
            "Managed parser execution failed without affecting the PyPDF2 "
            f"baseline: {type(exc).__name__}: {exc}"
        )
else:
    managed_parser_execution_status = "NOT_VISIBLE"

print(f"Managed parser execution status: {managed_parser_execution_status}")

# COMMAND ----------

managed_summary_df = spark.createDataFrame(
    managed_summary_rows,
    """
    document_id STRING,
    page_range STRING,
    element_count INT,
    error_count INT,
    element_type_counts STRING
    """,
)

display(managed_summary_df.orderBy("document_id"))

# COMMAND ----------

managed_elements_df = spark.createDataFrame(
    managed_element_rows,
    """
    document_id STRING,
    requested_page_range STRING,
    element_id INT,
    page_numbers STRING,
    element_type STRING,
    confidence DOUBLE,
    content_preview STRING
    """,
)

display(
    managed_elements_df
    .filter(
        "element_type IN ('title', 'section_header', 'text', 'table', "
        "'page_header', 'page_footer', 'page_number')"
    )
    .orderBy("document_id", "element_id")
    .limit(100)
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## How we will choose
# MAGIC
# MAGIC - **PyPDF2 baseline:** inexpensive and transparent; suitable when embedded
# MAGIC   text and reading order are good.
# MAGIC - **Databricks `ai_parse_document`:** managed parsing that can retain
# MAGIC   document structure; availability and cost must be validated.
# MAGIC - **OCR:** required when important pages contain images of text without a
# MAGIC   usable text layer.
# MAGIC
# MAGIC We will not persist extracted pages until we inspect these results.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Knowledge check
# MAGIC
# MAGIC A PDF has hundreds of pages, but the embedded-text extractor returns empty
# MAGIC strings for nearly every page. Which approach is the best next candidate?
# MAGIC
# MAGIC A. Increase the future chunk overlap.
# MAGIC
# MAGIC B. Use OCR or a document parser that supports image-based text.
# MAGIC
# MAGIC C. Create the vector index directly from the PDF bytes.
# MAGIC
# MAGIC D. Use a larger embedding dimension.
# MAGIC
# MAGIC Record your answer together with your quality observations.

