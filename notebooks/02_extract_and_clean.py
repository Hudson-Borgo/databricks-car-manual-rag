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
from pyspark.sql import functions as F

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
# MAGIC ## 7. Parser decision
# MAGIC
# MAGIC The representative-page comparison supports the following decision:
# MAGIC
# MAGIC - Use Databricks `ai_parse_document` version 2.0 as the primary parser.
# MAGIC - Disable figure descriptions during the initial text-retrieval pipeline.
# MAGIC - Preserve the raw `VARIANT` output before applying cleaning rules.
# MAGIC - Keep PyPDF2 as a low-cost diagnostic and fallback parser.
# MAGIC
# MAGIC The managed parser recovered useful structure that PyPDF2 flattened,
# MAGIC including section headers, page headers, page footers, page numbers, and
# MAGIC figures. It also improved spacing and reading order in the Peugeot sample.
# MAGIC Hyundai line-break hyphenation remains and will be handled explicitly in
# MAGIC the cleaning stage.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Incrementally parse the complete manuals
# MAGIC
# MAGIC Both manuals are below the managed parser's 500-page limit. We compare file
# MAGIC size and modification timestamp with the existing parse table, so an
# MAGIC unchanged PDF is not parsed again when this notebook is rerun.
# MAGIC
# MAGIC Figure descriptions remain disabled to control cost. The unmodified parser
# MAGIC output is stored as `VARIANT` in a managed Delta table.

# COMMAND ----------

PARSE_TABLE = "document_parses"
ELEMENT_TABLE = "document_elements_raw"
PARSE_STAGING_TABLE = "_document_parse_updates_staging"

FULL_PARSE_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{PARSE_TABLE}"
FULL_ELEMENT_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{ELEMENT_TABLE}"
FULL_PARSE_STAGING_TABLE_NAME = (
    f"{CATALOG}.{SCHEMA}.{PARSE_STAGING_TABLE}"
)

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`.`{PARSE_TABLE}` (
      document_id STRING,
      file_path STRING,
      file_size BIGINT,
      file_modification_time TIMESTAMP,
      parser_name STRING,
      parser_version STRING,
      parsed_document VARIANT,
      parsed_at TIMESTAMP
    )
    USING DELTA
    COMMENT 'Raw ai_parse_document output for governed car manual PDFs'
    """
)

parse_inputs = (
    binary_files
    .withColumn(
        "file_path",
        F.regexp_replace(F.col("path"), r"^(dbfs:|file:)", ""),
    )
    .select(
        "file_path",
        F.col("length").alias("file_size"),
        F.col("modificationTime").alias("file_modification_time"),
        "content",
    )
    .join(
        source_documents.select("document_id", "file_path"),
        on="file_path",
        how="inner",
    )
)

existing_parses = spark.table(FULL_PARSE_TABLE_NAME).select(
    "document_id",
    "file_size",
    "file_modification_time",
)

parse_candidates = (
    parse_inputs.alias("source")
    .join(
        existing_parses.alias("target"),
        on=(
            (F.col("source.document_id") == F.col("target.document_id"))
            & (F.col("source.file_size") == F.col("target.file_size"))
            & (
                F.col("source.file_modification_time")
                == F.col("target.file_modification_time")
            )
        ),
        how="left_anti",
    )
    .select("source.*")
)

parse_candidate_count = parse_candidates.count()
print(f"Documents requiring full parsing: {parse_candidate_count}")

# COMMAND ----------

if parse_candidate_count > 0:
    if managed_parser_execution_status != "PASS":
        raise RuntimeError(
            "The representative managed-parser test did not pass. "
            "Do not start the full parse until that issue is resolved."
        )

    parse_candidates.createOrReplaceTempView("document_parse_candidates")

    parsed_updates = spark.sql(
        """
        SELECT
          document_id,
          file_path,
          file_size,
          file_modification_time,
          'ai_parse_document' AS parser_name,
          '2.0' AS parser_version,
          ai_parse_document(
            content,
            map(
              'version', '2.0',
              'descriptionElementTypes', ''
            )
          ) AS parsed_document,
          current_timestamp() AS parsed_at
        FROM document_parse_candidates
        """
    )

    # Materialize model-backed output once before MERGE. This avoids repeating an
    # expensive parser call if the merge plan evaluates its source more than once.
    (
        parsed_updates.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(FULL_PARSE_STAGING_TABLE_NAME)
    )

    spark.sql(
        f"""
        MERGE INTO `{CATALOG}`.`{SCHEMA}`.`{PARSE_TABLE}` AS target
        USING `{CATALOG}`.`{SCHEMA}`.`{PARSE_STAGING_TABLE}` AS source
          ON target.document_id = source.document_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )

    spark.sql(
        f"DROP TABLE IF EXISTS "
        f"`{CATALOG}`.`{SCHEMA}`.`{PARSE_STAGING_TABLE}`"
    )
else:
    print("SKIPPED: Both PDFs already have current managed-parser results.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Validate raw parse results
# MAGIC
# MAGIC We retain the complete parser response for reproducibility. The summary
# MAGIC below extracts only operational fields from the `VARIANT` value.

# COMMAND ----------

parse_results = spark.table(FULL_PARSE_TABLE_NAME)

parse_validation = parse_results.select(
    "document_id",
    "parser_name",
    "parser_version",
    F.size(
        F.variant_get(
            "parsed_document",
            "$.document.pages",
            "array<variant>",
        )
    ).alias("parsed_page_count"),
    F.size(
        F.variant_get(
            "parsed_document",
            "$.document.elements",
            "array<variant>",
        )
    ).alias("element_count"),
    F.coalesce(
        F.size(
            F.variant_get(
                "parsed_document",
                "$.error_status",
                "array<variant>",
            )
        ),
        F.lit(0),
    ).alias("error_count"),
    "parsed_at",
)

display(parse_validation.orderBy("document_id"))

assert parse_validation.count() == 2, "Expected two persisted parse results."
assert parse_validation.filter("error_count > 0").count() == 0, (
    "One or more documents contain managed-parser page errors."
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Flatten raw document elements
# MAGIC
# MAGIC The raw parser result remains the source of truth. This derived Delta table
# MAGIC makes individual elements queryable for cleaning, exploratory analysis,
# MAGIC and future chunk construction.
# MAGIC
# MAGIC `pdf_page_numbers` are one-based physical PDF positions derived from
# MAGIC bounding boxes. Printed page labels such as `138` remain separate
# MAGIC `page_number` elements and must not be confused with physical PDF pages.

# COMMAND ----------

element_variants = parse_results.select(
    "document_id",
    "file_path",
    F.explode(
        F.variant_get(
            "parsed_document",
            "$.document.elements",
            "array<variant>",
        )
    ).alias("element"),
)

bbox_variants = F.variant_get(
    "element",
    "$.bbox",
    "array<variant>",
)

page_ids = F.array_sort(
    F.array_distinct(
        F.transform(
            bbox_variants,
            lambda bbox: F.variant_get(bbox, "$.page_id", "int"),
        )
    )
)

document_elements_raw = element_variants.select(
    "document_id",
    "file_path",
    F.variant_get("element", "$.id", "int").alias("element_id"),
    F.variant_get("element", "$.type", "string").alias("element_type"),
    F.variant_get("element", "$.content", "string").alias("content"),
    F.variant_get("element", "$.confidence", "double").alias("confidence"),
    page_ids.alias("page_ids"),
    F.transform(page_ids, lambda page_id: page_id + F.lit(1)).alias(
        "pdf_page_numbers"
    ),
)

(
    document_elements_raw.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FULL_ELEMENT_TABLE_NAME)
)

display(
    spark.table(FULL_ELEMENT_TABLE_NAME)
    .groupBy("document_id", "element_type")
    .count()
    .orderBy("document_id", "element_type")
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 11. Audit cleaning requirements across the corpus
# MAGIC
# MAGIC Cleaning must be evidence-driven. The audit below measures missing text,
# MAGIC line-break hyphenation, HTML entities, and element length across the full
# MAGIC corpus. Null figure content is expected because figure descriptions were
# MAGIC disabled; null text or section headings require investigation.

# COMMAND ----------

raw_elements = spark.table(FULL_ELEMENT_TABLE_NAME)

empty_content = F.col("content").isNull() | (F.trim(F.col("content")) == "")
line_break_hyphenation = F.col("content").rlike(
    r"-\s*\n\s*\p{L}"
)
html_entity = F.col("content").rlike(
    r"&#x[0-9A-Fa-f]+;|&#[0-9]+;|&[A-Za-z]+;"
)

cleaning_audit = (
    raw_elements
    .groupBy("document_id", "element_type")
    .agg(
        F.count("*").alias("element_count"),
        F.sum(F.when(empty_content, 1).otherwise(0)).alias(
            "empty_content_count"
        ),
        F.sum(F.when(line_break_hyphenation, 1).otherwise(0)).alias(
            "hyphenated_line_break_count"
        ),
        F.sum(F.when(html_entity, 1).otherwise(0)).alias(
            "html_entity_count"
        ),
        F.round(F.avg(F.length("content")), 1).alias(
            "average_character_count"
        ),
        F.expr("percentile_approx(length(content), 0.5)").alias(
            "median_character_count"
        ),
        F.max(F.length("content")).alias("maximum_character_count"),
    )
    .orderBy("document_id", "element_type")
)

display(cleaning_audit)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 12. Find repeated page furniture
# MAGIC
# MAGIC Page headers, footers, and printed page numbers are valuable citation
# MAGIC metadata but usually poor semantic-search content. Repeated normalized text
# MAGIC is surfaced here for review; nothing is deleted from the raw table.

# COMMAND ----------

page_furniture_types = ["page_header", "page_footer", "page_number"]

repeated_page_furniture = (
    raw_elements
    .filter(F.col("element_type").isin(page_furniture_types))
    .filter(~empty_content)
    .withColumn(
        "normalized_content",
        F.lower(
            F.trim(
                F.regexp_replace(F.col("content"), r"\s+", " ")
            )
        ),
    )
    .groupBy("document_id", "element_type", "normalized_content")
    .agg(
        F.count("*").alias("occurrences"),
        F.array_sort(
            F.array_distinct(F.flatten(F.collect_list("pdf_page_numbers")))
        ).alias("pdf_page_numbers"),
    )
    .filter(F.col("occurrences") > 1)
    .orderBy(
        "document_id",
        F.desc("occurrences"),
        "element_type",
    )
)

display(repeated_page_furniture)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 13. Inspect elements containing known artifacts
# MAGIC
# MAGIC These examples will become test fixtures for deterministic cleaning. We
# MAGIC will compare raw and cleaned values before writing a cleaned Delta table.

# COMMAND ----------

artifact_examples = (
    raw_elements
    .filter(line_break_hyphenation | html_entity)
    .select(
        "document_id",
        "element_id",
        "element_type",
        "pdf_page_numbers",
        "content",
    )
    .orderBy("document_id", "element_id")
)

display(artifact_examples.limit(50))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 14. Apply deterministic cleaning rules
# MAGIC
# MAGIC The cleaning implementation lives in `src/car_manual_rag/cleaning.py` so
# MAGIC it can be unit tested and reused outside this notebook. Databricks Git
# MAGIC folders add the repository root to Python's import path; this cell locates
# MAGIC the project's `src` directory without assuming a serverless working
# MAGIC directory.
# MAGIC
# MAGIC Every raw element remains represented in the cleaned table. Content roles
# MAGIC and `retrieval_eligible` determine downstream behavior without deleting
# MAGIC provenance.

# COMMAND ----------

import sys
from pathlib import Path

repository_roots = [
    Path(path)
    for path in sys.path
    if path and (Path(path) / "pyproject.toml").exists()
]

assert repository_roots, (
    "Could not locate the Git folder root on sys.path. "
    "Confirm that this notebook is running from the project Git folder."
)

source_directory = repository_roots[0] / "src"
if str(source_directory) not in sys.path:
    sys.path.insert(0, str(source_directory))

from car_manual_rag.cleaning import (  # noqa: E402
    CLEANING_VERSION,
    classify_content_role,
    clean_extracted_text,
    is_text_retrieval_eligible,
)

assert clean_extracted_text("sele-\ncionar") == "selecionar"
assert clean_extracted_text("MENU/\nVeículo") == "MENU/Veículo"
assert clean_extracted_text("Vista&#x20;Zoom") == "Vista Zoom"

print(f"Loaded deterministic cleaning rules: {CLEANING_VERSION}")

# COMMAND ----------

CLEAN_ELEMENT_TABLE = "document_elements_clean"
FULL_CLEAN_ELEMENT_TABLE_NAME = (
    f"{CATALOG}.{SCHEMA}.{CLEAN_ELEMENT_TABLE}"
)

clean_text_udf = F.udf(clean_extracted_text, "string")
content_role_udf = F.udf(classify_content_role, "string")
retrieval_eligible_udf = F.udf(is_text_retrieval_eligible, "boolean")

document_elements_clean = (
    raw_elements
    .withColumn("cleaned_content", clean_text_udf("content"))
    .withColumn("content_role", content_role_udf("element_type"))
    .withColumn(
        "retrieval_eligible",
        retrieval_eligible_udf("element_type", "cleaned_content"),
    )
    .withColumn("cleaning_version", F.lit(CLEANING_VERSION))
    .withColumn("cleaned_at", F.current_timestamp())
)

(
    document_elements_clean.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FULL_CLEAN_ELEMENT_TABLE_NAME)
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 15. Validate cleaned elements
# MAGIC
# MAGIC The clean table must preserve row count, eliminate the measured text
# MAGIC artifacts, and never mark empty content as retrieval eligible.

# COMMAND ----------

clean_elements = spark.table(FULL_CLEAN_ELEMENT_TABLE_NAME)

raw_element_count = raw_elements.count()
clean_element_count = clean_elements.count()
eligible_element_count = clean_elements.filter("retrieval_eligible").count()

remaining_hyphenation = clean_elements.filter(
    F.col("cleaned_content").rlike(r"-\s*\n\s*\p{L}")
).count()
remaining_html_entities = clean_elements.filter(
    F.col("cleaned_content").rlike(
        r"&#x[0-9A-Fa-f]+;|&#[0-9]+;|&[A-Za-z]+;"
    )
).count()
empty_eligible_elements = clean_elements.filter(
    F.col("retrieval_eligible")
    & (
        F.col("cleaned_content").isNull()
        | (F.trim("cleaned_content") == "")
    )
).count()

assert clean_element_count == raw_element_count, (
    "Cleaning must not add or remove raw parser elements."
)
assert remaining_hyphenation == 0, "Hyphenated line breaks remain after cleaning."
assert remaining_html_entities == 0, "HTML entities remain after cleaning."
assert empty_eligible_elements == 0, (
    "Empty content must never be eligible for text retrieval."
)

cleaning_validation = spark.createDataFrame(
    [
        ("raw_element_count", raw_element_count, "PASS"),
        ("clean_element_count", clean_element_count, "PASS"),
        ("retrieval_eligible_count", eligible_element_count, "PASS"),
        ("remaining_hyphenation", remaining_hyphenation, "PASS"),
        ("remaining_html_entities", remaining_html_entities, "PASS"),
        ("empty_eligible_elements", empty_eligible_elements, "PASS"),
    ],
    ["check", "observed_value", "status"],
)

display(cleaning_validation)

# COMMAND ----------

display(
    clean_elements
    .groupBy("document_id", "content_role", "retrieval_eligible")
    .count()
    .orderBy("document_id", "content_role", "retrieval_eligible")
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 16. Compare raw and cleaned artifact examples
# MAGIC
# MAGIC Review every changed hyphenation or HTML-entity example before accepting
# MAGIC the cleaning version. A deterministic transformation can still encode a
# MAGIC bad assumption, especially for legitimate Portuguese compound words.

# COMMAND ----------

changed_examples = (
    clean_elements.alias("clean")
    .join(
        artifact_examples.select("document_id", "element_id").alias("artifact"),
        on=["document_id", "element_id"],
        how="inner",
    )
    .select(
        "document_id",
        "element_id",
        "element_type",
        "pdf_page_numbers",
        F.col("content").alias("raw_content"),
        "cleaned_content",
        "retrieval_eligible",
    )
    .orderBy("document_id", "element_id")
)

display(changed_examples)

# COMMAND ----------
# MAGIC %md
# MAGIC ## How we will choose
# MAGIC
# MAGIC - **PyPDF2 baseline:** inexpensive and transparent; suitable when embedded
# MAGIC   text and reading order are good.
# MAGIC - **Databricks `ai_parse_document`:** selected primary parser because the
# MAGIC   experiment retained useful document structure and improved reading order.
# MAGIC - **OCR:** required when important pages contain images of text without a
# MAGIC   usable text layer.
# MAGIC
# MAGIC Raw managed-parser results and flattened elements are now persisted. Text
# MAGIC cleaning and chunk construction remain separate, testable stages.

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

