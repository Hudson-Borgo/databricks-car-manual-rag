# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Environment and Unity Catalog check
# MAGIC
# MAGIC This notebook establishes the governed namespace used by the car manual
# MAGIC RAG project before we create any files, tables, indexes, models, or
# MAGIC endpoints.
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC - Verify that the notebook is attached to working compute.
# MAGIC - Inspect the current Databricks identity, catalog, and schema.
# MAGIC - Confirm the `workspace.car_manual_rag` Unity Catalog namespace.
# MAGIC - Understand the `catalog.schema.object` naming model.
# MAGIC
# MAGIC **Exam alignment:** Data Preparation, Governance, and Assembling and
# MAGIC Deploying Applications.

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "car_manual_rag"
FULL_SCHEMA_NAME = f"{CATALOG}.{SCHEMA}"

print(f"Project namespace: {FULL_SCHEMA_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Inspect the current session
# MAGIC
# MAGIC `current_user()` shows the identity executing the notebook. The current
# MAGIC catalog and schema are session defaults; production code should still use
# MAGIC fully qualified Unity Catalog names when clarity and portability matter.

# COMMAND ----------

session_context = spark.sql(
    """
    SELECT
      current_user() AS current_user,
      current_catalog() AS current_catalog,
      current_schema() AS current_schema
    """
)

display(session_context)
print(f"Spark version: {spark.version}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Inspect available catalogs
# MAGIC
# MAGIC A catalog is the first level of the Unity Catalog three-level namespace.
# MAGIC It contains schemas, which contain objects such as tables, volumes,
# MAGIC functions, and models.

# COMMAND ----------

display(spark.sql("SHOW CATALOGS"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Create or confirm the project schema
# MAGIC
# MAGIC `IF NOT EXISTS` makes this operation idempotent: rerunning the notebook
# MAGIC does not fail when the schema already exists.

# COMMAND ----------

spark.sql(
    f"""
    CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`
    COMMENT 'Governed data and AI assets for the car manual RAG learning project'
    """
)

spark.sql(f"USE CATALOG `{CATALOG}`")
spark.sql(f"USE SCHEMA `{SCHEMA}`")

print(f"Using Unity Catalog schema: {FULL_SCHEMA_NAME}")

# COMMAND ----------

display(spark.sql(f"DESCRIBE SCHEMA EXTENDED `{CATALOG}`.`{SCHEMA}`"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Validate the namespace
# MAGIC
# MAGIC The validation below deliberately fails if the expected schema cannot be
# MAGIC found. A visible failure is safer than allowing later notebooks to create
# MAGIC assets in an unintended namespace.

# COMMAND ----------

matching_schemas = spark.sql(
    f"SHOW SCHEMAS IN `{CATALOG}` LIKE '{SCHEMA}'"
)

assert matching_schemas.count() == 1, (
    f"Expected exactly one schema named {FULL_SCHEMA_NAME}. "
    "Check the catalog name and your Unity Catalog privileges."
)

validation_result = spark.createDataFrame(
    [
        (
            "Unity Catalog schema",
            FULL_SCHEMA_NAME,
            "PASS",
        )
    ],
    ["check", "observed_value", "status"],
)

display(validation_result)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Unity Catalog naming model
# MAGIC
# MAGIC Our future objects will use names such as:
# MAGIC
# MAGIC ```text
# MAGIC workspace.car_manual_rag.source_documents
# MAGIC workspace.car_manual_rag.document_chunks
# MAGIC workspace.car_manual_rag.car_manual_index
# MAGIC ```
# MAGIC
# MAGIC The first two are planned Delta tables. The third is a planned AI Search
# MAGIC index. We have not created any of them yet.
# MAGIC
# MAGIC The PDF itself will use a volume path rather than a table identifier:
# MAGIC
# MAGIC ```text
# MAGIC /Volumes/workspace/car_manual_rag/manuals/<manual-file>.pdf
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ## Knowledge check
# MAGIC
# MAGIC Which name follows the Unity Catalog three-level namespace for the Delta
# MAGIC table that will store chunks?
# MAGIC
# MAGIC A. `car_manual_rag.document_chunks`
# MAGIC
# MAGIC B. `workspace.car_manual_rag.document_chunks`
# MAGIC
# MAGIC C. `/Volumes/workspace/car_manual_rag/document_chunks`
# MAGIC
# MAGIC D. `workspace.document_chunks.car_manual_rag`
# MAGIC
# MAGIC Record your answer before moving to the ingestion notebook.
