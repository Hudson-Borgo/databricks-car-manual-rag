# Architecture

## Planned data flow

```text
Owner's manual PDF
        |
        v
Unity Catalog volume
        |
        v
PDF extraction and cleaning
        |
        v
Delta tables: pages -> cleaned text -> chunks
        |
        v
Databricks AI Search Delta Sync index
        |
        v
Retriever -> grounded prompt -> foundation model
        |
        v
Answer + page citations + refusal when evidence is insufficient
        |
        v
MLflow tracing, evaluation, feedback, and monitoring
```

## Unity Catalog namespace

The project starts with the following namespace:

```text
Catalog: workspace
Schema:  car_manual_rag
```

Current retrieval objects:

```text
Chunk source: workspace.car_manual_rag.document_chunks
Endpoint:     car_manual_rag_endpoint
Index:        workspace.car_manual_rag.car_manual_chunks_index
```

The index uses Databricks-managed embeddings, hybrid retrieval, and triggered
Delta synchronization. Keeping these decisions explicit helps us understand
object ownership and avoid hidden setup code.

## Free Edition design constraint

The design must be validated against the current Free Edition documentation at
each platform-dependent step. We will not assume that a feature shown in a
full-workspace tutorial is enabled in this workspace.
