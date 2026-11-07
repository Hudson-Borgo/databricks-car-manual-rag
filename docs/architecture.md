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
Mosaic AI Vector Search / AI Search index
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

Exact volume, table, index, model, and endpoint names will be decided when the
corresponding component is implemented. Keeping those decisions explicit helps
us understand object ownership and avoid hidden setup code.

## Free Edition design constraint

The design must be validated against the current Free Edition documentation at
each platform-dependent step. We will not assume that a feature shown in a
full-workspace tutorial is enabled in this workspace.
