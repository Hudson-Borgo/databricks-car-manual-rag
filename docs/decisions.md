# Architecture decision log

Record decisions as they are made so the project explains *why* each component
exists, not only how to run it.

| ID | Status | Decision | Reason |
|---|---|---|---|
| ADR-001 | Accepted | Use `workspace.car_manual_rag` as the initial Unity Catalog namespace. | It isolates the project from earlier labs. |
| ADR-002 | Accepted | Do not store the owner's manual PDF in Git. | Preserve licensing context and keep source data governed in Unity Catalog. |
| ADR-003 | Accepted | Store Databricks notebooks in source `.py` format. | Keep reviews readable and exclude notebook outputs from normal commits. |
| ADR-004 | Accepted | Use a Delta Sync AI Search index on a Standard endpoint. | It incrementally synchronizes the governed Delta source; Direct Vector Access is not supported in Free Edition. |
| ADR-005 | Accepted | Store source manuals in the managed volume `workspace.car_manual_rag.manuals`. | The PDFs are non-tabular source data, and the project does not require direct access from systems outside Databricks. |
| ADR-006 | Accepted | Use Hyundai HB20 2022 and Peugeot 208 2023 manuals as the initial corpus. | Multiple vehicles let us practice metadata filtering and prevent cross-vehicle evidence leakage. |
| ADR-007 | Accepted | Use Databricks `ai_parse_document` 2.0 as the primary parser with PyPDF2 as a diagnostic fallback. | The sample comparison preserved section headers, page furniture, figures, and printed page labels while improving Peugeot spacing and reading order. |
| ADR-008 | Accepted | Preserve every parsed element in `document_elements_clean` and use content roles plus a retrieval flag instead of deleting non-body elements. | Page furniture and figures are useful provenance but would add noise to the initial text-only retriever; titles and section headers remain available as chunk context. |
| ADR-009 | Accepted | Start with section-aware chunks capped at 1,600 characters and up to 200 characters of element-boundary overlap. | This provides a practical context/precision baseline while retaining document, section, element, and PDF-page metadata for evaluation and citations. |
| ADR-010 | Accepted | Start with Databricks-managed embeddings, hybrid retrieval, and triggered sync. | This covers semantic and keyword retrieval while keeping synchronization explicit and avoiding an unnecessary custom embedding pipeline for the learning baseline. |
| ADR-011 | Accepted | Use strict `document_id` filtering, a grounded prompt, Qwen3-Next 80B A3B Instruct, and MLflow tracing for the baseline RAG chain. | This prevents cross-vehicle evidence leakage and makes retrieval and generation behavior inspectable before formal evaluation. |
| ADR-012 | Accepted | Evaluate structured RAG traces with a versioned Unity Catalog evaluation dataset, built-in MLflow judges, and project-specific guidelines. | Trace-based evaluation separates retrieval, grounding, relevance, safety, citation, and actionability failures while preserving rationales for iteration. |
| ADR-013 | Accepted | Complete the learning baseline with an operational validation notebook rather than a custom Model Serving endpoint. | The notebook-based chain already exercises AI Search, a Foundation Model API, tracing, and evaluation; this keeps the Free Edition project small while documenting the production deployment boundary. |
