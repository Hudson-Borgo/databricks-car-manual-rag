# Architecture decision log

Record decisions as they are made so the project explains *why* each component
exists, not only how to run it.

| ID | Status | Decision | Reason |
|---|---|---|---|
| ADR-001 | Accepted | Use `workspace.car_manual_rag` as the initial Unity Catalog namespace. | It isolates the project from earlier labs. |
| ADR-002 | Accepted | Do not store the owner's manual PDF in Git. | Preserve licensing context and keep source data governed in Unity Catalog. |
| ADR-003 | Accepted | Store Databricks notebooks in source `.py` format. | Keep reviews readable and exclude notebook outputs from normal commits. |
| ADR-004 | Proposed | Use a Delta-backed AI Search index. | Direct Vector Access is not supported in Free Edition; verify the exact supported configuration in the workspace before implementation. |
| ADR-005 | Accepted | Store source manuals in the managed volume `workspace.car_manual_rag.manuals`. | The PDFs are non-tabular source data, and the project does not require direct access from systems outside Databricks. |
| ADR-006 | Accepted | Use Hyundai HB20 2022 and Peugeot 208 2023 manuals as the initial corpus. | Multiple vehicles let us practice metadata filtering and prevent cross-vehicle evidence leakage. |
| ADR-007 | Proposed | Compare a PyPDF2 embedded-text baseline with Databricks `ai_parse_document` before selecting the production parser. | Extraction quality depends on text layers, layout, tables, and images; the `.pdf` extension alone is insufficient evidence. |
