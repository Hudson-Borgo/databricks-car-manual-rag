# Databricks Car Manual RAG

An end-to-end Retrieval-Augmented Generation (RAG) learning project built on
Databricks Free Edition using two car owner's manuals as the knowledge sources.

The project has two equally important outcomes:

1. Build a working, evidence-grounded multi-vehicle manual assistant.
2. Practice the objectives in the current Databricks Certified Generative AI
   Engineer Associate exam guide.

## Target question

> For a 2023 Peugeot 208, what should I do if the engine temperature warning
> light comes on?

The finished application should retrieve passages from the correct vehicle
manual, generate an answer grounded in those passages, cite the source document
and page, and avoid inventing an answer when the selected manual does not contain
enough evidence.

## Databricks architecture

The implementation will use current Databricks concepts and names:

- **Unity Catalog** for governed data objects.
- A **Unity Catalog volume** for the source PDFs.
- **Delta tables** for extracted pages, cleaned text, and chunks.
- **Mosaic AI Vector Search / AI Search** for semantic retrieval.
- A hosted embedding model and foundation model available in the workspace.
- **MLflow** for tracing, evaluation, and lifecycle management.
- **Model Serving** for deployment when supported by the Free Edition workspace.
- A Databricks **Git folder** connected to this GitHub repository.

The initial Unity Catalog namespace is:

```text
workspace.car_manual_rag
```

## Learning workflow

Every stage follows the same cycle:

1. Explain the concept and its exam relevance.
2. Implement one small, inspectable step.
3. Validate the result in Databricks.
4. Discuss alternatives and trade-offs.
5. Answer a short certification-style check.

We do not advance until the current checkpoint is confirmed.

## Project stages

| Stage | Databricks work | Exam focus |
|---|---|---|
| 0. Environment | Verify Free Edition capabilities and Unity Catalog objects | Platform selection and constraints |
| 1. Ingestion | Upload both PDFs and register document metadata | Source quality and governance |
| 2. Preparation | Extract, clean, chunk, and write Delta tables | Data Preparation |
| 3. Retrieval | Create and test an AI Search index | Retrieval, embeddings, and reranking |
| 4. RAG chain | Build prompts, retrieval flow, citations, and guardrails | Application Development |
| 5. Evaluation | Create an evaluation set and use MLflow tracing/scorers | Evaluation and Monitoring |
| 6. Deployment | Register and serve the application where supported | Assembling and Deploying Applications |
| 7. Operations | Inspect inference data, cost, safety, and feedback | Monitoring and Governance |

## Repository layout

```text
docs/                   Architecture, decisions, and exam mapping
notebooks/              Numbered Databricks source notebooks
src/car_manual_rag/     Reusable Python modules added as the project develops
tests/                  Local component tests
data/                   Documentation only; source PDFs are ignored by Git
```

## Source document policy

Do not commit the car manual PDFs to this repository. Keep them in a governed
Unity Catalog volume and record each document's manufacturer, model, model year,
language, edition, source URL, and usage rights. This policy is part of the
governance learning objective, not merely a repository-size decision.

## Official references

- [Generative AI Engineer Associate certification](https://www.databricks.com/learn/certification/genai-engineer-associate)
- [Databricks Free Edition limitations](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)
- [Create and manage Git folders](https://docs.databricks.com/aws/en/repos/git-operations-with-repos/)
