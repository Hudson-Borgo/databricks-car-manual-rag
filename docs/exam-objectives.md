# Exam objective mapping

This checklist follows the six sections in the current Databricks Certified
Generative AI Engineer Associate exam guide.

## 1. Design Applications

- Define the assistant's inputs, outputs, grounding behavior, and citations.
- Select the model task and order the RAG components.
- Design a structured answer format.
- Compare the RAG design with agent-oriented alternatives.

## 2. Data Preparation

- Select a suitable source manual and record its quality and usage rights.
- Choose a PDF extraction package based on the document structure.
- Remove headers, footers, repeated text, and other retrieval noise.
- Compare fixed-size, recursive, and structure-aware chunking.
- Write extracted content and chunks to Delta tables in Unity Catalog.
- Measure retrieval quality and study reranking.

## 3. Application Development

- Select an embedding model based on context length and trade-offs.
- Build a retriever and augment the prompt with retrieved context.
- Iterate from a baseline prompt to a grounded prompt.
- Add quality and safety guardrails.
- Use MLflow tracing while developing the application.

## 4. Assembling and Deploying Applications

- Create and query a Mosaic AI Vector Search / AI Search index.
- Define dependencies, input examples, and the model signature.
- Register the application with MLflow and Unity Catalog where supported.
- Deploy with Model Serving where supported.
- Treat prompt and component changes as versioned code.

## 5. Governance

- Track source ownership, licensing, and document versions.
- Keep sensitive or inappropriate content out of prompts and responses.
- Use Unity Catalog objects rather than unmanaged project data.
- Test prompt-injection and unsupported-answer behavior.

## 6. Evaluation and Monitoring

- Build a representative evaluation set with expected evidence.
- Evaluate retrieval separately from answer generation.
- Use MLflow tracing and scorers.
- Compare quality, latency, and cost trade-offs.
- Incorporate human subject-matter feedback.

## Coverage note

Some exam objectives may not be directly executable in Databricks Free Edition.
When that happens, the repository will document the production design and the
platform limitation separately from the hands-on implementation.
