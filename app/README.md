# Databricks App

The deployed application provides two authenticated interfaces over the same
RAG chain:

- `GET /` — browser chat interface.
- `POST /api/ask` — JSON API accepting `vehicle` and `question`.
- `GET /health` — deployment health check.

Example request body:

```json
{
  "vehicle": "peugeot",
  "question": "O que fazer se a luz de temperatura do motor acender?"
}
```

The Databricks App binds these least-privilege resources:

- `vector-search-index`: `SELECT` on the AI Search index.
- `model`: `CAN_QUERY` on the Foundation Model API endpoint.
- `experiment`: `CAN_EDIT` on the MLflow experiment.

Authentication uses the service principal created automatically for the App.
No user token or secret is stored in this repository.
