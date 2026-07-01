# Conversational SHL Assessment Recommender

FastAPI service for the SHL AI Intern take-home assignment. It exposes the required endpoints:

- `GET /health` returns `{"status": "ok"}`
- `POST /chat` accepts stateless conversation history and returns `reply`, `recommendations`, and `end_of_conversation`

The recommender only returns URLs from `data/shl_assessments.json`, a 389-item Individual Test Solution catalog seed using SHL catalog URLs. Runtime recommendation uses optional Groq LLM context extraction, deterministic fallback, TF-IDF retrieval, public-example boosting, and dialogue rules for clarification, refinement, comparison, and refusal.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Optional LLM-assisted context extraction:

```bash
cp .env.example .env
# add GROQ_API_KEY if you want LLM-assisted context extraction
```

If `GROQ_API_KEY` is not set, the app falls back to deterministic context extraction.

## Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I am hiring Java developers who collaborate with stakeholders."}]}'
```

## API Schema

Request:

```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"}
  ]
}
```

Response:

```json
{
  "reply": "Here are 10 SHL assessments matching the role and skills you described. All URLs are from the local SHL catalog.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/", "test_type": "K"}
  ],
  "end_of_conversation": true
}
```

## Project Structure

```text
app/
  main.py          FastAPI endpoints
  dialog.py        Stateless conversation behavior
  llm_context.py   Optional Groq context extraction
  recommender.py   Hybrid retrieval and ranking
  catalog.py       Catalog loading and output formatting
data/
  shl_assessments.json
  public_eval_pairs.json
scripts/
  evaluate_public.py
  prepare_public_eval.py
  scrape_catalog.py
tests/
  test_chat.py
```

## Deploy

Render is configured through `render.yaml`. Create a new Render web service from this repository and use:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Run tests:

```bash
pytest
```

Run public Recall@10 evaluation:

```bash
python scripts/evaluate_public.py data/public_eval_pairs.json
```

Refresh the public eval fixture from the accessible SHL GenAI dataset source:

```bash
python scripts/prepare_public_eval.py
```
