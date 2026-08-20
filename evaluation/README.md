# RAG Evaluation

This directory contains a lightweight framework to evaluate the accuracy and robustness of the Azure RAG pipeline.

## Files
- `questions.json`: A ground-truth dataset containing in-scope questions (with expected source documents and snippets) and out-of-scope questions (expecting refusal behavior).
- `evaluate.py`: A script that runs the evaluation against LIVE Azure resources.

## Running Evaluation
Since this script evaluates the end-to-end pipeline, it requires live Azure configuration in `.env`.

```bash
PYTHONPATH=.. ../.venv/bin/python evaluate.py
```
*(Or if running from root of repo: `PYTHONPATH=. .venv/bin/python evaluation/evaluate.py`)*

## Metrics
The evaluation script measures:
1. **Retrieval Success**: Did vector search return the expected source document?
2. **Grounded Answer**: Does the generated answer contain the correct factual snippet?
3. **Refusal Behavior**: Does the model correctly refuse to answer out-of-scope questions without hallucinating?
