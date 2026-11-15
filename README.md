# AI Governance RAG Portfolio Project

This project builds a grounded retrieval-and-generation pipeline over a curated set of public AI governance and security documents. It is designed as a portfolio-grade demo for evaluating whether an LLM can answer governance questions using cited evidence instead of unsupported memory.

## What this project does

- Cleans and chunks a fixed set of public AI governance/security PDFs
- Builds a retrieval baseline using BM25 + ID-boost logic
- Verifies a persistent ChromaDB vector store against the exact ranking logic
- Generates grounded answers with citations
- Runs a frozen 20-question benchmark and reports correctness/faithfulness
- Provides a small local demo app for interactive use

## Project status

The benchmark run is complete and the weak-case fixes were re-checked against the frozen questions.

Current benchmark summary:
- Retrieval pass: 18/19
- Correctness: 20/20
- Faithfulness: 20/20

## Repository structure

- `evaluation_set.csv` — frozen benchmark questions used for validation
- `retrieval_eval.py` — retrieval benchmark and exact-ranking logic
- `vector_store.py` — persistent ChromaDB build and parity verification
- `answer_generation.py` — grounded answer generation with strict JSON output
- `full_eval.py` — first-pass evaluation runner
- `demo_app.py` — small local browser UI for asking questions
- `results/` — generated benchmark outputs and answer files

## Local setup

1. Open a terminal in this project folder.
2. Create a virtual environment if needed.
3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Set any required API keys for the model providers used by the generation stage:

```bash
set GROQ_API_KEY=...
set GEMINI_API_KEY=...
```

On Linux/macOS, use `export` instead of `set`.

## Run the local demo

```bash
python demo_app.py
```

Then open:

```text
http://localhost:8000/
```

## Run the full evaluation

```bash
python full_eval.py
```

This reads the frozen question set and writes the benchmark summary to `results/full_eval.csv`.

## Notes

- The benchmark is intentionally frozen and treated as a unit test.
- The project avoids paid services or private-key handling inside chat transcripts.
- The retrieval logic in `retrieval_eval.py` remains the canonical source of truth for ranking behavior.

## Portfolio angle

This project is strong for a portfolio because it shows:
- retrieval over real-world governance documents
- evidence-grounded generation with citations
- benchmark-driven iteration instead of free-form prompting
- careful handling of weak cases and explicit evaluation rules

See [CASE_STUDY.md](CASE_STUDY.md) for a concise portfolio summary, results, and evaluation caveats.
