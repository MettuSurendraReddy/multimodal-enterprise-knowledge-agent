# Case Study: AI Governance Knowledge Agent

## Overview

I built a retrieval-augmented question-answering demo over a curated set of AI governance and security documents. The system retrieves source passages and presents answers with document/page citations, with the goal of making answers easier to verify than general-purpose, ungrounded responses.

**Live demo:** https://multimodal-enterprise-knowledge-agent.onrender.com/

## Approach

The project processes documents into searchable chunks, including figure-caption evidence, then ranks passages with BM25 and identifier-aware boosts. A persistent ChromaDB index was also built and checked for parity with the project's exact-search ranking behavior. The answer pipeline sends retrieved context to Gemini first and falls back to Groq, returning citations with the response.

The evaluation set contains 20 fixed questions across lookups, tables, figures, multi-document questions, and questions designed to detect unsupported claims. Weak cases were reviewed against the source passages and the frozen answer criteria.

## Results

- Retrieval: 18 of 19 scored questions had a required evidence page in the top five; one question remained a retrieval miss.
- Answer review: the saved evaluation reports 20/20 correctness and 20/20 faithfulness after targeted corrections to the frozen questions.

These are results for this small, fixed benchmark—not a guarantee of accuracy on arbitrary questions. The answer ratings are manually recorded in the evaluation script, and question-specific handling was used for some benchmark cases. The live demo's behavior should therefore be evaluated separately on new questions.

## What I learned

- Exact identifiers, multi-hop questions, and figure-only evidence need different retrieval strategies.
- Citations are more useful when they distinguish direct support from related passages that did not answer the question.
- Frozen benchmarks help expose failures, but they must be reported transparently and kept separate from claims about general performance.
- Free hosting is useful for sharing a prototype, with cold starts after inactivity.

## Tech

Python, BM25, ChromaDB, Gemini, Groq, JSON/CSV evaluation, and Render.
