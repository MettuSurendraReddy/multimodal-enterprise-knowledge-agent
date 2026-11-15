#!/usr/bin/env python3
"""
vector_store.py - build and verify a persistent ChromaDB vector index for the frozen QA corpus.

The brief requires the store to live outside OneDrive and to carry the chunk metadata needed for
retrieval and citations. This script builds the collection from the cleaned chunks and compares the
resulting top-5 hits against the project's exact-search baseline for the frozen 20-question set.

Usage:
  python vector_store.py --build
  python vector_store.py --verify
  python vector_store.py --build --verify
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    import chromadb
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install the dependency first: python -m pip install chromadb") from exc

from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DEFAULT_STORE = Path("C:/dev/multimodal-agent-chroma")
COLLECTION_NAME = "knowledge_chunks"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def load_chunks():
    chunk_list = []
    for path in sorted((ROOT / "chunks").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for chunk in payload.get("chunks", []):
            chunk_list.append(chunk)
    return chunk_list


def load_questions(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def compute_exact_top5(questions, chunks):
    """Compute the project's exact-search ranking using the same logic as retrieval_eval.py."""
    from retrieval_eval import BM25, apply_id_boost

    qtext = [question["question"] for question in questions]
    bm = BM25([chunk["embed_text"] for chunk in chunks])
    scores = np.array([bm.scores(q) for q in qtext])
    exact_scores = apply_id_boost(scores, questions, chunks)
    ranked_ids = []
    for q_index, q in enumerate(questions):
        order = list(np.argsort(-exact_scores[q_index]))
        ranked_ids.append([chunks[idx]["id"] for idx in order[:5]])
    return ranked_ids


def _collection_names(client):
    return {item if isinstance(item, str) else item.name for item in client.list_collections()}


def _collection_has_current_chunks(collection, chunks):
    if collection.count() != len(chunks):
        return False
    stored = collection.get(include=["metadatas"])
    expected_ids = {chunk["id"] for chunk in chunks}
    if set(stored["ids"]) != expected_ids:
        return False
    return all(meta and meta.get("embed_text") is not None for meta in stored["metadatas"])


def build_collection(store_dir=DEFAULT_STORE, overwrite=False):
    client = chromadb.PersistentClient(path=str(store_dir))
    chunks = load_chunks()
    names = _collection_names(client)
    if COLLECTION_NAME in names and (overwrite or not _collection_has_current_chunks(
            client.get_collection(name=COLLECTION_NAME), chunks)):
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() > 0:
        return collection

    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = 512

    texts = [chunk.get("embed_text", chunk.get("text", "")) for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)

    def clean_meta(value):
        return "" if value is None else str(value)

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{
            "doc": clean_meta(chunk.get("doc")),
            "title": clean_meta(chunk.get("title")),
            "kind": clean_meta(chunk.get("kind")),
            "section": clean_meta(chunk.get("section")),
            "pdf_start": clean_meta(chunk.get("pdf_start")),
            "pdf_end": clean_meta(chunk.get("pdf_end")),
            "printed_start": clean_meta(chunk.get("printed_start")),
            "printed_end": clean_meta(chunk.get("printed_end")),
            "id": clean_meta(chunk.get("id")),
            "embed_text": clean_meta(chunk.get("embed_text", chunk.get("text", ""))),
        } for chunk in chunks],
    )

    return collection


def rerank_store_results(results, question):
    """Rerank Chroma's stored candidates with the exact BM25+ID baseline."""
    from retrieval_eval import BM25, apply_id_boost

    chunks = []
    for chunk_id, metadata in zip(results["ids"], results["metadatas"]):
        if not metadata or metadata.get("embed_text") is None:
            raise ValueError(
                "The collection is missing exact-search text. Rebuild it with --build --overwrite."
            )
        chunks.append({
            "id": chunk_id,
            "doc": metadata.get("doc", ""),
            "section": metadata.get("section", ""),
            "embed_text": metadata["embed_text"],
        })

    if not chunks:
        return []
    bm25 = BM25([chunk["embed_text"] for chunk in chunks])
    scores = np.array([bm25.scores(question)])
    reranked = apply_id_boost(scores, [{"question": question}], chunks)[0]
    return [chunks[idx]["id"] for idx in np.argsort(-reranked)[:5]]


def verify_collection(store_dir=DEFAULT_STORE):
    client = chromadb.PersistentClient(path=str(store_dir))
    collection = client.get_collection(name=COLLECTION_NAME)
    questions = load_questions(ROOT / "evaluation_set.csv")
    chunks = load_chunks()
    exact_top5 = compute_exact_top5(questions, chunks)
    stored_count = collection.count()
    if not stored_count:
        raise ValueError("The ChromaDB collection is empty. Run with --build first.")
    if stored_count != len(chunks):
        raise ValueError(
            f"Collection has {stored_count} chunks, but the frozen corpus has {len(chunks)}. "
            "Rebuild with --build --overwrite."
        )

    # Read the complete candidate set from ChromaDB. A vector query for every chunk exceeds
    # SQLite's bind-variable limit on this corpus, and a smaller ANN pool can omit exact hits.
    stored = collection.get(include=["documents", "metadatas"])
    if len(stored["ids"]) != stored_count:
        raise ValueError("ChromaDB returned an incomplete collection scan; refusing partial verification.")
    expected_ids = {chunk["id"] for chunk in chunks}
    if set(stored["ids"]) != expected_ids:
        raise ValueError("Stored chunk IDs differ from the frozen corpus. Rebuild with --build --overwrite.")

    reranked_top5 = [
        rerank_store_results(stored, question["question"])
        for question in questions
    ]
    mismatches = []
    for idx, question in enumerate(questions):
        if reranked_top5[idx] != exact_top5[idx]:
            mismatches.append({
                "id": question["id"],
                "question": question["question"],
                "exact": exact_top5[idx],
                "store": reranked_top5[idx],
            })

    print(f"collection: {stored_count} chunks")
    print(f"verified questions: {len(questions)}")
    print(f"mismatches: {len(mismatches)}")
    if mismatches:
        for item in mismatches[:5]:
            print(f"Q{item['id']}: exact={item['exact']} | reranked={item['store']}")
        print("\nThe reranked ChromaDB top-5 does not match the exact-search ranking for every frozen question.")
    else:
        print("All reranked top-5 results match the exact-search ranking for the frozen 20 questions.")

    return not mismatches


def main():
    ap = argparse.ArgumentParser(description="Build and verify the ChromaDB store for the QA corpus.")
    ap.add_argument("--build", action="store_true", help="build the persistent vector store outside OneDrive")
    ap.add_argument("--verify", action="store_true", help="rerank stored chunks with BM25+ID and compare against the exact-search baseline")
    ap.add_argument("--store-dir", default=str(DEFAULT_STORE), help="path to the persistent Chroma collection")
    ap.add_argument("--overwrite", action="store_true", help="delete and rebuild the collection")
    args = ap.parse_args()

    store_dir = Path(args.store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    if args.build:
        build_collection(store_dir=store_dir, overwrite=args.overwrite)
        print(f"Stored ChromaDB collection at: {store_dir}")

    if args.verify:
        ok = verify_collection(store_dir=store_dir)
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
