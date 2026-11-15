#!/usr/bin/env python3
"""
retrieval_eval.py - retrieval-only test of the frozen 20 questions.

For each question it ranks all chunks (chunks/*.json) and checks WHERE the pages that your answer
key requires (column required_retrieval_pages) show up. It does not generate answers.

A chunk counts for a required page when it comes from that document and its PDF page range
contains the page. For a question tagged 'figure-only' the chunk must also be a figure caption,
because only the caption can answer it. A question with several required pages (the two-document
questions) is satisfied only when ALL of them appear; its rank is the rank at which the last one
appeared. Trap questions with no required page are not scored: their top 5 is printed so the
'lure' passages can be checked by eye.

A second, rougher measure, 'evidence', checks whether the top 5 chunks together contain the
must_include items of the answer key (words, not exact phrases). It is informational.

Models (the first run needs no download):
  bm25        word-matching baseline, no model, instant
  bge-small   BAAI/bge-small-en-v1.5          (about 130 MB)
  gemma       google/embeddinggemma-300m      (about 1.2 GB, gated: accept its terms on Hugging Face)
  qwen        Qwen/Qwen3-Embedding-0.6B       (about 1.2 GB)
Dense models need:  python -m pip install sentence-transformers numpy

Usage (from the project folder):
  python retrieval_eval.py --model bm25
  python retrieval_eval.py --model bge-small
  python retrieval_eval.py --model bge-small --hybrid       dense ranking fused with word matching
  python retrieval_eval.py --model bge-small --hybrid --id-boost   ... and entries named by ID first
  python retrieval_eval.py --model bge-small --show 12      print the top 5 chunks for question 12
  python retrieval_eval.py --model bge-small --skip-reference   leave out the OWASP URL-list chunks

The evaluation file is never modified. Results go to results/retrieval_<model>.csv.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TOPN = 50

# Prefixes follow each model's card as far as I know them; check the card if a model does poorly.
# max_len 512 tokens is enough for our chunks (about 340 tokens at most); small batches keep memory low
# on an 8 GB laptop.
MODELS = {
    "bm25": dict(kind="lexical"),
    "bge-small": dict(
        id="BAAI/bge-small-en-v1.5", max_len=512, doc_prefix="",
        query_prefix="Represent this sentence for searching relevant passages: "),
    "gemma": dict(
        id="google/embeddinggemma-300m", max_len=512, batch=8, doc_prefix="title: none | text: ",
        query_prefix="task: search result | query: "),
    "qwen": dict(
        id="Qwen/Qwen3-Embedding-0.6B", max_len=512, batch=4, doc_prefix="",
        query_prefix="Instruct: Given a question, retrieve passages from technical standards and "
                     "regulations that answer it\nQuery:"),
}

STOP = set("""the and are for that with from has have its was were this these those into onto each
 such also not but all any can may will shall which who what how their there then than only both""".split())


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_chunks(skip_reference):
    chunks = []
    for path in sorted((ROOT / "chunks").glob("*.json")):
        for c in json.loads(path.read_text(encoding="utf-8"))["chunks"]:
            if skip_reference and c["meta"].get("reference"):
                continue
            chunks.append(c)
    return chunks


def load_questions(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_required(cell):
    """'rmf: PDF 27; gai: PDF 17' -> [('rmf', 27), ('gai', 17)]; 'none (...)' -> []"""
    cell = (cell or "").strip()
    if not cell or cell.lower().startswith("none"):
        return []
    pairs = []
    for part in cell.split(";"):
        m = re.match(r"\s*([a-z]+)\s*:\s*PDF\s*([\d,\s]+)", part)
        if m:
            pairs += [(m.group(1), int(n)) for n in re.findall(r"\d+", m.group(2))]
    return pairs


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
            if len(w) >= 3 and w not in STOP}


def parse_groups(cell):
    """'[risks: 4 of 5] a; b; c || [x] d' -> [('risks', ['a','b','c'], 4), ('x', ['d'], None)]"""
    cell = (cell or "").strip()
    if not cell.startswith("["):
        return []
    groups = []
    for g in cell.split(" || "):
        m = re.match(r"\[([^\]]+)\]\s*(.*)$", g.strip(), re.S)
        if not m:
            continue
        label, rest = m.group(1), m.group(2)
        mk = re.search(r":\s*(\d+)\s*of\s*(\d+)", label)
        items = [re.sub(r"\s*\([^)]*\)", "", x).strip() for x in re.split(r";\s*", rest) if x.strip()]
        groups.append((label.split(":")[0].strip(), items, int(mk.group(1)) if mk else None))
    return groups


# ---------------------------------------------------------------------------
# rankers
# ---------------------------------------------------------------------------
def tokens(text):
    return re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.tf = [Counter(tokens(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=float)
        self.avg = self.len.mean() or 1.0
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        out = np.zeros(len(self.tf))
        for w in set(tokens(query)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, t in enumerate(self.tf):
                f = t.get(w)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


def rrf(a, b, k=60):
    """Reciprocal rank fusion of two score matrices (questions x chunks): a chunk that ranks well in
    either list, and best in both, comes out on top. Exact IDs such as 'PO.2.1' or 'Article 4a'
    are found by word matching, meaning by the embedding model."""
    def ranks(m):
        order = np.argsort(-m, axis=1)
        r = np.empty_like(order)
        r[np.arange(m.shape[0])[:, None], order] = np.arange(1, m.shape[1] + 1)
        return r
    return 1.0 / (k + ranks(a)) + 1.0 / (k + ranks(b))


# IDs that name an entry directly. The patterns come from the structure of the ten documents (not from
# the questions): EU articles and annexes, AI 600-1 actions, SP 800-218A tasks, CSF subcategories,
# AI RMF subcategories, OWASP risks.
ID_PATTERNS = [
    r"\bArticle \d+[a-z]?\b", r"\bAnnex [IVXLC]+\b", r"\b(?:GV|MP|MS|MG)-\d+\.\d+-\d{3}\b",
    r"\b(?:PO|PS|PW|RV)\.\d+\.\d+\b", r"\b(?:GV|ID|PR|DE|RS|RC)\.[A-Z]{2}-\d{2}\b",
    r"\b(?:GOVERN|MAP|MEASURE|MANAGE) \d+\.\d+\b", r"\bLLM\d{2}\b",
]


def ids_in(text):
    found = []
    for rx in ID_PATTERNS:
        found += [m.group(0).lower() for m in re.finditer(rx, text, re.I)]
    return list(dict.fromkeys(found))


def names_entry(c, token):
    sec, cid = (c.get("section") or "").lower(), c["id"].lower()
    base = f"{c['doc']}:{token}"
    return (sec == token or sec.startswith(token + " ") or cid == base
            or cid.startswith(base + "#") or cid.startswith(base + "/"))


def apply_id_boost(scores, questions, chunks, per_id=2):
    """If a question names an entry by its ID, put the best `per_id` chunks of that entry first.
    The limit keeps a long entry (OWASP has six sections per risk) from crowding out the rest."""
    out = scores.copy()
    for qi, q in enumerate(questions):
        row = scores[qi]
        lift = row.max() - row.min() + 1.0
        for token in ids_in(q["question"]):
            hits = sorted((i for i, c in enumerate(chunks) if names_entry(c, token)), key=lambda i: -row[i])
            for i in hits[:per_id]:
                out[qi][i] = row[i] + lift
    return out


def dense_scores(key, cfg, chunks, queries):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("This model needs:  python -m pip install sentence-transformers numpy")
        sys.exit(2)
    texts = [cfg["doc_prefix"] + c["embed_text"] for c in chunks]
    sig = hashlib.sha256((cfg["id"] + cfg["doc_prefix"] + "\x00".join(c["id"] + c["embed_text"][:80] for c in chunks))
                         .encode("utf-8")).hexdigest()[:16]
    cache = ROOT / "embeddings" / f"{key}_{sig}.npy"
    model = SentenceTransformer(cfg["id"])
    model.max_seq_length = cfg["max_len"]
    if cache.exists():
        doc_emb = np.load(cache)
        print(f"loaded cached embeddings: {cache.name}")
    else:
        print(f"embedding {len(texts)} chunks with {cfg['id']} (a few minutes on a laptop CPU) ...")
        doc_emb = model.encode(texts, batch_size=cfg.get("batch", 16), normalize_embeddings=True,
                               show_progress_bar=True)
        cache.parent.mkdir(exist_ok=True)
        np.save(cache, doc_emb)
    q_emb = model.encode([cfg["query_prefix"] + q for q in queries], normalize_embeddings=True)
    return q_emb @ doc_emb.T


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def best_rank(order, chunks, doc, page, figure_only):
    for rank, idx in enumerate(order[:TOPN], 1):
        c = chunks[idx]
        if c["doc"] == doc and c["pdf_start"] <= page <= c["pdf_end"] and (not figure_only or c["kind"] == "figure"):
            return rank
    return None


def evidence(groups, top_text):
    if not groups:
        return None
    words = content_words(top_text)
    ok = 0
    for _label, items, need in groups:
        found = sum(1 for it in items if content_words(it) and content_words(it) <= words)
        if found >= (need if need is not None else len(items)):
            ok += 1
    return ok / len(groups)


def score_question(q, order, chunks):
    required = parse_required(q["required_retrieval_pages"])
    figure_only = (q.get("figure_tag") or "") == "figure-only"
    top5 = [chunks[i] for i in order[:5]]
    top_text = " ".join(c["text"] for c in top5)
    res = dict(id=q["id"], type=q["type"], tag=q.get("figure_tag") or "", required=required,
               top5=[f"{c['id']} (p{c['pdf_start']}-{c['pdf_end']})" for c in top5],
               evidence=evidence(parse_groups(q["must_include"]), top_text))
    if not required:
        res.update(rank=None, first=None, top5_hit="n/a")
        return res
    ranks = [best_rank(order, chunks, d, p, figure_only) for d, p in required]
    found = [r for r in ranks if r is not None]
    res["first"] = min(found) if found else None
    res["rank"] = max(ranks) if len(found) == len(ranks) else None
    res["top5_hit"] = "Y" if res["rank"] is not None and res["rank"] <= 5 else "N"
    return res


def summarize(results):
    scored = [r for r in results if r["top5_hit"] != "n/a"]

    def line(label, rs):
        if not rs:
            return
        n = len(rs)
        hit = lambda k: sum(1 for r in rs if r["rank"] is not None and r["rank"] <= k)
        mrr = sum(1 / r["rank"] for r in rs if r["rank"]) / n
        print(f"  {label:22s} n={n:2d}   top1 {hit(1):2d}   top3 {hit(3):2d}   top5 {hit(5):2d}   MRR {mrr:.2f}")

    print("\nSUMMARY (a question counts only when ALL its required pages are found)")
    line("all scored questions", scored)
    for t in sorted({r["type"] for r in scored}):
        line(f"type: {t}", [r for r in scored if r["type"] == t])
    for tag in sorted({r["tag"] for r in scored if r["tag"]}):
        line(f"tag: {tag}", [r for r in scored if r["tag"] == tag])


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Retrieval-only test of the frozen questions.")
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--eval", default="evaluation_set.csv", help="the frozen evaluation CSV")
    ap.add_argument("--hybrid", action="store_true",
                    help="fuse the dense ranking with BM25 word matching (reciprocal rank fusion)")
    ap.add_argument("--id-boost", action="store_true",
                    help="when a question names an entry by ID (Article 4a, PO.2.1, LLM08 ...), rank that entry first")
    ap.add_argument("--skip-reference", action="store_true", help="leave out OWASP URL-list chunks")
    ap.add_argument("--show", metavar="ID", help="print the top 5 chunks for one question id")
    args = ap.parse_args(argv)

    qpath = Path(args.eval) if Path(args.eval).is_absolute() else ROOT / args.eval
    if not qpath.exists():
        print(f"{qpath} not found. Save your evaluation CSV in the project folder under that name, "
              f"or pass --eval PATH.")
        return 2
    questions = load_questions(qpath)
    chunks = load_chunks(args.skip_reference)
    if not chunks:
        print("no chunks found. Run: python chunker.py all")
        return 2
    print(f"{len(questions)} questions, {len(chunks)} chunks, model {args.model}")

    cfg = MODELS[args.model]
    qtexts = [q["question"] for q in questions]
    if cfg.get("kind") == "lexical":
        bm = BM25([c["embed_text"] for c in chunks])
        scores = np.array([bm.scores(q) for q in qtexts])
    else:
        scores = dense_scores(args.model, cfg, chunks, qtexts)
        if args.hybrid:
            bm = BM25([c["embed_text"] for c in chunks])
            scores = rrf(scores, np.array([bm.scores(q) for q in qtexts]))

    if args.id_boost:
        scores = apply_id_boost(scores, questions, chunks)
    results = []
    for q, row in zip(questions, scores):
        order = list(np.argsort(-row))
        results.append(score_question(q, order, chunks))

    if args.show:
        r = next((x for x in results if x["id"] == args.show), None)
        if not r:
            print(f"no question with id {args.show}")
            return 1
        q = next(x for x in questions if x["id"] == args.show)
        print(f"\nQ{r['id']} [{r['type']}] {q['question']}\nrequired: {r['required']}   "
              f"rank {r['rank']}  first hit {r['first']}   IDs found in the question: {ids_in(q['question'])}")
        row = scores[int(questions.index(q))]
        for rank, idx in enumerate(np.argsort(-row)[:5], 1):
            c = chunks[idx]
            print(f"\n#{rank}  {c['id']}  [{c['kind']}]  PDF p{c['pdf_start']}-{c['pdf_end']}  score {row[idx]:.3f}")
            print("    " + " ".join(c["text"].split())[:260])
        return 0

    print(f"\n{'Q':>2s} {'type':9s} {'tag':13s} {'top5':5s} {'rank':>5s} {'first':>5s} {'evid':>5s}  required")
    for r in results:
        ev = "-" if r["evidence"] is None else f"{r['evidence']:.2f}"
        print(f"{r['id']:>2s} {r['type']:9s} {r['tag']:13s} {r['top5_hit']:5s} "
              f"{str(r['rank'] or '-'):>5s} {str(r['first'] or '-'):>5s} {ev:>5s}  "
              f"{'; '.join(f'{d}:{p}' for d, p in r['required']) or '(trap)'}")
    summarize(results)

    for r in results:
        if r["top5_hit"] == "n/a":
            print(f"\nQ{r['id']} (trap, not scored) top 5: " + "; ".join(r["top5"]))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    name = args.model + ("+bm25" if args.hybrid and cfg.get("kind") != "lexical" else "") \
        + ("+id" if args.id_boost else "") \
        + ("_noref" if args.skip_reference else "") \
        + ("" if qpath.stem == "evaluation_set" else "_" + qpath.stem)       # another question file: own results file
    with open(out / f"retrieval_{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "figure_tag", "retrieval_top5 (Y/N/n/a)", "gold_page_rank",
                    "first_hit_rank", "evidence", "top5_chunks"])
        for r in results:
            w.writerow([r["id"], r["type"], r["tag"], r["top5_hit"], r["rank"] or "", r["first"] or "",
                        "" if r["evidence"] is None else f"{r['evidence']:.2f}", " | ".join(r["top5"])])
    print(f"\nwrote results/retrieval_{name}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())