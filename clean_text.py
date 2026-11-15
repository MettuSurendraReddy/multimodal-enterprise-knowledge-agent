#!/usr/bin/env python3
"""
clean_text.py - turn a PDF's raw text into clean paragraphs, one record per page.

What it does, and why:
  1. Reads each page as text BLOCKS (paragraph-like units), not raw lines, so joining
     lines is reliable.
  2. Normalises characters: ligatures (the single 'fi' character), invisible characters,
     non-breaking spaces. The search system must see 'first', not a ligature.
  3. Finds running headers and footers by REPETITION (the same text in the top or bottom
     band of many pages), removes them, and keeps the printed page number as metadata.
  4. Joins the lines of each block into paragraphs and repairs end-of-line hyphens:
         orga- / nizational  ->  organizational       GOV- / ERN     ->  GOVERN
         GV- / 1.1-001       ->  GV-1.1-001 (kept)    cross- / cutting -> cross-cutting
                                                       (only if the document itself
                                                        writes it with the hyphen)
  5. Logs everything it removes and every hyphen decision, so nothing disappears silently.

Usage (from the project folder):
  python clean_text.py rmf               clean the AI RMF
                                         (writes cleaned/rmf.json and cleaned/rmf_log.txt)
  python clean_text.py all               clean every configured document, print one summary table
  python clean_text.py rmf --show 27     show raw text vs cleaned paragraphs for PDF page 27
  python clean_text.py rmf --blocks 27   print where each text block sits on PDF page 27
  python clean_text.py --selftest        test the rules on small synthetic input (no PDFs needed)

The AI RMF (rmf) and the Generative AI Profile (gai) are reviewed. The other documents have
first-pass rules that are refined after reading each log. Other documents get their own entry in DOC_CONFIG
(their own noise lines and paragraph-start patterns) one at a time.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
# key -> (file name in the project folder, first 16 hex characters of its SHA-256)
FILES = {
    "rmf": ("NIST.AI.100-1.pdf", "7576edb531d98488"),
    "gai": ("NIST.AI.600-1.pdf", "6e73620ab6b64e90"),     # Generative AI Profile
    "aml": ("NIST.AI.100-2e2025.pdf", "4811fb6ad73f9c91"),            # adversarial ML taxonomy
    "csf": ("NIST.CSWP.29.pdf", "3c31f46fee98cac0"),                  # Cybersecurity Framework 2.0
    "deploy": ("Deploying AI Systems 240415.pdf", "f6358ac4af5d59f4"),  # Deploying AI Systems Securely
    "data": ("CSI_AI_DATA_SECURITY.pdf", "bce55417db5b25f7"),           # AI Data Security
    "ssdf": ("Secure Development for GenAI NIST.SP.800-218A.pdf", "e088c8bc75716824"),
    "eu": ("EU AI Act CELEX_02024R1689-20260727_EN_TXT.pdf", "1ccd38d1c78482cf"),
    "uk": ("AI_Playbook_for_the_UK_Government__12_02_.pdf", "3417156b104aa32e"),
    "owasp": ("OWASP-Top-10-for-LLMs-v2025.pdf", "eee39216f0e6750f"),
}

# Per-document rules.
#   band             fraction of the page height treated as header (top) and footer (bottom)
#   noise_lines      lines to drop wherever they appear (regular expressions, full line)
#   noise_blocks     whole blocks to drop (regular expression on the block text with its
#                    line breaks turned into single spaces)
#   keep_hyphen      compounds whose hyphen must survive a line break (from the log review)
#   drop_garbled     True: remove paragraphs that are mostly non-ASCII characters (an unreadable
#                    text layer), and log them
#   merge_continuations  True: join a paragraph that stops without sentence punctuation to
#                    a next paragraph that starts in lower case (list items split in two)
#   page_number_patterns  regular expressions (number in group 1) for a page number that sits
#                    inside a longer header or footer, such as "39 genai.owasp.org"
#   strip_line_prefix  regular expressions removed from the start of a line (for example a
#                    list bullet that the extractor turned into a letter)
#   join_hyphen      words that are really broken in the middle (reviewed), so the hyphen is
#                    dropped even where the default would keep it
#   unknown_hyphen   "auto" lets the document decide (it counts breaks inside words against
#                    real compounds). Otherwise, what to do with a line-end hyphen the
#                    document gives no evidence about:
#                    "drop" (default) for documents typeset with automatic hyphenation, where
#                    a break is usually inside a word; "keep" for documents that break lines
#                    only at real hyphens (compounds, URLs)
#   paragraph_starts lines that always begin a NEW paragraph (so table entries such as
#                    'GOVERN 1.1: ...' are never glued together)
DOC_CONFIG = {
    "rmf": dict(
        band=0.09,
        noise_lines=[r"^Continued on next page$"],
        noise_blocks=[r"^Categories Subcategories$"],      # table column headings
        # compounds that must keep their hyphen when a line breaks inside them. Found by
        # reviewing the hyphen log: each appears only once, split, so the document itself
        # gives no evidence for the hyphen.
        keep_hyphen=["context-relevant", "cost-effective", "off-label"],
        paragraph_starts=[r"^(?:GOVERN|MAP|MEASURE|MANAGE) \d+(?:\.\d+)?:"],
    ),
    # AI 600-1. Every action ID such as GV-1.1-001 starts a new paragraph. The document has no
    # running header, and the table header row repeats at every subcategory. It is typeset
    # without automatic hyphenation (the first log showed line-end hyphens only at real
    # compounds such as state-of-the-art and in URLs), so an UNKNOWN line-end hyphen is kept.
    "gai": dict(
        band=0.09,
        merge_continuations=True,      # action text wraps into a second block
        noise_lines=[],
        noise_blocks=[r"^Action ID Suggested Action GAI Risks$"],
        unknown_hyphen="keep",
        keep_hyphen=[],
        paragraph_starts=[r"^(?:GV|MP|MS|MG)-\d+\.\d+-\d{3}\b"],
    ),
    # ---- first passes for the other documents: generic detection, plus only what the earlier
    # ---- ID scans showed. Everything else is added after reading each log.
    # AI 100-2 uses both styles: real compounds and breaks inside words. The words below were
    # found in the hyphen review (the last five are names and a title in the references).
    "aml": dict(band=0.09, unknown_hyphen="auto", noise_lines=[r"^Back to Index$"],
                noise_blocks=[], paragraph_starts=[],
                join_hyphen=["combinations", "unperturbed", "robustbench", "elham",
                             "gerchick", "mansour", "macdiarmid", "deepsteal", "hebbar",
                             "tensorflow"]),
    "csf": dict(band=0.09, merge_continuations=True, unknown_hyphen="auto", noise_lines=[], noise_blocks=[],
                # the extractor turns the list bullet into a letter o before each ID
                strip_line_prefix=[r"^o\s+(?=(?:GV|ID|PR|DE|RS|RC)\.[A-Z]{2}-\d{2}:)"],
                paragraph_starts=[r"^(?:GV|ID|PR|DE|RS|RC)\.[A-Z]{2}-\d{2}:"]),
    "deploy": dict(band=0.09, merge_continuations=True, unknown_hyphen="auto", noise_lines=[], noise_blocks=[],
                   paragraph_starts=[]),
    "data": dict(band=0.09, merge_continuations=True, unknown_hyphen="auto", noise_lines=[], noise_blocks=[],
                 paragraph_starts=[]),
    "ssdf": dict(band=0.09, unknown_hyphen="auto", noise_lines=[], noise_blocks=[],
                 paragraph_starts=[r"^(?:PO|PS|PW|RV)\.\d+\.\d+\b"]),
    # the consolidated text carries amendment markers such as a triangle followed by B or M1
    "eu": dict(band=0.09, unknown_hyphen="auto", noise_blocks=[],
               noise_lines=[r"^[\u25bc\u25ba]\s*(?:B|[MCA]\d+)$", r"^[\u25c4\u25c0]$"],
               # the running header ends with the page number: "02024R1689 — EN — 27.07.2026 — 001.001 — 12"
               page_number_patterns=[r"^02024R1689\s*—.*—\s*(\d{1,4})$"],
               # list items arrive as two blocks (first line / rest) and some words are split
               # at a soft hyphen between blocks
               merge_continuations=True,
               paragraph_starts=[r"^Article \d+[a-z]?\s*$"]),
    "uk": dict(band=0.09, merge_continuations=True, unknown_hyphen="auto", noise_lines=[], noise_blocks=[],
               paragraph_starts=[]),
    # the contents pages repeat every risk with dot leaders; those lines are dropped
    "owasp": dict(band=0.09, unknown_hyphen="auto", noise_lines=[r".*(?:\.\s+){4,}.*"],
                  # the running header was removed on most pages but stayed as a paragraph on
                  # pages 12, 19, 22 and 24; the last two pages have an unreadable text layer
                  noise_blocks=[r"^OWASP Top 10 for LLM Applications v2\.0$"],
                  drop_garbled=True,
                  paragraph_starts=[r"^LLM(?:0[1-9]|10):"],
                  # the footer reads like "39 genai.owasp.org"
                  page_number_patterns=[r"^(\d{1,3}) genai\.owasp\.org$"]),
}

BULLET_START = r"^[\u2022\u25cf\u25aa\u25e6\u2023]\s*"
REPEAT_SHARE = 0.25          # a header/footer must repeat on at least this share of pages...
REPEAT_MIN_PAGES = 3         # ...and on at least this many pages
EDGE_BLOCKS = 3              # the 3 topmost and 3 bottommost blocks of a page are also
                             # header/footer candidates, whatever the band says
SOFT_HYPHEN = "\u00ad"
INVISIBLE = ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff")
PAGE_NUM_RE = re.compile(
    r"^(?:page\s*)?(\d{1,4}|(?=[ivxlcdm])m{0,3}(?:cm|cd|d?c{0,3})"
    r"(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3}))$",
    re.I,
)
TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
# A hyphen followed by one of these is not a line-break hyphen but a suspended hyphen, as in
# "high- or low-probability": the hyphen stays and a space follows it.
SUSPENDED_WORDS = {"or", "and", "nor", "to"}     # only when written in lower case
# ...and only when the word stands alone: "ready- / to-deploy" is a compound, not suspended
SUSPENDED_RE = re.compile(r"(?:" + "|".join(sorted(SUSPENDED_WORDS)) + r")(?![\w-])")
# the end of a line that is inside a URL: such a line is joined to the next WITHOUT a space
URL_TAIL_RE = re.compile(r"(?:https?://|www\.)\S*[/\-_=&?#]$")


# ---------------------------------------------------------------------------
# CHARACTERS
# ---------------------------------------------------------------------------
def normalize_chars(text):
    """NFKC turns ligatures into plain letters and non-breaking spaces into spaces.
    Soft hyphens are kept on purpose: at a line end they mark a hyphenation break, and
    join_block uses that. Any soft hyphen left over is removed when a paragraph is finished."""
    text = unicodedata.normalize("NFKC", text)
    for ch in INVISIBLE:
        text = text.replace(ch, "")
    return text


def finalize(paragraph):
    """Tidy a paragraph. A soft hyphen at its very end is kept for now: it says the word
    continues in the next block, and merge_split_paragraphs uses it."""
    trailing = paragraph.endswith(SOFT_HYPHEN)
    paragraph = paragraph.replace(SOFT_HYPHEN, "")
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    return paragraph + SOFT_HYPHEN if trailing else paragraph


# ---------------------------------------------------------------------------
# PAGE BLOCKS, HEADERS AND FOOTERS
# ---------------------------------------------------------------------------
def page_blocks(page):
    blocks = []
    for item in page.get_text("blocks"):
        x0, y0, x1, y1, text, _no, btype = item[:7]
        if btype != 0:               # 0 = text, 1 = image
            continue
        text = normalize_chars(text).strip("\n")
        if text.strip():
            blocks.append(dict(y0=y0, y1=y1, text=text))
    return blocks


def band_of(block, height, band):
    if block["y1"] <= band * height:
        return "header"
    if block["y0"] >= (1 - band) * height:
        return "footer"
    return None


def mark_bands(blocks, height, band):
    """Mark which blocks could be a running header or footer. A block is a candidate if it
    lies in the top/bottom band of the page OR is one of the first/last few blocks on the
    page. Being a candidate removes nothing by itself: a candidate is removed only if its
    text repeats on many pages or it is just a page number."""
    for b in blocks:
        b["band"] = band_of(b, height, band)
    order = sorted(range(len(blocks)), key=lambda i: blocks[i]["y0"])
    for i in order[:EDGE_BLOCKS]:
        if not blocks[i]["band"]:
            blocks[i]["band"] = "header"
    for i in order[-EDGE_BLOCKS:]:
        if not blocks[i]["band"]:
            blocks[i]["band"] = "footer"


def repeat_key(text):
    """Digits become '#', so 'Page 22' and 'Page 23' count as the same repeating text."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text.lower())).strip()


def page_number_from_patterns(text, patterns):
    """A page number inside a longer header or footer, e.g. "39 genai.owasp.org". The pattern
    must hold the number in its first group."""
    flat = re.sub(r"\s+", " ", text).strip()
    for rx in patterns:
        m = rx.match(flat)
        if m:
            return m.group(1)
    return None


def printed_page_of(text):
    m = PAGE_NUM_RE.match(re.sub(r"\s+", " ", text).strip())
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# HYPHEN REPAIR
# ---------------------------------------------------------------------------
def add_line_tokens(vocab, line):
    """Collect the document's own words. The last word of a line that ends in a hyphen is
    a fragment, so it is left out."""
    s = line.rstrip()
    trailing = s.endswith("-") or s.endswith(SOFT_HYPHEN)
    tokens = TOKEN_RE.findall(s.replace(SOFT_HYPHEN, ""))
    if trailing and tokens:
        tokens = tokens[:-1]
    vocab.update(t.lower() for t in tokens)


def build_vocab(pages_blocks, repeating):
    vocab = Counter()
    for _pno, blocks in pages_blocks:
        for b in blocks:
            if b["band"] and repeat_key(b["text"]) in repeating:
                continue
            for line in b["text"].split("\n"):
                add_line_tokens(vocab, line)
    return vocab


def _count(vocab, word):
    """How often the document writes this word (works for a Counter or a plain set)."""
    if hasattr(vocab, "get"):
        return vocab.get(word, 0)
    return 1 if word in vocab else 0


def decide(left, right, vocab, keep=(), default="drop", next_text="", join=()):
    """Decide what to do with a hyphen at a line end. Returns (action, kind).
    'drop' = remove the hyphen and join the word, 'keep' = keep it and join without a space."""
    if (left + "-" + right).lower() in keep:
        return "keep", "allowlist"                   # reviewed compound, e.g. cost-effective
    if (left + right).lower() in join:
        return "drop", "allowlist-join"              # reviewed: a real mid-word break
    if SUSPENDED_RE.match(next_text):
        return "keep-space", "suspended"             # high- / or low-probability
    if right[0].isdigit():
        return "keep", "digit"                       # GV- / 1.1-001  (an ID)
    if len(left) >= 2 and left.isupper() and right.isupper():
        return "drop", "caps"                        # GOV- / ERN
    if right[0].isupper():
        return "keep", "capital"                     # a proper-noun compound
    joined = (left + right).lower()
    hyphenated = (left + "-" + right).lower()
    n_joined, n_hyph = _count(vocab, joined), _count(vocab, hyphenated)
    if n_hyph and not n_joined:
        return "keep", "vocab-hyphen"                # the document writes it with a hyphen
    if n_joined and not n_hyph:
        return "drop", "vocab-joined"                # the document writes it as one word
    if n_joined and n_hyph:                          # both forms occur: the commoner one wins
        return ("keep" if n_hyph >= n_joined else "drop"), "vocab-both"
    return default, "guess"      # unknown: "drop" assumes a hyphenation break (TeX style),
                                 # "keep" assumes a real compound or URL (Word style)


def attach(cur, nxt, vocab, log, pno):
    """Add the next line to the paragraph so far."""
    if cur.endswith(SOFT_HYPHEN):
        log["hyphens"].append(dict(page=pno, kind="soft", action="drop", across=False,
                                   before=cur[-20:], after=nxt[:20]))
        return cur[:-1] + nxt
    if URL_TAIL_RE.search(cur.rsplit(" ", 1)[-1]):
        log["hyphens"].append(dict(page=pno, kind="url", action="keep", across=False,
                                   before=cur[-20:], after=nxt[:20]))
        return cur + nxt                              # a URL broken across lines: no space
    if not re.search(r"[A-Za-z0-9]-$", cur):
        return cur + " " + nxt
    m_right = re.match(r"[A-Za-z0-9]+", nxt)
    m_left = re.search(r"([A-Za-z0-9]+)-$", cur)
    if not m_right or not m_left:
        return cur + " " + nxt
    action, kind = decide(m_left.group(1), m_right.group(0), vocab,
                          log.get("keep_hyphen", ()), log.get("hyphen_default", "drop"),
                          nxt, log.get("join_hyphen", ()))
    log["hyphens"].append(dict(page=pno, kind=kind, action=action, across=False,
                               before=cur[-20:], after=nxt[:20]))
    if action == "drop":
        return cur[:-1] + nxt
    if action == "keep-space":
        return cur + " " + nxt
    return cur + nxt


def join_block(lines, vocab, starts, log, pno):
    """Turn the lines of one block into paragraphs."""
    paragraphs, cur = [], ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if cur and any(rx.match(line) for rx in starts):
            paragraphs.append(cur)
            cur = line
        elif not cur:
            cur = line
        else:
            cur = attach(cur, line, vocab, log, pno)
    if cur:
        paragraphs.append(cur)
    return [finalize(p) for p in paragraphs if finalize(p)]


def join_across_pages(pages, vocab, starts, log):
    """A word hyphenated at the bottom of a page continues at the top of the next page. The
    first word of the next page is moved back to complete it, so every paragraph keeps the
    page number it really sits on (the word-ending stays with the page where the word starts)."""
    for i in range(len(pages) - 1):
        cur_pars, nxt_pars = pages[i]["paragraphs"], pages[i + 1]["paragraphs"]
        if not cur_pars or not nxt_pars:
            continue
        last, first = cur_pars[-1], nxt_pars[0]
        broken = re.search(r"[A-Za-z0-9]-$", last) or last.endswith(SOFT_HYPHEN)
        if not broken or any(rx.match(first) for rx in starts):
            continue
        m = re.match(r"(\S+)\s*(.*)$", first, re.S)
        if not m:
            continue
        word, rest = m.group(1), m.group(2)
        before = len(log["hyphens"])
        cur_pars[-1] = finalize(attach(last, word, vocab, log, pages[i]["pdf_page"]))
        if len(log["hyphens"]) > before:
            log["hyphens"][-1]["across"] = True
            log["hyphens"][-1]["page_break"] = True
        if rest:
            nxt_pars[0] = rest
        else:
            nxt_pars.pop(0)


SENTENCE_END = re.compile(r"[.;:!?\u201d\"]$")
# a continuation starts in lower case, or with a legal citation such as "2016/679" (not with a
# list number such as "2. ")
CONTINUATION_START = re.compile(r"[a-z]|\d+/\d+")


def merge_split_paragraphs(paragraphs, vocab, starts, log, pno, continuation=False):
    """PyMuPDF sometimes ends a block in the middle of a sentence. Joins, within one page:
      - a paragraph ending in a hyphen after a letter, or in a soft hyphen (a hyphenation
        break), with the start of the next paragraph;
      - if `continuation` is on, a paragraph that does not end in sentence punctuation with a
        next paragraph that starts in lower case (a line of one list item or sentence that the
        extractor placed in its own block).
    A paragraph that starts like an ID or heading of the document is never joined to the one
    before it. Joins are tagged in the log so they can be reviewed."""
    out = []
    for p in paragraphs:
        if out and not any(rx.match(p) for rx in starts):
            prev = out[-1]
            if prev.endswith(SOFT_HYPHEN) or re.search(r"[A-Za-z0-9]-$", prev):
                before = len(log["hyphens"])
                out[-1] = finalize(attach(prev, p, vocab, log, pno))
                if len(log["hyphens"]) > before:
                    log["hyphens"][-1]["across"] = True
                continue
            if continuation and not SENTENCE_END.search(prev) and CONTINUATION_START.match(p):
                log["hyphens"].append(dict(page=pno, kind="continuation", action="join",
                                           across=True, before=prev[-20:], after=p[:20]))
                out[-1] = finalize(prev + " " + p)
                continue
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# ONE DOCUMENT
# ---------------------------------------------------------------------------
def auto_hyphen_default(pages_blocks, repeating, vocab):
    """Decide from the document itself how to treat a line-end hyphen it gives no evidence
    about. Count line-end hyphens whose joined form appears elsewhere as one word (breaks
    inside words, so joining is right) against those whose hyphenated form appears elsewhere
    (real compounds, so keeping is right)."""
    joined = hyph = 0
    for _pno, blocks in pages_blocks:
        for b in blocks:
            if b["band"] and repeat_key(b["text"]) in repeating:
                continue
            lines = [ln.strip() for ln in b["text"].split("\n") if ln.strip()]
            for cur, nxt in zip(lines, lines[1:]):
                m_left = re.search(r"([A-Za-z]+)-$", cur)
                m_right = re.match(r"[a-z]+", nxt)
                if not m_left or not m_right:
                    continue
                left, right = m_left.group(1).lower(), m_right.group(0)
                if (left + right) in vocab:
                    joined += 1
                if (left + "-" + right) in vocab:
                    hyph += 1
    return ("drop" if joined > hyph else "keep"), (joined, hyph)


def clean_document(pdf_path, cfg):
    doc = pymupdf.open(pdf_path)
    n_pages = len(doc)

    pages_blocks, key_pages = [], Counter()
    for pno, page in enumerate(doc, start=1):
        height = page.rect.height
        blocks = page_blocks(page)
        mark_bands(blocks, height, cfg["band"])
        pages_blocks.append((pno, blocks))
        for k in {repeat_key(b["text"]) for b in blocks if b["band"]}:
            key_pages[k] += 1

    threshold = max(REPEAT_MIN_PAGES, math.ceil(REPEAT_SHARE * n_pages))
    repeating = {k for k, c in key_pages.items() if k and c >= threshold}
    noise = [re.compile(p) for p in cfg.get("noise_lines", [])]
    noise_blocks = [re.compile(p) for p in cfg.get("noise_blocks", [])]
    prefix_strip = [re.compile(p) for p in cfg.get("strip_line_prefix", [])]
    pn_patterns = [re.compile(p) for p in cfg.get("page_number_patterns", [])]
    starts = [re.compile(p) for p in cfg.get("paragraph_starts", []) + [BULLET_START]]
    # Digits are ignored when looking for repetition, so headings such as "Article 7" and
    # "Article 8" look like the same text. A block that begins like an ID or heading of this
    # document is therefore never removed as a running header or footer.
    protect = [re.compile(p) for p in cfg.get("paragraph_starts", [])]
    vocab = build_vocab(pages_blocks, repeating)
    default, auto_counts = cfg.get("unknown_hyphen", "drop"), None
    if default == "auto":
        default, auto_counts = auto_hyphen_default(pages_blocks, repeating, vocab)
    log = dict(hyphens=[], threshold=threshold, repeating=sorted(repeating),
               keep_hyphen={w.lower() for w in cfg.get("keep_hyphen", [])},
               join_hyphen={w.lower() for w in cfg.get("join_hyphen", [])},
               hyphen_default=default, auto_counts=auto_counts)

    pages = []
    for pno, blocks in pages_blocks:
        removed, paragraphs, printed = [], [], None
        for b in blocks:
            text = b["text"]
            if b["band"]:
                pn = printed_page_of(text) or page_number_from_patterns(text, pn_patterns)
                if pn and printed is None:
                    printed = pn
                    removed.append(dict(text=text.strip(), reason="page number"))
                    continue
                key = repeat_key(text)
                first_line = text.strip().split("\n")[0].strip()
                if key in repeating and not any(rx.match(first_line) for rx in protect):
                    removed.append(dict(
                        text=text.strip(),
                        reason=f"repeating {b['band']} (on {key_pages[key]} pages)"))
                    continue
            flat = re.sub(r"\s+", " ", text).strip()
            if any(rx.match(flat) for rx in noise_blocks):
                removed.append(dict(text=flat, reason="noise block"))
                continue
            lines = []
            for line in text.split("\n"):
                if any(rx.match(line.strip()) for rx in noise):
                    removed.append(dict(text=line.strip(), reason="noise pattern"))
                else:
                    for rx in prefix_strip:
                        line = rx.sub("", line, count=1)
                    lines.append(line)
            paragraphs += join_block(lines, vocab, starts, log, pno)
        paragraphs = merge_split_paragraphs(paragraphs, vocab, starts, log, pno,
                                            cfg.get("merge_continuations", False))
        pages.append(dict(pdf_page=pno, printed_page=printed,
                          paragraphs=paragraphs, removed=removed))
    join_across_pages(pages, vocab, starts, log)
    for page in pages:                      # no soft hyphen is left in the final text
        page["paragraphs"] = [t.replace(SOFT_HYPHEN, "") for t in page["paragraphs"]]
    if cfg.get("drop_garbled"):
        for page in pages:
            keep = []
            for t in page["paragraphs"]:
                if len(t) > 15 and sum(ord(c) > 127 for c in t) > 0.2 * len(t):
                    page["removed"].append(dict(text=t[:80], reason="garbled text"))
                else:
                    keep.append(t)
            page["paragraphs"] = keep
    return pages, log


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_log(path, key, pages, log):
    auto = log.get("auto_counts")
    hyph_line = (f"Unknown line-end hyphen default: {log.get('hyphen_default')}"
                 + (f" (auto: {auto[0]} breaks inside words vs {auto[1]} real compounds)"
                    if auto else ""))
    out = [f"Cleaning log for {key}", hyph_line,
           f"Header/footer rule: text repeated in the top or bottom band on at least "
           f"{log['threshold']} pages",
           "", "Repeating header/footer patterns (digits shown as #):"]
    out += [f"  {k!r}" for k in log["repeating"]]
    out.append("")
    for p in pages:
        if not p["removed"]:
            continue
        out.append(f"== PDF page {p['pdf_page']} (printed {p['printed_page']}) ==")
        for r in p["removed"]:
            out.append(f"  removed [{r['reason']}]: {r['text']!r}")
    out += ["", "Hyphen decisions (kind, action, count):"]
    counts = Counter((h["kind"], h["action"]) for h in log["hyphens"])
    for (kind, action), c in sorted(counts.items()):
        out.append(f"  {kind:14s} {action:5s} {c}")
    out += ["", "Hyphen decisions worth a look (guesses, capitals, ambiguous):"]
    for h in log["hyphens"]:
        if (h["kind"] in ("guess", "vocab-both", "caps", "capital", "suspended", "allowlist",
                            "allowlist-join")
                or h.get("across")):
            tag = ("across-page " if h.get("page_break") else "across-block ") if h.get("across") else ""
            out.append(f"  p{h['page']:<4} {h['kind']:11s} {h['action']:5s} "
                       f"{tag}{h['before']!r} + {h['after']!r}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def run_key(key, root, quiet=False):
    """Clean one document. Returns (exit code, statistics for the summary table)."""
    def say(*a):
        if not quiet:
            print(*a)

    if key not in DOC_CONFIG or key not in FILES:
        print(f"No cleaning rules for '{key}' yet. Available: {', '.join(DOC_CONFIG)}")
        return 2, None
    fname, expected = FILES[key]
    pdf = root / fname
    if not pdf.exists():
        print(f"{key}: {pdf} not found")
        return 2, None
    sha = sha256_file(pdf)
    if not sha.startswith(expected):
        print(f"{key}: {fname}: SHA-256 starts {sha[:16]} but the answer key was written for "
              f"{expected}. This is a different version of the file.")
        return 2, None

    pages, log = clean_document(pdf, DOC_CONFIG[key])
    out = root / "cleaned"
    out.mkdir(exist_ok=True)
    (out / f"{key}.json").write_text(
        json.dumps(dict(doc=key, pdf=fname, pdf_sha256=sha, pages=pages),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    write_log(out / f"{key}_log.txt", key, pages, log)

    n_removed = sum(len(p["removed"]) for p in pages)
    n_paras = sum(len(p["paragraphs"]) for p in pages)
    kinds = Counter(h["kind"] for h in log["hyphens"])
    reasons = Counter(r["reason"].split(" (")[0] for p in pages for r in p["removed"])
    with_num = sum(1 for p in pages if p["printed_page"])
    across = sum(1 for h in log["hyphens"] if h.get("across") and not h.get("page_break"))
    over_pages = sum(1 for h in log["hyphens"] if h.get("page_break"))
    stats = dict(pages=len(pages), paragraphs=n_paras, removed=n_removed, reasons=dict(reasons),
                 printed=with_num, across_blocks=across, across_pages=over_pages,
                 hyphens=dict(kinds), hyphen_default=log["hyphen_default"],
                 auto=log.get("auto_counts"))
    say(f"{key}: {len(pages)} pages, {n_paras} paragraphs, {n_removed} removed items")
    say("removed by reason:", dict(reasons) or "none")
    say(f"printed page number found on {with_num} of {len(pages)} pages")
    say(f"paragraphs repaired across block breaks: {across}, across page breaks: {over_pages}")
    say("hyphen decisions:", dict(kinds) or "none")
    say(f"wrote cleaned/{key}.json and cleaned/{key}_log.txt")
    return 0, stats


def show(key, root, pno):
    json_path = root / "cleaned" / f"{key}.json"
    if not json_path.exists():
        print(f"{json_path} not found. Run: python clean_text.py {key}")
        return 2
    rec = json.loads(json_path.read_text(encoding="utf-8"))["pages"][pno - 1]
    raw = pymupdf.open(root / FILES[key][0])[pno - 1].get_text()
    print(f"===== RAW, PDF page {pno} =====")
    print(raw)
    print(f"===== CLEANED (printed page {rec['printed_page']}) =====")
    for i, p in enumerate(rec["paragraphs"], 1):
        print(f"[{i}] {p}\n")
    print("===== REMOVED =====")
    for r in rec["removed"]:
        print(f"[{r['reason']}] {r['text']!r}")
    return 0


# ---------------------------------------------------------------------------
# SELF-TEST (synthetic input, no real PDFs)
# ---------------------------------------------------------------------------
def run_all(root):
    """Clean every configured document and print one summary table."""
    rows, worst = [], 0
    for key in DOC_CONFIG:
        code, st = run_key(key, root, quiet=True)
        worst = max(worst, code)
        if st:
            rows.append((key, st))
    print()
    print(f"{'doc':7s}{'pages':>6s}{'paras':>7s}{'removed':>8s}{'numbers':>9s}"
          f"{'blk':>5s}{'pg':>4s}  unknown-hyphen default, guesses")
    for key, st in rows:
        auto = f" (auto {st['auto'][0]}:{st['auto'][1]})" if st["auto"] else ""
        print(f"{key:7s}{st['pages']:6d}{st['paragraphs']:7d}{st['removed']:8d}"
              f"{st['printed']:9d}{st['across_blocks']:5d}{st['across_pages']:4d}  "
              f"{st['hyphen_default']}{auto}, {st['hyphens'].get('guess', 0)} guesses")
    print("(numbers = pages with a printed page number found; blk / pg = words repaired "
          "across block / page breaks; auto a:b = breaks inside words : real compounds)")
    print("removed by reason:")
    for key, st in rows:
        print(f"  {key:7s}{st['reasons']}")
    return worst


def show_blocks(key, root, pno):
    """Print where each text block sits on a page. Use it to see why a header or footer
    was or was not recognised."""
    page = pymupdf.open(root / FILES[key][0])[pno - 1]
    height = page.rect.height
    blocks = page_blocks(page)
    mark_bands(blocks, height, DOC_CONFIG[key]["band"])
    print(f"page height {height:.0f} pt; band = top/bottom {DOC_CONFIG[key]['band'] * 100:.0f}% "
          f"(header if y1 <= {DOC_CONFIG[key]['band'] * height:.0f}, "
          f"footer if y0 >= {(1 - DOC_CONFIG[key]['band']) * height:.0f})")
    print("(header/footer here only means CANDIDATE: a candidate is removed only if its text "
          "repeats on many pages or it is just a page number)")
    for i, b in enumerate(blocks):
        flat = re.sub(r"\s+", " ", b["text"])[:55]
        print(f"[{i:2d}] y0={b['y0']:6.1f} y1={b['y1']:6.1f} {b['band'] or '-':6s} {flat!r}")
    return 0


def selftest():
    failures = []

    def check(name, ok):
        print(("PASS  " if ok else "FAIL  ") + name)
        if not ok:
            failures.append(name)

    # 1. characters
    check("ligature becomes plain letters", normalize_chars("\ufb01rst \ufb02ow") == "first flow")

    # 2. line joining on text taken from the AI RMF page 27 extraction
    lines = [
        "GOVERN is a cross-cutting function that is infused throughout AI risk management and",
        "enables the other functions of the process. Aspects of GOVERN, especially those related to",
        "compliance or evaluation, should be integrated into each of the other functions.",
        "Strong governance can drive and enhance internal practices and norms to facilitate orga-",
        "nizational risk culture. Documentation can enhance transparency and the organizational",
        "hierarchy of an AI system's lifespan.",
        "After putting in place the structures, systems, processes, and teams described in the GOV-",
        "ERN function, organizations should benefit from a purpose-driven culture. The GOVERN",
        "function is continual.",
        "GOVERN 1.1: Legal and regulatory requirements involving AI",
        "are understood, managed, and documented.",
        "GOVERN 1.2: The characteristics of trustworthy AI are inte-",
        "grated into organizational policies, processes, procedures, and",
        "practices.",
        "See GV-",
        "1.1-001 for the first action.",
    ]
    vocab = set()
    for ln in lines:
        add_line_tokens(vocab, ln)
    starts = [re.compile(DOC_CONFIG["rmf"]["paragraph_starts"][0]), re.compile(BULLET_START)]
    log = dict(hyphens=[])
    paras = join_block(lines, vocab, starts, log, 1)
    text = "\n".join(paras)
    check("hyphen break inside a word is joined (orga- / nizational)",
          "organizational risk culture" in text)
    check("hyphen break inside a capitalised word is joined (GOV- / ERN)",
          "GOVERN function, organizations" in text and "GOV-" not in text)
    check("a real compound keeps its hyphen (cross-cutting)", "cross-cutting function" in text)
    check("a hyphenated compound inside a line is untouched (purpose-driven)",
          "purpose-driven culture" in text)
    check("hyphen kept inside an ID split across lines (GV- / 1.1-001)",
          "See GV-1.1-001 for the first action." in text)
    check("table entries stay separate paragraphs",
          "GOVERN 1.1: Legal and regulatory requirements involving AI are understood, "
          "managed, and documented." in paras
          and any(p.startswith("GOVERN 1.2: The characteristics of trustworthy AI are "
                               "integrated into organizational policies") for p in paras))
    soft = join_block(["orga" + SOFT_HYPHEN, "nization of work"], set(), starts, dict(hyphens=[]), 1)
    check("soft hyphen at a line end joins the word", soft == ["organization of work"])

    # 3. a synthetic PDF with a header, a footer with page numbers and a noise line
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    tmp = Path(tempfile.mkdtemp()) / "synthetic.pdf"
    doc = pymupdf.open()
    for n in range(6):
        page = doc.new_page()                       # A4, height 842
        page.insert_text((72, 30), "TEST DOC 1.0", fontsize=9)
        page.insert_text((72, 44), "Governance Framework", fontsize=9)
        if n == 1:
            page.insert_text((72, 60), "Appendix A title", fontsize=9)   # one-off, must stay
        page.insert_text((72, 100),
                         f"This {words[n]} framework supports organizational risk culture and the\n"
                         "governance of each system. Strong governance requires orga-\n"
                         "nizational support.", fontsize=10)
        page.insert_text((72, 700), "Continued on next page", fontsize=9)
        page.insert_text((280, 812), f"Page {n + 10}", fontsize=9)
    doc.save(str(tmp))
    cfg = dict(band=0.09, noise_lines=[r"^Continued on next page$"], paragraph_starts=[])
    pages, plog = clean_document(tmp, cfg)
    all_text = "\n".join(t for p in pages for t in p["paragraphs"])
    check("printed page numbers are read (10..15)",
          [p["printed_page"] for p in pages] == [str(i) for i in range(10, 16)])
    check("repeating header text is removed",
          "TEST DOC" not in all_text and "Governance Framework" not in all_text)
    check("a one-off line in the header band is kept (not over-cleaned)",
          any("Appendix A title" in t for t in pages[1]["paragraphs"]))
    check("noise line is removed", "Continued on next page" not in all_text)
    check("hyphenated word is joined inside a real PDF block",
          "organizational support" in all_text)
    check("removals are logged with a reason",
          all(any(r["reason"] == "page number" for r in p["removed"]) for p in pages))

    # 4. header and footer OUTSIDE the narrow band (as in the real AI RMF), and a sentence
    #    split across two blocks at a hyphen
    tmp2 = Path(tempfile.mkdtemp()) / "synthetic2.pdf"
    doc = pymupdf.open()
    for n in range(6):
        page = doc.new_page()                       # A4: the band is only the top/bottom 75 pt
        page.insert_text((72, 90), "NIST TEST 1", fontsize=9)        # y 90: outside the band
        page.insert_text((400, 90), "FRAMEWORK 1.0", fontsize=9)
        page.insert_text((72, 140), f"Strong {words[n]} governance matters and the framework supports the\n"
                                    "GOVERN function in every organization.", fontsize=10)
        page.insert_text((72, 200), f"After putting in place the {words[n]} structures described in the GOV-",
                         fontsize=10)
        page.insert_text((72, 260), f"ERN function, {words[n]} organizations should benefit from clear roles.",
                         fontsize=10)
        page.insert_text((280, 740), f"Page {n + 20}", fontsize=9)  # y 740: outside the band
    doc.save(str(tmp2))
    pages2, _ = clean_document(tmp2, dict(band=0.09, noise_lines=[], paragraph_starts=[]))
    text2 = "\n".join(t for p in pages2 for t in p["paragraphs"])
    check("header outside the narrow band is still removed",
          "NIST TEST" not in text2 and "FRAMEWORK 1.0" not in text2)
    check("footer outside the narrow band is removed and its page number read",
          "Page 2" not in text2
          and [p["printed_page"] for p in pages2] == [str(i) for i in range(20, 26)])
    check("a hyphen split across two blocks is repaired",
          "GOVERN function, alpha organizations" in text2 and "GOV-" not in text2)
    check("real body text is not removed", "Strong alpha governance matters" in text2)

    # 5. suspended hyphen, block-level noise, and a word split across a page break
    susp = join_block(["Events of short-term, high-", "or low-probability type."],
                      set(), starts, dict(hyphens=[]), 1)
    check("suspended hyphen keeps its hyphen (high- or low-probability)",
          susp == ["Events of short-term, high- or low-probability type."])

    keep_log = dict(hyphens=[], keep_hyphen={"cost-effective"})
    kept = join_block(["a plan that is cost-", "effective for all and a cost-", "benefit view."],
                      set(), starts, keep_log, 1)
    check("reviewed compound keeps its hyphen (allowlist), unknown ones are joined",
          kept == ["a plan that is cost-effective for all and a costbenefit view."])

    url = join_block(["See https://example.org/news/ai-explained-", "part-1/ for details."],
                     set(), starts, dict(hyphens=[]), 1)
    check("a URL broken across lines is joined without a space",
          url == ["See https://example.org/news/ai-explained-part-1/ for details."])
    cap = join_block(["Memes From Text-", "To-Image Models."], set(), starts, dict(hyphens=[]), 1)
    check("a capitalised To is not treated as a suspended hyphen (Text-To-Image)",
          cap == ["Memes From Text-To-Image Models."])
    word = join_block(["a state-of-", "the-art system and a round-", "trip flight"], set(), starts,
                      dict(hyphens=[], hyphen_default="keep"), 1)
    check("with unknown_hyphen=keep, compounds survive (state-of-the-art, round-trip)",
          word == ["a state-of-the-art system and a round-trip flight"])

    v1 = set()
    for ln in ["the organiza-", "tion of work and the organization", "of the plan"]:
        add_line_tokens(v1, ln)
    d1, c1 = auto_hyphen_default([(1, [dict(band=None, text="the organiza-\ntion of work")])], set(), v1)
    v2 = set()
    for ln in ["a cost-effective plan", "and a cost-"]:
        add_line_tokens(v2, ln)
    d2, c2 = auto_hyphen_default([(1, [dict(band=None, text="and a cost-\neffective plan")])], set(), v2)
    check("auto hyphen default: breaks inside words -> drop, compounds -> keep",
          d1 == "drop" and d2 == "keep")

    tmp5 = Path(tempfile.mkdtemp()) / "synthetic5.pdf"
    doc = pymupdf.open()
    for n in range(6):
        page = doc.new_page()
        page.insert_text((72, 90), "NIST TEST 1", fontsize=9)
        page.insert_text((72, 110), f"Section {n + 1}.{n + 1}", fontsize=11)   # top of every page
        page.insert_text((72, 300), f"Body of the {words[n]} section.", fontsize=10)
    doc.save(str(tmp5))
    pages5, _ = clean_document(tmp5, dict(band=0.09, noise_lines=[],
                                          paragraph_starts=[r"^Section \d+"]))
    text5 = "\n".join(t for p in pages5 for t in p["paragraphs"])
    check("headings that repeat at the top of every page are protected from removal",
          all(f"Section {n}.{n}" in text5 for n in range(1, 7)) and "NIST TEST" not in text5)

    comp = join_block(["tested on ready-", "to-deploy systems and a low-", "or high-risk case"],
                      set(), starts, dict(hyphens=[], hyphen_default="keep"), 1)
    check("ready-/to-deploy is a compound; low-/or high-risk is a suspended hyphen",
          comp == ["tested on ready-to-deploy systems and a low- or high-risk case"])
    vb = Counter({"fine-tuning": 5, "finetuning": 1})
    check("when both forms occur, the commoner one wins",
          decide("fine", "tuning", vb)[0] == "keep"
          and decide("fine", "tuning", Counter({"fine-tuning": 1, "finetuning": 4}))[0] == "drop")
    jl = join_block(["more modality combina-", "tions increase"], set(), starts,
                    dict(hyphens=[], hyphen_default="keep", join_hyphen={"combinations"}), 1)
    check("a reviewed mid-word break is joined even when the default is keep",
          jl == ["more modality combinations increase"])

    tmp6 = Path(tempfile.mkdtemp()) / "synthetic6.pdf"
    doc = pymupdf.open()
    for n in range(5):
        page = doc.new_page()
        page.insert_text((72, 140), f"Body text of the {words[n]} page.", fontsize=10)
        page.insert_text((200, 740), f"{n + 30} example.org", fontsize=9)
    doc.save(str(tmp6))
    pages6, _ = clean_document(tmp6, dict(band=0.09, noise_lines=[], paragraph_starts=[],
                                          page_number_patterns=[r"^(\d{1,3}) example\.org$"]))
    check("a page number inside footer text is read ('31 example.org')",
          [p["printed_page"] for p in pages6] == [str(i) for i in range(30, 35)]
          and "example.org" not in "".join(t for p in pages6 for t in p["paragraphs"]))

    sh = merge_split_paragraphs(["(b) subject to technical limi" + SOFT_HYPHEN, "tations on the re-use."],
                                set(), starts, dict(hyphens=[]), 1)
    check("a word split by a soft hyphen between two blocks is joined",
          [t.replace(SOFT_HYPHEN, "") for t in sh] == ["(b) subject to technical limitations on the re-use."])
    cont_in = ["(a) the bias detection cannot be fulfilled by", "processing other data;",
               "(b) all conditions are met.", "Done.", "also lower case"]
    on = merge_split_paragraphs(list(cont_in), set(), starts, dict(hyphens=[]), 1, True)
    off = merge_split_paragraphs(list(cont_in), set(), starts, dict(hyphens=[]), 1, False)
    check("continuation merge joins a split list item, and only when switched on",
          on == ["(a) the bias detection cannot be fulfilled by processing other data;",
                 "(b) all conditions are met.", "Done.", "also lower case"] and off == cont_in)
    cit = merge_split_paragraphs(["(f) the records pursuant to Regulations (EU)",
                                  "2016/679 and (EU) 2018/1725 include the reasons.",
                                  "2. A new numbered paragraph."], set(), starts, dict(hyphens=[]), 1, True)
    check("a split ending in a closing bracket is joined to a citation, but not to a new list number",
          cit == ["(f) the records pursuant to Regulations (EU) 2016/679 and (EU) 2018/1725 include the reasons.",
                  "2. A new numbered paragraph."])

    tmp7 = Path(tempfile.mkdtemp()) / "synthetic7.pdf"
    doc = pymupdf.open()
    for n in range(3):
        page = doc.new_page()
        page.insert_text((72, 140), f"Readable text of the {words[n]} page.", fontsize=10)
        page.insert_text((72, 200), "\u00e9\u00fc\u00f1\u00e9\u00fc\u00f1\u00e9\u00fc\u00f1\u00e9\u00fc\u00f1\u00e9\u00fc\u00f1", fontsize=10)
        page.insert_text((72, 300), "Stray header text v2.0" if n == 1 else f"Other {words[n + 3]} text", fontsize=10)
    doc.save(str(tmp7))
    pages7, _ = clean_document(tmp7, dict(band=0.09, noise_lines=[], paragraph_starts=[], drop_garbled=True,
                                          noise_blocks=[r"^Stray header text v2\.0$"]))
    text7 = "\n".join(t for p in pages7 for t in p["paragraphs"])
    check("garbled paragraphs and a named stray header are removed, readable text stays",
          "\u00e9\u00fc\u00f1" not in text7 and "Stray header" not in text7 and "Readable text of the alpha" in text7)

    tmp3 = Path(tempfile.mkdtemp()) / "synthetic3.pdf"
    doc = pymupdf.open()
    for n in range(4):
        page = doc.new_page()
        page.insert_text((72, 90), "NIST TEST 1", fontsize=9)
        page.insert_text((72, 300), "Categories\nSubcategories", fontsize=10)
        if n == 1:
            page.insert_text((72, 140), "Please consider the case for exam-", fontsize=10)
        elif n == 2:
            page.insert_text((72, 140), "ple, how the framework applies in practice.", fontsize=10)
        else:
            page.insert_text((72, 140), f"Page body {words[n]} text of the framework.", fontsize=10)
        page.insert_text((280, 740), f"Page {n + 1}", fontsize=9)
    doc.save(str(tmp3))
    pages3, _ = clean_document(tmp3, dict(band=0.09, noise_lines=[], noise_blocks=[r"^Categories Subcategories$"],
                                          paragraph_starts=[]))
    text3 = "\n".join(t for p in pages3 for t in p["paragraphs"])
    check("block-level noise (table column headings) is removed", "Categories" not in text3)
    tmp4 = Path(tempfile.mkdtemp()) / "synthetic4.pdf"
    doc = pymupdf.open()
    for n in range(3):
        page = doc.new_page()
        page.insert_text((72, 140), f"o GV.OC-0{n + 1}: The {words[n]} mission is understood and\n"
                                    "informs risk management.", fontsize=10)
        page.insert_text((280, 740), f"Page {n + 1}", fontsize=9)
    doc.save(str(tmp4))
    pages4, _ = clean_document(tmp4, dict(
        band=0.09, noise_lines=[], paragraph_starts=[r"^GV\.OC-\d{2}:"],
        strip_line_prefix=[r"^o\s+(?=GV\.OC-\d{2}:)"]))
    text4 = "\n".join(t for p in pages4 for t in p["paragraphs"])
    check("a bullet turned into a letter o is stripped before an ID",
          "GV.OC-01: The alpha mission" in text4 and "o GV.OC" not in text4)
    check("word split across a page break is repaired",
          pages3[1]["paragraphs"][-1].endswith("for example,") and "exam-" not in text3)
    check("the rest of the sentence stays on the next page",
          pages3[2]["paragraphs"] and pages3[2]["paragraphs"][0].startswith("how the framework"))

    print()
    print("selftest: " + ("all checks passed" if not failures
                          else f"{len(failures)} check(s) FAILED"))
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Clean a PDF's text into paragraphs.")
    ap.add_argument("key", nargs="?", help="document key, e.g. rmf")
    ap.add_argument("--show", type=int, metavar="PDF_PAGE",
                    help="show raw vs cleaned text for one PDF page (run the cleaning first)")
    ap.add_argument("--blocks", type=int, metavar="PDF_PAGE",
                    help="print where each text block sits on one PDF page (diagnostic)")
    ap.add_argument("--selftest", action="store_true", help="run the built-in tests")
    ap.add_argument("--dir", help="folder with the PDFs (default: this script's folder)")
    args = ap.parse_args(argv)
    root = Path(args.dir) if args.dir else Path(__file__).resolve().parent

    if args.selftest:
        return selftest()
    if not args.key:
        ap.print_help()
        return 0
    if args.key == "all":
        return run_all(root)
    if args.show:
        return show(args.key, root, args.show)
    if args.blocks:
        return show_blocks(args.key, root, args.blocks)
    code, _ = run_key(args.key, root)
    return code


if __name__ == "__main__":
    sys.exit(main())