#!/usr/bin/env python3
"""
chunker.py - cut the cleaned documents into chunks for search.

A chunk is the piece of text that is embedded, searched and shown to the language model.
Cutting at the document's own structure (one subcategory, one suggested action) keeps an
answer and its context together; a blind 500-character slice would not.

Two kinds of chunk:
  entry      one structured item, with its context attached:
               AI RMF   a subcategory (GOVERN 1.1) plus its category statement
               AI 600-1 a suggested action (GV-1.1-001) plus its subcategory statement,
                        its GAI risk tags and the subcategory's AI Actor Tasks
  narrative  ordinary text: consecutive paragraphs packed up to a word limit, never cut in
             the middle of a sentence, a heading kept with the paragraph after it, and a
             one-sentence overlap between neighbouring chunks.

Every chunk has two texts:
  text        what is shown and cited
  embed_text  the same text with a short context header in front (document, ID), which is what
              is embedded, so that a chunk is findable by where it comes from

Every chunk records its PDF page range and printed page range, for citations.

Usage (from the project folder, after python clean_text.py all):
  python chunker.py rmf                 chunk the AI RMF   (writes chunks/rmf.json)
  python chunker.py gai                 chunk AI 600-1     (writes chunks/gai.json)
  python chunker.py all                 both
  python chunker.py gai --show GV-1.1-001     print the chunk(s) whose id contains this text
  python chunker.py --selftest          test the parsing rules on small examples
Then:  python check_chunks.py
"""
import argparse
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

DOCS = {
    "rmf": dict(title="NIST AI 100-1 (AI RMF 1.0)"),
    "gai": dict(title="NIST AI 600-1 (Generative AI Profile)"),
    "csf": dict(title="NIST CSWP 29 (Cybersecurity Framework 2.0)"),
    "ssdf": dict(title="NIST SP 800-218A (Secure Software Development for Generative AI)"),
    "owasp": dict(title="OWASP Top 10 for LLM Applications 2025"),
    "eu": dict(title="EU AI Act (Regulation (EU) 2024/1689)"),
    "aml": dict(title="NIST AI 100-2 E2025 (Adversarial Machine Learning)"),
    "deploy": dict(title="Deploying AI Systems Securely (NSA, CISA and partners)"),
    "data": dict(title="AI Data Security (NSA, CISA and partners)"),
    "uk": dict(title="AI Playbook for the UK Government"),
}

HEADING_RUN_WORDS = 40      # headings alone up to this many words are kept with the paragraph after them
MAX_WORDS = 220            # about 300 tokens: fits small embedding models (256 to 512 tokens)
OVERLAP_SENTENCES = 1      # sentences repeated at the start of the next narrative chunk

# the twelve GAI risks of AI 600-1, in their canonical spelling, and variants the document itself
# uses (found in the first run on the real file)
KNOWN_GAI_RISKS = [
    "CBRN Information or Capabilities", "Confabulation", "Dangerous, Violent, or Hateful Content",
    "Data Privacy", "Environmental Impacts", "Harmful Bias and Homogenization",
    "Human-AI Configuration", "Information Integrity", "Information Security",
    "Intellectual Property", "Obscene, Degrading, and/or Abusive Content",
    "Value Chain and Component Integration"]
CANON = {r.lower(): r for r in KNOWN_GAI_RISKS}
CANON.update({
    "cbrn information and capability": "CBRN Information or Capabilities",
    "cbrn information and capabilities": "CBRN Information or Capabilities",
    "cbrn information or capability": "CBRN Information or Capabilities",
    "environmental impact": "Environmental Impacts",
    "environmental": "Environmental Impacts",      # the cell wraps and only this word follows the action
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def words(text):
    return len(text.split())


def stream(pages):
    for p in pages:
        for t in p["paragraphs"]:
            yield dict(text=t, pdf=p["pdf_page"], printed=p["printed_page"])


def span(pieces):
    pdfs = [p["pdf"] for p in pieces]
    printed = [p["printed"] for p in pieces if p["printed"]]
    return dict(pdf_start=min(pdfs), pdf_end=max(pdfs),
                printed_start=printed[0] if printed else None,
                printed_end=printed[-1] if printed else None)


def make_chunk(doc, title, cid, kind, section, text, pieces, header, meta=None):
    chunk = dict(id=cid, doc=doc, title=title, kind=kind, section=section,
                 text=text, embed_text=f"{header}\n{text}", words=words(text),
                 meta=meta or {})
    chunk.update(span(pieces))
    return chunk


HEADING_END = re.compile(r"[.;:!?,]$")


def looks_like_heading(text):
    w = text.split()
    return 0 < len(w) <= 12 and not HEADING_END.search(text) and not text.startswith("\u2022")


def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\u201c\"])", text)


SPLITTERS = [r"(?<=[;:])\s+", r"(?<=,)\s+"]


def hard_split(text, max_words, level=0):
    """Split text that has no usable sentence boundary: at semicolons and colons first, then at
    commas, and only as a last resort at a word boundary. Every piece is at most max_words."""
    if words(text) <= max_words:
        return [text]
    if level < len(SPLITTERS):
        parts = re.split(SPLITTERS[level], text)
        if len(parts) > 1:
            out, cur = [], ""
            for part in parts:
                for sub in hard_split(part, max_words, level + 1):
                    if cur and words(cur + " " + sub) > max_words:
                        out.append(cur)
                        cur = sub
                    else:
                        cur = (cur + " " + sub).strip()
            if cur:
                out.append(cur)
            return out
        return hard_split(text, max_words, level + 1)
    toks = text.split()
    return [" ".join(toks[i:i + max_words]) for i in range(0, len(toks), max_words)]


def split_long(text, max_words):
    """Split a paragraph that is longer than max_words at sentence boundaries; a single sentence
    that is itself too long is split further by hard_split."""
    pieces, cur = [], []
    for sent in split_sentences(text):
        if words(sent) > max_words:
            if cur:
                pieces.append(" ".join(cur))
                cur = []
            pieces.extend(hard_split(sent, max_words))
        elif cur and words(" ".join(cur + [sent])) > max_words:
            pieces.append(" ".join(cur))
            cur = [sent]
        else:
            cur.append(sent)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


# ---------------------------------------------------------------------------
# narrative text: pack paragraphs up to a word limit
# ---------------------------------------------------------------------------
def pack(items, doc, title, max_words=MAX_WORDS, overlap=OVERLAP_SENTENCES, header=None,
         id_prefix=None, kind="narrative", section=None, meta=None):
    """Pack paragraphs into chunks. With `header`, every chunk gets that context header; with
    `id_prefix`, chunk ids are prefix, prefix#2, prefix#3 (otherwise doc:n0001, ...)."""
    out, pending, last_heading, chunk_section = [], [], "", ""

    def emit():
        real = [p for p in pending if not p["overlap"]]
        if not real:
            return
        text = "\n".join(p["text"] for p in pending)
        n = len(out) + 1
        cid = (id_prefix if n == 1 else f"{id_prefix}#{n}") if id_prefix else f"{doc}:n{n:04d}"
        hdr = header if header is not None else title + (f", {chunk_section}" if chunk_section else "")
        sec = section if section is not None else chunk_section
        out.append(make_chunk(doc, title, cid, kind, sec, text, real, hdr, dict(meta or {})))

    def overlap_piece(pieces):
        if overlap <= 0:            # sentences[-0:] would be the whole paragraph, not nothing
            return None
        for p in reversed(pieces):
            if p["heading"] or p["overlap"]:
                continue
            sentences = split_sentences(p["text"])
            if len(sentences) > 1:
                return dict(text=" ".join(sentences[-overlap:]), pdf=p["pdf"],
                            printed=p["printed"], heading=False, overlap=True)
            return None
        return None

    for it in items:
        text = it["text"]
        is_heading = looks_like_heading(text)
        if is_heading:
            last_heading = text
        parts = split_long(text, max_words) if words(text) > max_words else [text]
        for part in parts:
            piece = dict(text=part, pdf=it["pdf"], printed=it["printed"],
                         heading=is_heading, overlap=False)
            total = sum(words(p["text"]) for p in pending)
            # A chunk that holds only a few headings is not closed (the next paragraph just joins
            # them); but a long run of short heading-like lines, such as a list of figure labels,
            # must still be cut at the limit.
            only_headings = not any(not p["heading"] and not p["overlap"] for p in pending)
            if pending and (not only_headings or total > HEADING_RUN_WORDS) and total + words(part) > max_words:
                carried = []
                if pending[-1]["heading"] and len(pending) > 1:
                    carried = [pending.pop()]          # a heading stays with what follows it
                emit()
                ov = overlap_piece(pending)
                pending = ([ov] if ov else []) + carried
                chunk_section = last_heading if not carried else carried[0]["text"]
            if not any(not p["overlap"] for p in pending):
                chunk_section = last_heading
            pending.append(piece)
    emit()
    return out


# ---------------------------------------------------------------------------
# AI RMF: one chunk per subcategory, with its category statement
# ---------------------------------------------------------------------------
RMF_CAT = re.compile(r"^(GOVERN|MAP|MEASURE|MANAGE) (\d+):")
RMF_SUB = re.compile(r"^(GOVERN|MAP|MEASURE|MANAGE) (\d+\.\d+):")


def chunk_rmf(items, doc, title):
    # category statements are collected first, so it does not matter whether the extraction
    # put a category before or after its subcategories
    cats = {}
    for it in items:
        m = RMF_CAT.match(it["text"])
        if m:
            cats.setdefault(f"{m.group(1)} {m.group(2)}", it)
    entries, narrative = [], []
    for it in items:
        t = it["text"]
        ms = RMF_SUB.match(t)
        if ms:
            fn, num = ms.group(1), ms.group(2)
            cat_key = f"{fn} {num.split('.')[0]}"
            cat = cats.get(cat_key)
            text = (cat["text"] + "\n" if cat else "") + t
            entries.append(make_chunk(
                doc, title, f"{doc}:{fn} {num}", "entry", f"{fn} {num}", text, [it],
                f"{title}, subcategory {fn} {num}",
                dict(function=fn, category=cat_key, category_found=bool(cat))))
        elif RMF_CAT.match(t):
            continue                      # attached to its subcategories above
        else:
            narrative.append(it)
    return entries + pack(narrative, doc, title)


# ---------------------------------------------------------------------------
# AI 600-1: one chunk per suggested action
# ---------------------------------------------------------------------------
GAI_SUB = RMF_SUB
GAI_ID = re.compile(r"^((?:GV|MP|MS|MG)-\d+\.\d+-\d{3})\b[:\s]*(.*)$", re.S)
TASKS = re.compile(r"^AI Actor Tasks:\s*(.*)$", re.S)


INLINE_TASKS = re.compile(r"\s*AI Actors?(?: Tasks?)?:\s*(.+)$", re.S)
TAIL = re.compile(r"[.!?)\u201d]\s+([^.!?:]+)$")          # text after the last sentence end


NAME_RX = re.compile("|".join(re.escape(n) for n in sorted(CANON, key=len, reverse=True)), re.I)


def names_only(text):
    """Text that consists only of known risk names separated by semicolons or commas (the
    source sometimes types a comma where a semicolon belongs)."""
    found = list(NAME_RX.finditer(text))
    if found and re.fullmatch(r"[;,\s]*", NAME_RX.sub("", text)):
        return [CANON[re.sub(r"\s+", " ", m.group(0)).lower()] for m in found]
    return None


def parse_tags(text):
    """The list of risk tags if `text` is a line of tags (items separated by ';'), else None.
    At least one item must be a known risk or a known variant of one; the other items only have
    to look like a tag (short, no sentence punctuation), for example 'Civil Rights violations'."""
    items = [p.strip() for p in re.split(r";\s*", text.strip()) if p.strip()]
    if not items:
        return None
    out, known = [], 0
    for it in items:
        c = CANON.get(re.sub(r"\s+", " ", it).lower())
        if c:
            known += 1
            out.append(c)
        elif len(it.split()) <= 8 and not re.search(r"[.!?:]", it):
            out.append(it)
        else:
            return names_only(text)
    return out if known else names_only(text)


def split_tail_tags(body):
    """Risk tags are sometimes in the same paragraph as the action text, after its last
    sentence: 'Establish policies... for GAI systems. Information Integrity'."""
    m = TAIL.search(body)
    if m and parse_tags(m.group(1)):
        return body[:m.start() + 1].rstrip(), m.group(1).strip()
    return body, None


def parse_entry(parts):
    """From the paragraphs of one action, return (action text, raw risk tags, tasks)."""
    parts = list(parts)
    tasks = None
    if parts:
        m = INLINE_TASKS.search(parts[-1])                 # 'AI Actors: AI Design' inside the paragraph
        if m:
            tasks = m.group(1).strip()
            rest = parts[-1][:m.start()].rstrip()
            if rest:
                parts[-1] = rest
            else:
                parts.pop()
    risks_raw = None
    for idx in range(len(parts) - 1, -1, -1):             # tags as a paragraph of their own,
        if parse_tags(parts[idx]):                        # usually last, sometimes in the middle
            risks_raw = parts.pop(idx)                    # of the action text (reading order)
            break
    action = " ".join(parts).strip()
    if risks_raw is None:
        action, risks_raw = split_tail_tags(action)        # tags at the end of the action paragraph
    return action, risks_raw, tasks


def chunk_gai(items, doc, title):
    blocks, narrative, current = [], [], None
    for it in items:
        t = it["text"]
        ms = GAI_SUB.match(t)
        if ms:                                                   # a subcategory starts a block
            current = dict(sub_id=f"{ms.group(1)} {ms.group(2)}", sub=it, entries=[], tasks=None)
            blocks.append(current)
            continue
        if current is not None:
            mt = TASKS.match(t)
            if mt:                                               # the tasks line ends the block
                current["tasks"] = mt.group(1).strip()
                current = None
                continue
            mi = GAI_ID.match(t)
            if mi:
                current["entries"].append(dict(id=mi.group(1), parts=[mi.group(2)] if mi.group(2) else [],
                                               items=[it]))
                continue
            if current["entries"]:                               # action text or risk tags
                current["entries"][-1]["parts"].append(t)
                current["entries"][-1]["items"].append(it)
                continue
        narrative.append(it)

    entries = []
    for b in blocks:
        parsed = [parse_entry(e["parts"]) for e in b["entries"]]
        tasks = b["tasks"] or next((t for _, _, t in reversed(parsed) if t), None)   # shared by the block
        for e, (action, risks_raw, _) in zip(b["entries"], parsed):
            lines = [b["sub"]["text"], f"{e['id']}: {action}" if action else e["id"]]
            if risks_raw:
                lines.append(f"GAI Risks: {risks_raw}")
            if tasks:
                lines.append(f"AI Actor Tasks: {tasks}")
            text = "\n".join(lines)
            entries.append(make_chunk(
                doc, title, f"{doc}:{e['id']}", "entry", e["id"], text, e["items"],
                f"{title}, suggested action {e['id']} (subcategory {b['sub_id']})",
                dict(subcategory=b["sub_id"], risks=parse_tags(risks_raw) if risks_raw else [],
                     tasks=tasks, has_action_text=bool(action))))
    return entries + pack(narrative, doc, title)


def selftest():
    failures = []

    def check(name, ok):
        print(("PASS  " if ok else "FAIL  ") + name)
        if not ok:
            failures.append(name)

    check("tags as their own paragraph",
          parse_entry(["Do the thing for GAI systems.", "Information Integrity"])
          == ("Do the thing for GAI systems.", "Information Integrity", None))
    check("tags at the end of the action paragraph",
          parse_entry(["Do the thing for GAI systems. Information Integrity"])
          == ("Do the thing for GAI systems.", "Information Integrity", None))
    check("a variant spelling of a risk is recognised and canonicalised",
          parse_tags("CBRN Information and Capability; Information Security")
          == ["CBRN Information or Capabilities", "Information Security"])
    check("an unlisted tag next to a known one is accepted",
          parse_tags("Obscene, Degrading, and/or Abusive Content; Data Privacy; Civil Rights violations")
          == ["Obscene, Degrading, and/or Abusive Content", "Data Privacy", "Civil Rights violations"])
    check("tags and tasks both inside the last paragraph",
          parse_entry(["Engage in threat modeling. CBRN Information or Capabilities; Information Security"
                       " AI Actors: AI Design"])
          == ("Engage in threat modeling.", "CBRN Information or Capabilities; Information Security", "AI Design"))
    check("tags separated by a comma instead of a semicolon are recognised",
          parse_entry(["Do the thing. Human-AI Configuration, Dangerous, Violent, or Hateful Content"])
          == ("Do the thing.", "Human-AI Configuration, Dangerous, Violent, or Hateful Content", None)
          and parse_tags("Human-AI Configuration, Dangerous, Violent, or Hateful Content")
          == ["Human-AI Configuration", "Dangerous, Violent, or Hateful Content"])
    check("a tags paragraph in the middle of the action text is found and the text rejoined",
          parse_entry(["First half of the action text,", "Harmful Bias and Homogenization",
                       "second half of the action text."])
          == ("First half of the action text, second half of the action text.",
              "Harmful Bias and Homogenization", None))
    check("Environmental alone is read as the Environmental Impacts risk",
          parse_tags("Environmental") == ["Environmental Impacts"]
          and parse_entry(["Do it in design decisions. Environmental"])
          == ("Do it in design decisions.", "Environmental", None))
    check("a closing sentence that is not a risk is not taken for tags",
          parse_entry(["Do the thing. Governance and Oversight"])
          == ("Do the thing. Governance and Oversight", None, None))
    items = [dict(text=t, pdf=17, printed="13") for t in [
        "GOVERN 3.2: Policies define roles.",
        "GV-3.2-001 First action text. Human-AI Configuration",
        "GV-3.2-002", "Second action text.", "Data Privacy",
        "GV-3.2-003 Third action. Information Security AI Actors: AI Design"]]
    chunks = [c for c in chunk_gai(items, "gai", "T") if c["kind"] == "entry"]
    check("a tasks line found inside the last action is shared by every action of the subcategory",
          len(chunks) == 3 and all(c["meta"]["tasks"] == "AI Design" for c in chunks)
          and all(c["text"].rstrip().endswith("AI Actor Tasks: AI Design") for c in chunks))
    check("risk tags are found in all three shapes",
          [c["meta"]["risks"] for c in chunks] == [["Human-AI Configuration"], ["Data Privacy"],
                                                    ["Information Security"]])
    long_par = " ".join(f"Sentence number {i} has some words in it." for i in range(26))   # about 210 words
    head_only = pack([dict(text="LLM09: Misinformation", pdf=5, printed="1"),
                      dict(text="Prevention and Mitigation Strategies", pdf=5, printed="1"),
                      dict(text=long_par, pdf=5, printed="1")], "x", "T", overlap=0)
    check("headings are never left alone in a chunk of their own",
          all(any(not looks_like_heading(q) for q in c["text"].split("\n")) for c in head_only))
    om = OW_RISK.match("LLM01:2025 Prompt Injection"), OW_RISK.match("LLM04: Data and Model Poisoning")
    check("both OWASP heading styles are read (LLM01:2025 ... and LLM04: ...)",
          om[0] and om[0].group(2) == "Prompt Injection" and om[1] and om[1].group(2) == "Data and Model Poisoning")
    ss_items = [dict(text=t, pdf=27, printed="19") for t in
                ["Practice Name (RV.3): Do the thing.", "RV.3.4: Do the last task.", "High R1: A note.",
                 "AI RMF: Govern 1.1", "Appendix A", " ".join(["Appendix text"] * 300)]]
    ss = chunk_ssdf(ss_items, "ssdf", "T")
    ss_entry = [c for c in ss if c["kind"] == "entry"]
    check("the last SP 800-218A task stops at the appendix and does not swallow it",
          len(ss_entry) == 1 and ss_entry[0]["words"] < 40
          and any("Appendix text" in c["text"] for c in ss if c["kind"] == "narrative"))
    cap = clean_caption("**1. Figure title:**\nFig. 5. Using the CSF.\n\n**2. Labels:**\n* Executives\n  * Managers")
    check("caption markup is removed and the words are kept",
          cap == ["1. Figure title:", "Fig. 5. Using the CSF.", "2. Labels:", "- Executives", "- Managers"])
    plain = [dict(text=f"Paragraph {i} " + "word " * 90, pdf=1, printed="1") for i in range(6)]
    for ov in (0, 1):
        got = pack(plain, "x", "T", overlap=ov)
        joined = " ".join(c["text"] for c in got)
        if ov == 0:
            check("with overlap=0 no text is repeated between chunks",
                  len(got) > 1 and all(joined.count(f"Paragraph {i} ") == 1 for i in range(6)))
        else:
            check("with overlap=1 only one sentence is repeated",
                  len(got) > 1 and joined.count("Paragraph 0 ") == 1)
    labels = [dict(text=f"- Label number {i} in a long table", pdf=1, printed="1") for i in range(300)]
    cut = pack(labels, "x", "T", overlap=0)
    check("a long list of short heading-like lines (figure labels) is still cut at the limit",
          len(cut) > 5 and all(c["words"] <= 235 for c in cut)
          and sum(c["words"] for c in cut) == sum(words(x["text"]) for x in labels))
    runon = "; ".join(f"item {i} describes a measure with several words in it" for i in range(60))   # no sentence end
    solid = " ".join(f"word{i}" for i in range(500))                                                   # no punctuation at all
    check("text with no sentence boundary is split into pieces within the limit",
          all(words(x) <= 220 for x in split_long(runon, 220)) and len(split_long(runon, 220)) > 1
          and all(words(x) <= 220 for x in split_long(solid, 220))
          and " ".join(split_long(solid, 220)) == solid)
    print("\nselftest: " + ("all checks passed" if not failures else f"{len(failures)} FAILED"))
    return 1 if failures else 0



# ---------------------------------------------------------------------------
# CSF 2.0: one chunk per subcategory, with its function and category statements
# ---------------------------------------------------------------------------
CSF_FUNC = re.compile(r"^(GOVERN|IDENTIFY|PROTECT|DETECT|RESPOND|RECOVER) \((GV|ID|PR|DE|RS|RC)\):")
CSF_SUB = re.compile(r"^((?:GV|ID|PR|DE|RS|RC)\.[A-Z]{2}-\d{2}):\s*(.*)$", re.S)
# a category line starts with a bullet: "\u2022 Organizational Context (GV.OC): ..."
CSF_CAT = re.compile(r"^\u2022\s*(.+?) \(((?:GV|ID|PR|DE|RS|RC)\.[A-Z]{2})\):\s*(.*)$", re.S)


def chunk_csf(items, doc, title):
    funcs, cats, consumed = {}, {}, set()
    for idx, it in enumerate(items):
        t = it["text"]
        if CSF_SUB.match(t):
            continue
        mf = CSF_FUNC.match(t)
        if mf:
            funcs.setdefault(mf.group(2), t)
            consumed.add(idx)
            continue
        mc = CSF_CAT.match(t)
        if mc:
            cats.setdefault(mc.group(2), dict(name=mc.group(1).strip(), text=t.lstrip("\u2022 ").strip()))
            consumed.add(idx)
    entries, narrative = [], []
    for idx, it in enumerate(items):
        t = it["text"]
        ms = CSF_SUB.match(t)
        if ms:
            ident = ms.group(1)
            fkey, ckey = ident[:2], ident.split("-")[0]
            ftext, cat = funcs.get(fkey), cats.get(ckey)
            lines = ([ftext] if ftext else []) + ([cat["text"]] if cat else []) + [t]
            entries.append(make_chunk(
                doc, title, f"{doc}:{ident}", "entry", ident, "\n".join(lines), [it],
                f"{title}, subcategory {ident} ({cat['name'] if cat else ckey})",
                dict(function=fkey, category=ckey, function_found=bool(ftext), category_found=bool(cat))))
        elif idx not in consumed:
            narrative.append(it)
    return entries + pack(narrative, doc, title)


# ---------------------------------------------------------------------------
# SP 800-218A: one chunk per task (PO.2.1), with its practice and the notes under it
# ---------------------------------------------------------------------------
SS_TASK = re.compile(r"^((?:PO|PS|PW|RV)\.\d+\.\d+):\s*(.*)$", re.S)
SS_PRACTICE = re.compile(r"^(?:Informative References\s+)?(.+?) \(((?:PO|PS|PW|RV)\.\d+)\):\s*(.*)$", re.S)
PRIORITY = re.compile(r"^(High|Medium|Low)\b")
TASK_STOP = re.compile(r"^(Appendix\b|References\b|Glossary\b|Acknowledg)")
MAX_TASK_WORDS = 450       # the notes of one task are far shorter; more means text after the table


def chunk_ssdf(items, doc, title):
    entries, narrative, practice, current = [], [], None, None

    def close():
        nonlocal current
        if current:
            lines = ([practice["text"]] if practice else []) + [current["task"]] + current["parts"]
            priority = None
            for q in current["parts"]:
                m = PRIORITY.match(q)
                if m:
                    priority = m.group(1)
                    break
            refs = [q for q in current["parts"] if q.startswith("AI RMF:")]
            entries.append(make_chunk(
                doc, title, f"{doc}:{current['id']}", "entry", current["id"], "\n".join(lines),
                [current["item"]] + current["items"],
                f"{title}, task {current['id']}" + (f" ({practice['name']})" if practice else ""),
                dict(practice=practice["id"] if practice else None, priority=priority, refs=refs)))
        current = None

    for it in items:
        t = it["text"]
        mt = SS_TASK.match(t)
        if mt:
            close()
            current = dict(id=mt.group(1), task=t, item=it, parts=[], items=[])
            continue
        mp = SS_PRACTICE.match(t)
        if mp:
            close()
            # a column heading ("Informative References") is glued to the front of some practices
            practice = dict(id=mp.group(2), name=mp.group(1).strip(),
                            text=re.sub(r"^Informative References\s+", "", t))
            continue
        if current is not None and (TASK_STOP.match(t) or sum(words(q) for q in current["parts"])
                                    + words(t) > MAX_TASK_WORDS):
            close()                                     # the table has ended: the rest is ordinary text
        if current is not None:
            current["parts"].append(t)
            current["items"].append(it)
        else:
            narrative.append(it)
    close()
    return entries + pack(narrative, doc, title)


# ---------------------------------------------------------------------------
# OWASP Top 10 for LLM Applications: one chunk (or a few) per risk section
# ---------------------------------------------------------------------------
OW_RISK = re.compile(r"^LLM(\d\d)(?::2025)?:?\s+(.+)$", re.S)
OW_SECTION = re.compile(
    r"^(Description|Common Examples of (?:Vulnerability|Vulnerabilities|Risks?)|"
    r"Prevention and Mitigation Strategies|Example Attack Scenarios|Reference Links|"
    r"Related Frameworks and Taxonomies)$", re.I)
OW_END = re.compile(r"^Appendix\b")
OW_REFERENCE_SECTIONS = ("Reference Links", "Related Frameworks and Taxonomies")


def chunk_owasp(items, doc, title):
    entries, narrative, risks, risk = [], [], [], None
    for it in items:
        t = it["text"]
        mr = OW_RISK.match(t)
        if mr:
            risk = dict(id=f"LLM{mr.group(1)}", name=mr.group(2).strip(), head=it,
                        sections=[["Overview", []]])    # head keeps the heading as written
            risks.append(risk)
            continue
        if risk is not None and OW_END.match(t):
            risk = None                                  # the appendices are not part of a risk
        if risk is None:
            narrative.append(it)
        elif OW_SECTION.match(t):
            risk["sections"].append([re.sub(r"\s+", " ", t).strip(), []])
        else:
            risk["sections"][-1][1].append(it)
    for r in risks:
        for name, body in r["sections"]:
            if not body:
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            lead = dict(text=r["head"]["text"], pdf=r["head"]["pdf"], printed=r["head"]["printed"])
            sect = dict(text=name, pdf=body[0]["pdf"], printed=body[0]["printed"])
            entries += pack([lead, sect] + body, doc, title, overlap=0, kind="entry",
                            header=f"{title}, {r['id']} {r['name']}, {name}",
                            id_prefix=f"{doc}:{r['id']}/{slug}", section=f"{r['id']} {name}",
                            meta=dict(risk=r["id"], risk_name=r["name"], section_title=name,
                                      reference=name in OW_REFERENCE_SECTIONS))
    return entries + pack(narrative, doc, title)


# ---------------------------------------------------------------------------
# EU AI Act: one chunk (or a few) per article; annexes by block
# ---------------------------------------------------------------------------
EU_ART = re.compile(r"^Article (\d+[a-z]?)$")
EU_CHAPTER = re.compile(r"^CHAPTER [IVXLC]+$")
EU_SECTION = re.compile(r"^SECTION \d+$")
EU_ANNEX = re.compile(r"^ANNEX [IVXLC]+$")


def _title_like(text):
    w = text.split()
    return 0 < len(w) <= 25 and not re.search(r"[.;:]$", text) and not re.match(r"^(\d+\.|\()", text)


def chunk_eu(items, doc, title):
    entries, narrative, blocks = [], [], []
    chapter = section = cur = None
    i = 0
    while i < len(items):
        it = items[i]
        t = it["text"]
        nxt = items[i + 1] if i + 1 < len(items) else None
        has_title = bool(nxt and _title_like(nxt["text"]))
        if EU_CHAPTER.match(t):
            chapter = t + (" \u2014 " + nxt["text"] if has_title else "")
            section = cur = None
            i += 2 if has_title else 1
            continue
        if EU_SECTION.match(t):
            section = t + (" \u2014 " + nxt["text"] if has_title else "")
            cur = None
            i += 2 if has_title else 1
            continue
        ma, mx = EU_ART.match(t), EU_ANNEX.match(t)
        if ma or mx:
            cur = dict(kind="article" if ma else "annex",
                       key=f"Article {ma.group(1)}" if ma else f"Annex {t.split()[-1]}",
                       num=ma.group(1) if ma else t.split()[-1],
                       title=nxt["text"] if has_title else "", head=it, body=[], shown=t,
                       ctx=" | ".join(x for x in (chapter, section) if x) if ma else "")
            blocks.append(cur)
            i += 2 if has_title else 1
            continue
        (cur["body"] if cur is not None else narrative).append(it)
        i += 1
    for b in blocks:
        label = f"{b['key']}: {b['title']}" if b["title"] else b["key"]
        shown = f"{b['shown']}: {b['title']}" if b["title"] else b["shown"]      # as the document writes it
        where = dict(pdf=b["head"]["pdf"], printed=b["head"]["printed"])
        lead = ([dict(text=b["ctx"], **where)] if b["ctx"] else []) + [dict(text=shown, **where)]
        entries += pack(lead + b["body"], doc, title, overlap=0, kind="entry",
                        header=f"{title}, {label}", id_prefix=f"{doc}:{b['key']}", section=b["key"],
                        meta=dict(article=b["num"] if b["kind"] == "article" else None,
                                  annex=b["num"] if b["kind"] == "annex" else None,
                                  article_title=b["title"], context=b["ctx"]))
    return entries + pack(narrative, doc, title)


# ---------------------------------------------------------------------------
# Documents without IDs (AI 100-2, Deploying AI, AI Data Security, UK Playbook):
# ordinary packed text
# ---------------------------------------------------------------------------
def chunk_narrative(items, doc, title):
    return pack(items, doc, title)


# ---------------------------------------------------------------------------
# Figure captions: one chunk per figure (a very long caption is split like any text)
# ---------------------------------------------------------------------------
def clean_caption(text):
    """The captions are written in light markdown. Remove the markup, keep the words."""
    lines = []
    for line in text.replace("**", "").split("\n"):
        line = re.sub(r"^\s*[\*\-\u2022]\s+", "- ", line.rstrip())
        if line.strip():
            lines.append(line.strip())
    return lines


def run_figures(root):
    cap_dir = root / "captions"
    latest = {}
    for path in sorted(cap_dir.glob("*.json")):                    # production captions only
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if rec.get("purpose", "production") != "production" or not rec.get("text"):
            continue
        name = rec["name"]
        if name not in latest or rec.get("created", "") > latest[name].get("created", ""):
            latest[name] = rec                                     # the newest caption of each figure
    if not latest:
        print(f"no captions found in {cap_dir}")
        return 2
    printed = {}
    chunks = []
    for name in sorted(latest):
        rec = latest[name]
        doc, page = name.split("_")[0], rec["pdf_page"]
        if doc not in printed:
            cleaned = root / "cleaned" / f"{doc}.json"
            printed[doc] = ({p["pdf_page"]: p["printed_page"] for p in
                             json.loads(cleaned.read_text(encoding="utf-8"))["pages"]}
                            if cleaned.exists() else {})
        title = DOCS[doc]["title"]
        items = [dict(text=line, pdf=page, printed=printed[doc].get(page)) for line in clean_caption(rec["text"])]
        chunks += pack(items, doc, title, overlap=0, kind="figure", id_prefix=f"fig:{name}",
                       header=f"{title}, figure on PDF page {page}", section=f"figure, PDF page {page}",
                       meta=dict(figure=name, model=rec.get("model"), used_fallback=rec.get("used_fallback")))
    out = root / "chunks"
    out.mkdir(exist_ok=True)
    (out / "figures.json").write_text(json.dumps(
        dict(doc="figures", title="figure captions",
             created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
             params=dict(max_words=MAX_WORDS), chunks=chunks), ensure_ascii=False, indent=2), encoding="utf-8")
    w = [c["words"] for c in chunks]
    print(f"figures: {len(chunks)} chunks for {len(latest)} captions; words: min {min(w)}, "
          f"median {int(statistics.median(w))}, max {max(w)}")
    print("  wrote chunks/figures.json")
    return 0


CHUNKERS = {"rmf": chunk_rmf, "gai": chunk_gai, "csf": chunk_csf, "ssdf": chunk_ssdf,
            "owasp": chunk_owasp, "eu": chunk_eu, "aml": chunk_narrative, "deploy": chunk_narrative,
            "data": chunk_narrative, "uk": chunk_narrative}


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def run(key, root):
    src = root / "cleaned" / f"{key}.json"
    if not src.exists():
        print(f"{src} not found. Run: python clean_text.py all")
        return 2
    data = json.loads(src.read_text(encoding="utf-8"))
    info = DOCS[key]
    chunks = CHUNKERS[key](list(stream(data["pages"])), key, info["title"])
    out = root / "chunks"
    out.mkdir(exist_ok=True)
    (out / f"{key}.json").write_text(json.dumps(
        dict(doc=key, title=info["title"], created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
             params=dict(max_words=MAX_WORDS, overlap_sentences=OVERLAP_SENTENCES), chunks=chunks),
        ensure_ascii=False, indent=2), encoding="utf-8")
    entries = [c for c in chunks if c["kind"] == "entry"]
    narr = [c for c in chunks if c["kind"] == "narrative"]
    print(f"{key}: {len(chunks)} chunks = {len(entries)} entry + {len(narr)} narrative")
    for label, group in (("entry", entries), ("narrative", narr)):
        if group:
            w = [c["words"] for c in group]
            print(f"  {label:9s} words: min {min(w)}, median {int(statistics.median(w))}, max {max(w)}")
    if key == "gai":
        no_tags = [c["id"] for c in entries if not c["meta"]["risks"]]
        no_action = [c["id"] for c in entries if not c["meta"]["has_action_text"]]
        print(f"  entries without recognised risk tags: {len(no_tags)}; without action text: {len(no_action)}")
    print(f"  wrote chunks/{key}.json")
    return 0


def show(key, root, needle):
    path = root / "chunks" / f"{key}.json"
    if not path.exists():
        print(f"{path} not found. Run: python chunker.py {key}")
        return 2
    hits = [c for c in json.loads(path.read_text(encoding="utf-8"))["chunks"] if needle in c["id"]]
    if not hits:
        print(f"no chunk id contains {needle!r}")
        return 1
    for c in hits[:3]:
        print(f"===== {c['id']} | {c['kind']} | PDF p{c['pdf_start']}-{c['pdf_end']} "
              f"(printed {c['printed_start']}-{c['printed_end']}) | {c['words']} words")
        print("--- embed_text header:", c["embed_text"].split("\n")[0])
        print("--- text:")
        print(c["text"])
        print("--- meta:", c["meta"])
        print()
    return 0


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Cut cleaned documents into chunks.")
    ap.add_argument("key", nargs="?", help="a document key (rmf, gai, csf, ssdf, owasp, eu, aml, deploy, data, uk), figures, or all")
    ap.add_argument("--selftest", action="store_true", help="run the built-in tests")
    ap.add_argument("--show", metavar="TEXT", help="print chunks whose id contains TEXT")
    ap.add_argument("--dir", help="project folder (default: this script's folder)")
    args = ap.parse_args(argv)
    root = Path(args.dir) if args.dir else Path(__file__).resolve().parent
    if args.selftest:
        return selftest()
    if not args.key:
        ap.print_help()
        return 0
    keys = (list(CHUNKERS) + ["figures"]) if args.key == "all" else [args.key]
    worst = 0
    for k in keys:
        if k not in CHUNKERS and k != "figures":
            print(f"No chunker for '{k}'. Available: {', '.join(CHUNKERS)}, figures")
            return 2
        if args.show:
            worst = max(worst, show(k, root, args.show))
        else:
            worst = max(worst, run_figures(root) if k == "figures" else run(k, root))
    return worst


if __name__ == "__main__":
    sys.exit(main())