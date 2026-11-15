#!/usr/bin/env python3
"""
pages_to_images.py - caption the figure pages of the evaluation PDFs with Gemini,
once, and keep every result on disk so re-running never spends quota twice.

Your RAG system cannot search an image, so the figures reach it only through these
captions. That is the multimodal step.

Run from the project folder (or pass --dir):

  python pages_to_images.py --list                   show every job name and its settings
  python pages_to_images.py --all-figures --dry-run  render the 21 figure images only
                                                     (NO API calls) so you can look at them
  python pages_to_images.py --only csf_p15           caption one figure
  python pages_to_images.py --all-figures            caption all 21 figures (cached ones skipped)
  python pages_to_images.py --tests                  run the diagnostic test jobs
  python pages_to_images.py --force ...              ignore the cache and ask again

Where results go:
  captions\\            production captions: the ONLY folder your indexer should read
  captions\\test\\       diagnostic runs, kept apart so they can never reach the index

Each caption is saved as <job>__<key>.json (caption + every setting that produced it) and
<job>__<key>.png (the exact image the model saw). The key changes automatically when the
PDF's SHA-256, page, dpi, rotation, crop or prompt changes, so an old caption is never reused
for a different file or setting.
"""
import argparse
import hashlib
import io
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pymupdf
from PIL import Image
from google import genai
from google.genai import errors, types

# ----------------------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------------------
# The first model is the PINNED captioner. The others are used only when it fails, and every
# caption records which model made it and whether a fallback was used. The order never
# changes during a run, so one hiccup cannot quietly switch the whole run to another model.
MODELS = ["gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.8-flash"]

# MODELS = ["gemini-3.5-flash-lite"]

TIMEOUT_MS = 180_000             # per request
MAX_ATTEMPTS_PER_MODEL = 3
RATE_LIMIT_WAIT_S = 30           # base wait after a 429 (grows with each attempt)
MAX_OUTPUT_TOKENS = 8192         # room for the answer (and any model thinking)
MAX_CONSECUTIVE_FAILURES = 2     # stop the run after this many failed jobs in a row
DEFAULT_DPI = 300
MAX_IMAGE_BYTES = 18_000_000     # inline image requests are limited in size
SAVE_IMAGES = True               # keep the PNG that was sent, to check what the model saw
STRICT_HASH = True               # stop on a PDF whose SHA-256 does not match the table below

# key -> (file name in the project folder, first 16 hex characters of its SHA-256)
FILES = {
    "rmf":   ("NIST.AI.100-1.pdf",                                  "7576edb531d98488"),
    "aml":   ("NIST.AI.100-2e2025.pdf",                             "4811fb6ad73f9c91"),
    "csf":   ("NIST.CSWP.29.pdf",                                   "3c31f46fee98cac0"),
    "uk":    ("AI_Playbook_for_the_UK_Government__12_02_.pdf",      "3417156b104aa32e"),
    "owasp": ("OWASP-Top-10-for-LLMs-v2025.pdf",                             "eee39216f0e6750f"),
}

# The 21 figures: file key -> PDF page numbers (1-based, as your PDF viewer shows them).
FIGURES = {
    "rmf":   [10, 15, 16, 17, 25],
    "aml":   [17, 48, 49, 50, 51, 52, 60, 68],
    "csf":   [8, 10, 11, 13, 15, 18],
    "uk":    [17],
    "owasp": [43],
}

# Pages that come out SIDEWAYS in the rendered image need a clockwise turn (90, 180 or 270).
# Do NOT guess: run with --dry-run, open the PNGs, and add an entry only for a page that
# really is sideways. Pages not listed here are not rotated.
# (rmf p16 was reported as a rotated figure page; confirm it, and remove the entry if the
#  image is already upright.)
ROTATE_CW = {
    ("rmf", 16): 90,
    ("owasp", 43): 90,
}

# One generic prompt for every figure. It must NOT contain anything taken from the test
# questions: captions are made before anyone asks a question.
FIGURE_PROMPT = (
    "This image is a page that contains a figure or diagram. Ignore page headers, "
    "footers, page numbers and ordinary body-text paragraphs.\n"
    "1. Figure title: copy the figure's own title or caption line if one is shown "
    "(for example a line starting 'Figure' or 'Fig.'); otherwise write 'none'.\n"
    "2. Labels: list every text label that is inside the figure itself, copied exactly "
    "as written. Do not include the figure title here.\n"
    "3. Layout: describe the layout and which labels are grouped together, in 3 to 6 "
    "sentences.\n"
    "4. Flow: if the figure shows a clear overall direction of flow or sequence, describe "
    "it in at most 3 sentences, naming the labels it passes through in order. Do not list "
    "individual arrows and do not describe arrowheads you cannot clearly see. If the "
    "direction is not clear, write 'flow unclear'.\n"
    "If a label or any character is unreadable, write [unreadable] instead of guessing. "
    "For codes and numbers, copy them character by character and add (uncertain) after any "
    "that is not perfectly clear. Do not add labels that are not visible."
)


# ----------------------------------------------------------------------------------------
# JOBS
# ----------------------------------------------------------------------------------------
def make_job(name, file, page, dpi=DEFAULT_DPI, prompt=FIGURE_PROMPT, crop=None,
             purpose="production"):
    """crop = (left, top, right, bottom) as fractions of the image AFTER rotation."""
    return dict(name=name, file=file, page=page, dpi=dpi, prompt=prompt, crop=crop,
                rotate_cw=ROTATE_CW.get((file, page), 0), purpose=purpose)


def production_jobs():
    return [make_job(f"{key}_p{page}", key, page)
            for key, pages in FIGURES.items() for page in pages]


# Diagnostic jobs: they answer "does a setting change help?", and are saved apart from the
# production captions. Never write a prompt here that names things from the test questions,
# and never tune a crop to a question: if a fix is needed, it must apply to every dense page.
TEST_JOBS = [
    make_job("test_owasp_p43_600dpi", "owasp", 43, dpi=600, purpose="test"),
]


def all_jobs():
    return production_jobs() + TEST_JOBS


def outdir_for(job, base):
    d = base / "captions"
    if job["purpose"] == "test":
        d = d / "test"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------------------
# ERRORS
# ----------------------------------------------------------------------------------------
class FatalError(Exception):
    """Retrying cannot help (bad key, no permission). Stop the whole run."""


class JobError(Exception):
    """This one job cannot succeed as it is. Record it as failed and go on to the next."""


class WrongFile(Exception):
    """The PDF in the folder is not the version the answer key was written against."""


class EmptyResponse(Exception):
    """The model returned no text (blocked or empty)."""


# ----------------------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_key(job, pdf_sha):
    payload = json.dumps(
        dict(pdf=pdf_sha, page=job["page"], dpi=job["dpi"], rot=job["rotate_cw"],
             crop=job["crop"], prompt=job["prompt"]),
        sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_json(path, record):
    """Write to a temporary file, then rename it, so a crash never leaves a half-written file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def find_cached(outdir, key):
    """Return the saved record for this key, or None. Unreadable files count as a miss."""
    for path in sorted(outdir.glob(f"*__{key[:10]}.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print(f"  note: ignoring unreadable cache file {path.name}")
            continue
        if rec.get("key") == key and rec.get("text"):
            return rec
    return None


def render(job, pdf_path):
    """Render one page, optionally rotate it upright and crop it. Returns (png bytes, (w, h))."""
    doc = pymupdf.open(pdf_path)
    page = doc[job["page"] - 1]
    pix = page.get_pixmap(dpi=job["dpi"])
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    if job["rotate_cw"]:
        img = img.rotate(-job["rotate_cw"], expand=True)      # PIL turns counter-clockwise
    if job["crop"]:
        w, h = img.size
        l, t, r, b = job["crop"]
        img = img.crop((int(w * l), int(h * t), int(w * r), int(h * b)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    if len(data) > MAX_IMAGE_BYTES:
        raise JobError(f"image is {len(data) / 1e6:.1f} MB, too large to send; lower the dpi")
    return data, img.size


def prepare(job, base, sha_cache):
    fname, expected = FILES[job["file"]]
    pdf = base / fname
    if not pdf.exists():
        raise FileNotFoundError(f"{pdf} not found")
    if fname not in sha_cache:
        sha_cache[fname] = sha256_file(pdf)
    sha = sha_cache[fname]
    if not sha.startswith(expected):
        msg = (f"{fname}: SHA-256 starts {sha[:16]} but the answer key was written for "
               f"{expected}. This is a different version of the file.")
        if STRICT_HASH:
            raise WrongFile(msg)
        print("  WARNING:", msg)
    return pdf, sha, make_key(job, sha)


def finish_reason_of(response):
    try:
        fr = response.candidates[0].finish_reason
        return getattr(fr, "name", str(fr))
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------------------
# API CALL WITH RETRIES AND MODEL FALLBACK
# ----------------------------------------------------------------------------------------
def ask(client, contents, config, models):
    """Try each model in order, always starting from the pinned one.
    5xx / timeout / empty -> back off and retry; 429 -> wait longer and retry;
    404 -> skip the model; 401/403 -> stop the run; other 4xx -> this job fails."""
    for model in models:
        for attempt in range(1, MAX_ATTEMPTS_PER_MODEL + 1):
            t0 = time.time()
            try:
                print(f"  -> {model} (attempt {attempt}/{MAX_ATTEMPTS_PER_MODEL})", flush=True)
                r = client.models.generate_content(model=model, contents=contents, config=config)
                text = (r.text or "").strip()
                if not text:
                    raise EmptyResponse("no text returned")
                print(f"  answered in {time.time() - t0:.0f}s", flush=True)
                return model, text, finish_reason_of(r)
            except errors.ClientError as e:
                if e.code == 429:
                    wait = RATE_LIMIT_WAIT_S * attempt
                    print(f"  {model}: quota / rate limit (429), waiting {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                if e.code == 404:
                    print(f"  {model}: not available to this key (404), skipping it", flush=True)
                    break
                if e.code in (401, 403):
                    raise FatalError(f"{model}: HTTP {e.code} {e.status}: {e.message}") from e
                raise JobError(f"{model}: HTTP {e.code} {e.status}: {e.message}") from e
            except (errors.ServerError, httpx.TransportError, EmptyResponse) as e:
                wait = 2 ** attempt + random.random()
                code = getattr(e, "code", "")
                msg = str(getattr(e, "message", e))[:200].replace("\n", " ")
                print(f"  {model} failed after {time.time() - t0:.0f}s "
                      f"({type(e).__name__} {code}: {msg}), retrying in {wait:.0f}s",
                      flush=True)
                time.sleep(wait)
        print(f"  {model} gave up, trying the next model", flush=True)
    raise RuntimeError("All models unavailable. Try again later.")


# ----------------------------------------------------------------------------------------
# ONE JOB
# ----------------------------------------------------------------------------------------
def run_job(job, base, client, config, models, force, dry_run, sha_cache):
    outdir = outdir_for(job, base)
    pdf, sha, key = prepare(job, base, sha_cache)
    stem = f"{job['name']}__{key[:10]}"

    if not dry_run and not force:
        hit = find_cached(outdir, key)
        if hit:
            return "cached", hit

    png, size = render(job, pdf)
    if SAVE_IMAGES or dry_run:
        (outdir / f"{stem}.png").write_bytes(png)
    print(f"  image: {size[0]}x{size[1]} px, {len(png) / 1e6:.2f} MB")
    if dry_run:
        return "dry-run", {"text": f"(saved {stem}.png, no API call)"}

    contents = [types.Part.from_bytes(data=png, mime_type="image/png"), job["prompt"]]
    model, text, finish = ask(client, contents, config, models)
    if finish == "MAX_TOKENS":
        raise JobError("the caption was cut off (MAX_TOKENS); raise MAX_OUTPUT_TOKENS")

    record = dict(
        key=key, name=job["name"], purpose=job["purpose"], file=pdf.name, pdf_sha256=sha,
        pdf_page=job["page"], dpi=job["dpi"], rotate_cw=job["rotate_cw"], crop=job["crop"],
        prompt=job["prompt"], model=model, pinned_model=models[0],
        used_fallback=(model != models[0]), finish_reason=finish,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        image_px=list(size), text=text)
    save_json(outdir / f"{stem}.json", record)
    return "new", record


# ----------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------
def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Caption figure pages with Gemini, with a disk cache.")
    ap.add_argument("--dir", help="folder that holds the PDFs (default: this script's folder)")
    ap.add_argument("--only", help="comma-separated job names to run")
    ap.add_argument("--all-figures", action="store_true", help="run the 21 production figures")
    ap.add_argument("--tests", action="store_true", help="run the diagnostic test jobs")
    ap.add_argument("--dry-run", action="store_true", help="render images only, no API calls")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    ap.add_argument("--list", action="store_true", help="list job names and exit")
    args = ap.parse_args(argv)

    base = Path(args.dir) if args.dir else Path(__file__).resolve().parent

    if args.list:
        for j in all_jobs():
            print(f"{j['name']:24s} {j['purpose']:10s} {FILES[j['file']][0]}  PDF p.{j['page']}"
                  f"  dpi {j['dpi']}  rotate {j['rotate_cw']}  crop {j['crop']}")
        return 0

    if args.only:
        wanted = [n.strip() for n in args.only.split(",") if n.strip()]
        known = {j["name"]: j for j in all_jobs()}
        unknown = [n for n in wanted if n not in known]
        if unknown:
            print("Unknown job name(s):", ", ".join(unknown), "\nUse --list to see the names.")
            return 2
        jobs = [known[n] for n in wanted]
    else:
        jobs = []
        if args.all_figures:
            jobs += production_jobs()
        if args.tests:
            jobs += TEST_JOBS
        if not jobs:
            print("Nothing selected. Use --all-figures, --tests or --only NAME "
                  "(add --dry-run to render images without calling the API).")
            return 0

    # Plan: which jobs would spend quota?
    sha_cache, status = {}, {}
    for j in jobs:
        try:
            _, _, key = prepare(j, base, sha_cache)
            hit = find_cached(outdir_for(j, base), key) if not args.force else None
            status[j["name"]] = "cached" if hit else "needs API"
        except (FileNotFoundError, WrongFile) as e:
            status[j["name"]] = f"BLOCKED: {e}"
    need = sum(1 for s in status.values() if s == "needs API")
    print(f"{len(jobs)} job(s): {need} need an API call, "
          f"{sum(1 for s in status.values() if s == 'cached')} cached, "
          f"{sum(1 for s in status.values() if s.startswith('BLOCKED'))} blocked.")
    for name, s in status.items():
        if s.startswith("BLOCKED"):
            print(f"  {name}: {s}")
    if args.dry_run:
        print("(dry run: no API calls will be made)")

    client, config, models = None, None, list(MODELS)
    if not args.dry_run and need:
        try:
            client = genai.Client(http_options=types.HttpOptions(timeout=TIMEOUT_MS))
        except ValueError as e:
            print("Could not create the Gemini client:", e)
            return 2
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))

    results, fallbacks, failures_in_a_row = [], [], 0
    for j in jobs:
        print(f"\nJob: {j['name']}  ({FILES[j['file']][0]}, PDF page {j['page']})", flush=True)
        t0 = time.time()
        try:
            state, rec = run_job(j, base, client, config, models,
                                 args.force, args.dry_run, sha_cache)
        except FatalError as e:
            print("  FATAL:", e, "\n  Stopping: retrying will not help. Check the key and "
                  "its permissions.")
            results.append((j["name"], "FATAL", "", time.time() - t0))
            break
        except Exception as e:                      # one bad job must not lose the others
            print(f"  FAILED: {type(e).__name__}: {e}")
            results.append((j["name"], "FAILED", "", time.time() - t0))
            failures_in_a_row += 1
            if failures_in_a_row >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n{failures_in_a_row} jobs in a row failed. Stopping so a quota or "
                      "outage problem does not waste the session. Run again later: finished "
                      "captions are cached.")
                break
            continue
        if state == "new":
            failures_in_a_row = 0
        print("=" * 60)
        model = rec.get("model", "")
        print(f"{j['name']} | {state}" + (f" | model: {model}" if model else ""))
        print(rec["text"])
        if state == "new" and rec.get("used_fallback"):
            fallbacks.append(f"{j['name']} ({model})")
        results.append((j["name"], state, model, time.time() - t0))

    print("\nSUMMARY")
    for name, state, model, secs in results:
        print(f"  {name:24s} {state:8s} {model:18s} {secs:5.0f}s")
    if fallbacks:
        print("\nFallback models made these captions (not the pinned model):")
        for item in fallbacks:
            print("  ", item)
    ok = all(s in ("new", "cached", "dry-run") for _, s, _, _ in results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())