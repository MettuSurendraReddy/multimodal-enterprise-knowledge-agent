import time
import httpx 
import pymupdf
from google import genai
from google.genai import types, errors

MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash"]

JOBS = [
    {   # Q14 retest: higher resolution, focused prompt
        "name": "owasp_appendix1",
        "pdf": r"C:\Users\METTU SURENDRA REDDY\OneDrive\Desktop\Multimodal Agent Project\OWASP-Top-10-for-LLMs-v2025.pdf",
        "page_index": 42,
        "dpi": 300,
        "prompt": (
            "This page contains a diagram. Ignore page headers, footers and "
            "author lines. Find the label 'Grounding/URL Scraping' and the "
            "legend entry that explains 'TB'. For the box or boxes next to "
            "'Grounding/URL Scraping', copy the text exactly, character by "
            "character, including every number. If any digit is unclear, "
            "say so instead of guessing."
        ),
    },
    {   # Q15: Fields of AI diagram
        "name": "uk_fields_of_ai",
        "pdf": r"C:\Users\METTU SURENDRA REDDY\OneDrive\Desktop\Multimodal Agent Project\AI_Playbook_for_the_UK_Government__12_02_.pdf",
        "page_index": 16,
        "dpi": 200,
        "prompt": (
            "This page contains a diagram. Ignore page headers and footers. "
            "List every text label in the diagram, copied exactly as written. "
            "Then describe which labels are grouped together and how they are "
            "linked to each other. If a label is unreadable, say so. "
            "Do not guess or add labels that are not visible."
        ),
    },
]

# timeout is in milliseconds: 120000 = 2 minutes per request
client = genai.Client(http_options=types.HttpOptions(timeout=120_000))

import httpx   # add this to the imports at the top

MODELS = ["gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.8-flash"]

def ask(contents):
    for model in MODELS:
        for attempt in range(3):
            start = time.time()
            try:
                print(f"  sending to {model} (attempt {attempt + 1})...", flush=True)
                r = client.models.generate_content(model=model, contents=contents)
                print(f"  answered in {time.time() - start:.0f}s", flush=True)
                return model, r.text
            except (errors.ServerError, httpx.TimeoutException) as e:
                wait = 2 ** attempt
                print(f"  {model} failed after {time.time() - start:.0f}s "
                      f"({type(e).__name__}), retrying in {wait}s", flush=True)
                time.sleep(wait)
        print(f"  {model} gave up, trying the next model", flush=True)
    raise RuntimeError("All models unavailable. Try again later.")

for job in JOBS:
    print(f"Job: {job['name']}", flush=True)
    doc = pymupdf.open(job["pdf"])
    png = doc[job["page_index"]].get_pixmap(dpi=job["dpi"]).tobytes("png")
    print(f"  image size: {len(png) / 1_000_000:.1f} MB", flush=True)
    contents = [types.Part.from_bytes(data=png, mime_type="image/png"), job["prompt"]]
    model_used, text = ask(contents)
    print("=" * 60)
    print(job["name"], "| answered by:", model_used)
    print(text)