import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pages_to_images_2 as m   # only reads the job list; does not run anything

CAPS = Path("captions")

# python caption_status.py NAME  -> print one caption in full
if len(sys.argv) > 1:
    for p in sorted(CAPS.glob(f"{sys.argv[1]}__*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        print(r["name"], "|", r["model"], "|", r.get("finish_reason"))
        print(r["text"])
    sys.exit(0)

expected = [j["name"] for j in m.production_jobs()]
saved = {}
for p in sorted(CAPS.glob("*.json")):
    try:
        r = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        print("unreadable file:", p.name)
        continue
    saved[r["name"]] = r

print(f"{'figure':10s} {'model':18s} {'fallback':9s} {'finish':8s} {'chars':>6s}  flags")
for name in expected:
    r = saved.get(name)
    if not r:
        continue
    text = r.get("text", "")
    flags = [w for w in ("[unreadable]", "(uncertain)", "unclear") if w in text]
    print(f"{name:10s} {r['model']:18s} {str(r.get('used_fallback')):9s} "
          f"{str(r.get('finish_reason')):8s} {len(text):6d}  {','.join(flags) or '-'}")

missing = [n for n in expected if n not in saved]
print(f"\n{len(saved)} of {len(expected)} captioned. Still missing: {', '.join(missing) or 'none'}")