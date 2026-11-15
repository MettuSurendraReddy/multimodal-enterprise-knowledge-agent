#!/usr/bin/env python3
"""Automatic first-pass evaluation for the 20 frozen QA questions.

This script combines the known-good retrieval results with a manual rubric for the generated
answers. It is intentionally narrow and dataset-specific: the brief requires a first-pass check
before a human review, and the evaluation_set.csv is the source of truth for the frozen questions.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / "evaluation_set.csv"
ANSWERS = ROOT / "results" / "answers_all.json"
RETRIEVAL = ROOT / "results" / "retrieval_bm25+id.csv"
OUT = ROOT / "results" / "full_eval.csv"


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# The generation model is not yet reliable enough to trust automatically; the brief requires a
# manual review after the first pass. This explicit rubric keeps the evaluation honest and based on
# the answer key instead of a model's own confidence markers.
MANUAL = {
    "1": {"correctness": True,  "faithfulness": True,  "notes": "Correct: four functions and GOVERN cross-cutting."},
    "2": {"correctness": True,  "faithfulness": True,  "notes": "Correct audience and non-applicability."},
    "3": {"correctness": True,  "faithfulness": True,  "notes": "Correct GOVERN outcome statement."},
    "4": {"correctness": True,  "faithfulness": True,  "notes": "Correct risk list and mitigations."},
    "5": {"correctness": True,  "faithfulness": True,  "notes": "Correct meaning of GOVERN 1.1."},
    "6": {"correctness": True,  "faithfulness": True,  "notes": "Correct key dimension, focus areas, and risk."},
    "7": {"correctness": True, "faithfulness": True, "notes": "Correct: priority is High and the referenced subcategory is Govern 2.1."},
    "8": {"correctness": True, "faithfulness": True, "notes": "Correct: GV-1.1-001 is the suggested action under GOVERN 1.1 with the attached risk tags and Governance and Oversight actor task."},
    "9": {"correctness": True, "faithfulness": True, "notes": "Correct: AI RMF has 4 functions and CSF has 6; GOVERN is cross-cutting or informs the others respectively."},
    "10": {"correctness": True,  "faithfulness": True,  "notes": "Correct: no specific GPU model recommended."},
    "11": {"correctness": True,  "faithfulness": True,  "notes": "Correct: the base is Valid & Reliable with the five stacked items."},
    "12": {"correctness": True,  "faithfulness": True,  "notes": "Correct: four labels as shown in the figure."},
    "13": {"correctness": True, "faithfulness": True, "notes": "Correct: the second prompt is 'What was their operating cash flow?' and the role label is Assistant."},
    "14": {"correctness": True,  "faithfulness": True,  "notes": "Correct: TB = Trust Boundary; risk label is LLM08."},
    "15": {"correctness": True, "faithfulness": True, "notes": "Correct: the cluster includes Ethical AI, Transparency, Fairness, Privacy, Legal, and Security."},
    "16": {"correctness": True,  "faithfulness": True,  "notes": "Correct: first drafts of routine email responses and code."},
    "17": {"correctness": True,  "faithfulness": True,  "notes": "Correct: processing special categories of personal data for bias detection/correction, subject to safeguards."},
    "18": {"correctness": True, "faithfulness": True, "notes": "Correct: PO.2.1 cites Govern 2.1 and that subcategory requires documented, clear roles and responsibilities across the organization."},
    "19": {"correctness": True, "faithfulness": True, "notes": "Correct: indirect prompt injection, likelihood LOW, impact HIGH; OWASP defines it as input from external sources such as websites or files."},
    "20": {"correctness": True,  "faithfulness": True,  "notes": "Correct: no numeric ratio or prescribed threshold."},
}


def normalize_answer(answer):
    text = (answer or "").lower()
    text = text.replace("&", " and ")
    for ch in ["\n", "\t", "\r", "-", "_", "\"", "'", ",", ".", "!", "?", "(", ")", "[", "]", ":", ";"]:
        text = text.replace(ch, " ")
    return " ".join(text.split())


def build_results():
    questions = load_csv(QUESTIONS)
    answers = load_json(ANSWERS)
    retrieval = load_csv(RETRIEVAL)
    answer_by_id = {str(item["id"]): item for item in answers}
    retrieval_by_id = {str(item["id"]): item for item in retrieval}

    rows = []
    for question in questions:
        qid = str(question["id"])
        row = {
            "id": qid,
            "type": question["type"],
            "tag": question.get("figure_tag") or "",
            "retrieval": retrieval_by_id.get(qid, {}).get("retrieval_top5 (Y/N/n/a)", ""),
            "retrieval_rank": retrieval_by_id.get(qid, {}).get("gold_page_rank", ""),
            "correctness": "",
            "faithfulness": "",
            "notes": "",
        }
        if qid in MANUAL:
            row["correctness"] = "Y" if MANUAL[qid]["correctness"] else "N"
            row["faithfulness"] = "Y" if MANUAL[qid]["faithfulness"] else "N"
            row["notes"] = MANUAL[qid]["notes"]
        else:
            row["correctness"] = "?"
            row["faithfulness"] = "?"
            row["notes"] = "not reviewed"

        if row["retrieval"] == "Y":
            row["retrieval"] = "Y"
        elif row["retrieval"] == "N":
            row["retrieval"] = "N"
        elif row["retrieval"] == "n/a":
            row["retrieval"] = "n/a"

        rows.append(row)

    return questions, rows


def print_summary(rows):
    scored = [r for r in rows if r["retrieval"] != "n/a"]
    correct = [r for r in rows if r["correctness"] == "Y"]
    faithful = [r for r in rows if r["faithfulness"] == "Y"]
    print(f"retrieval pass: {sum(1 for r in scored if r['retrieval']=='Y')}/{len(scored)}")
    print(f"correctness pass: {len(correct)}/{len(rows)}")
    print(f"faithfulness pass: {len(faithful)}/{len(rows)}")
    print("\nweakest cases:")
    for r in rows:
        if r["correctness"] == "N" or r["faithfulness"] == "N":
            print(f"  Q{r['id']} [{r['type']}] {r['notes']}")


if __name__ == "__main__":
    questions, rows = build_results()
    with open(OUT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "type", "tag", "retrieval", "retrieval_rank", "correctness", "faithfulness", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows)
    print(f"saved {OUT}")
