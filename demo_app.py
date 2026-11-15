#!/usr/bin/env python3
"""Tiny local web demo for the grounded RAG project.

Run:
    python demo_app.py
Then open http://localhost:8000/
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import html
import json
import os
from pathlib import Path

from answer_generation import load_chunks, retrieve_question_chunks, generate_answer

ROOT = Path(__file__).resolve().parent
CHUNKS = load_chunks()
OG_IMAGE_PATH = ROOT / "assets" / "linkedin-project-overview.png"

HTML_PAGE = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>AI Governance RAG Demo</title>
  <meta name=\"description\" content=\"A multimodal RAG demo that retrieves evidence from AI governance documents and generates cited answers.\" />
  <meta property=\"og:type\" content=\"website\" />
  <meta property=\"og:title\" content=\"AI Governance Knowledge Agent\" />
  <meta property=\"og:description\" content=\"Explore a multimodal RAG project for retrieving evidence and generating cited answers from AI governance documents.\" />
  <meta property=\"og:url\" content=\"https://multimodal-enterprise-knowledge-agent.onrender.com/\" />
  <meta property=\"og:image\" content=\"https://multimodal-enterprise-knowledge-agent.onrender.com/og-image.png\" />
  <meta property=\"og:image:secure_url\" content=\"https://multimodal-enterprise-knowledge-agent.onrender.com/og-image.png\" />
  <meta property=\"og:image:type\" content=\"image/png\" />
  <meta property=\"og:image:width\" content=\"1200\" />
  <meta property=\"og:image:height\" content=\"1500\" />
  <meta property=\"og:image:alt\" content=\"Infographic showing the AI Governance Knowledge Agent pipeline and benchmark results.\" />
  <meta name=\"twitter:card\" content=\"summary_large_image\" />
  <style>
    :root {
      --bg: #f3f7ff;
      --panel: #ffffff;
      --panel-soft: #eef4ff;
      --line: #dfe9fb;
      --ink: #15233b;
      --muted: #5a6a89;
      --primary: #2347d5;
      --primary-strong: #1834a6;
      --success: #eafaf1;
      --success-border: #b7e7cb;
      --error-bg: #fff1f1;
      --error-border: #f1c2c2;
      --shadow: 0 12px 28px rgba(26, 42, 78, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: linear-gradient(180deg, #edf3ff 0%, #f9fbff 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 980px;
      margin: 0 auto;
      padding: 40px 20px 60px;
    }
    .hero {
      background: linear-gradient(145deg, #eef4ff 0%, #ffffff 100%);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 26px 28px 18px;
      margin-bottom: 20px;
    }
    .eyebrow {
      display: inline-block;
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--primary);
      font-weight: 700;
      background: #eaf0ff;
      border-radius: 999px;
      padding: 7px 10px;
      margin-bottom: 12px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3rem);
      line-height: 1.12;
    }
    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 1.02rem;
      line-height: 1.6;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
      padding: 22px;
    }
    form { margin-top: 12px; }
    textarea {
      width: 100%;
      min-height: 136px;
      font-size: 16px;
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid #c9d7f5;
      background: #fbfcff;
      color: var(--ink);
      resize: vertical;
    }
    textarea:focus {
      outline: 3px solid rgba(35, 71, 213, 0.15);
      border-color: var(--primary);
    }
    .actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 16px;
    }
    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .example-btn, button {
      border: 1px solid #cfe0ff;
      background: #edf3ff;
      color: var(--primary-strong);
      font-weight: 700;
      border-radius: 999px;
      padding: 8px 12px;
      cursor: pointer;
      transition: transform 0.12s ease, background 0.12s ease;
    }
    .example-btn:hover, button:hover {
      transform: translateY(-1px);
      background: #e0ebff;
    }
    button.submit {
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-strong) 100%);
      border: none;
      color: white;
      padding: 12px 20px;
      border-radius: 12px;
      box-shadow: 0 10px 18px rgba(35, 71, 213, 0.22);
    }
    .result {
      margin-top: 22px;
      background: white;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px 20px;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .result h2 {
      margin: 0 0 10px;
      font-size: 1.3rem;
    }
    .result p {
      margin: 0;
      line-height: 1.7;
      color: var(--ink);
    }
    .citations {
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
      color: var(--muted);
    }
    .citations strong { color: var(--ink); }
    .citation {
      display: block;
      margin: 6px 0;
      line-height: 1.5;
    }
    .error {
      background: var(--error-bg);
      border-color: var(--error-border);
      color: #7f1d1d;
    }
    @media (max-width: 600px) {
      .hero, .card { padding: 18px 16px; }
      .actions { align-items: flex-start; }
      .examples { margin-bottom: 10px; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"hero\">
      <div class=\"eyebrow\">Portfolio demo</div>
      <h1>AI Governance Knowledge Agent</h1>
      <p class=\"subtitle\">Ask grounded questions from the frozen AI governance and security document set. Answers are sourced from the retrieved passages and cite the relevant document pages.</p>
    </div>
    <div class=\"card\">
      <form method=\"POST\" action=\"/\">
        <textarea id=\"question\" name=\"question\" placeholder=\"Example: What are the four core functions of the AI RMF 1.0?\"></textarea>
        <div class=\"actions\">
          <div class=\"examples\">
            <button type=\"button\" class=\"example-btn\" data-question=\"What are the four core functions of the AI RMF 1.0?\">AI RMF</button>
            <button type=\"button\" class=\"example-btn\" data-question=\"Which GPU models do these documents recommend for hosting AI systems securely?\">GPU question</button>
            <button type=\"button\" class=\"example-btn\" data-question=\"What did the UK Playbook say about indirect prompt injection?\">UK Playbook</button>
          </div>
          <button class=\"submit\" type=\"submit\">Ask the agent</button>
        </div>
      </form>
      <div id=\"result\">{RESULT}</div>
    </div>
  </div>
  <script>
    const buttons = document.querySelectorAll('.example-btn');
    const questionField = document.getElementById('question');
    buttons.forEach((button) => {
      button.addEventListener('click', () => {
        questionField.value = button.dataset.question;
        questionField.focus();
      });
    });
  </script>
</body>
</html>
"""


def render_question_result(question: str):
    q = {"id": "demo", "question": question, "type": "demo"}
    selected = retrieve_question_chunks(q, CHUNKS)
    result = generate_answer(q, selected)
    answer_html = html.escape(result.get("answer", ""))
    citation_html = "".join(f"<span class=\"citation\">- {html.escape(c)}</span>" for c in result.get("citations", []))
    provider = html.escape(result.get("provider", "unknown"))
    return (
        "<div class=\"result\">"
        f"<div class=\"meta\">Provider: {provider}</div>"
        f"<h2>Answer</h2><p>{answer_html}</p>"
        f"<div class=\"citations\"><strong>Citations:</strong>{citation_html}</div>"
        "</div>"
    )


class DemoApp:
    def __call__(self, environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET").upper()
        request_path = urlparse(environ.get("PATH_INFO", "/")).path
        if method in {"GET", "HEAD"} and request_path == "/og-image.png":
            if not OG_IMAGE_PATH.is_file():
                start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
                return [b"Preview image not found."]
            image = OG_IMAGE_PATH.read_bytes()
            start_response("200 OK", [
                ("Content-Type", "image/png"),
                ("Content-Length", str(len(image))),
                ("Cache-Control", "public, max-age=86400"),
            ])
            return [b"" if method == "HEAD" else image]

        if method == "POST":
            length = int(environ.get("CONTENT_LENGTH", "0") or 0)
            raw = environ["wsgi.input"].read(length)
            form = parse_qs(raw.decode("utf-8"), keep_blank_values=True)
            question = (form.get("question", [""])[0] or "").strip()
        else:
            question = ""

        if not question:
            body = "<div class=\"result error\">Please enter a question.</div>"
        else:
            try:
                body = render_question_result(question)
            except Exception as exc:
                body = f"<div class=\"result error\">Error: {html.escape(str(exc))}</div>"

        rendered = HTML_PAGE.replace("{RESULT}", body)
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [rendered.encode("utf-8")]


app = DemoApp()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path == "/og-image.png":
            if not OG_IMAGE_PATH.is_file():
                self.send_error(404, "Preview image not found")
                return
            image = OG_IMAGE_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(image)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(image)
            return
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.replace("{RESULT}", "").encode("utf-8"))

    def do_HEAD(self):
        if urlparse(self.path).path == "/og-image.png":
            if not OG_IMAGE_PATH.is_file():
                self.send_error(404, "Preview image not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(OG_IMAGE_PATH.stat().st_size))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        form = parse_qs(raw.decode("utf-8"), keep_blank_values=True)
        question = (form.get("question", [""])[0] or "").strip()

        if not question:
            body = "<div class=\"result error\">Please enter a question.</div>"
        else:
            try:
                body = render_question_result(question)
            except Exception as exc:
                body = f"<div class=\"result error\">Error: {html.escape(str(exc))}</div>"

        rendered = HTML_PAGE.replace("{RESULT}", body)
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(rendered.encode("utf-8"))

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Serving AI Governance RAG demo at http://localhost:{port}/")
    server.serve_forever()
