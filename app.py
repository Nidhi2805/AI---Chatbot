"""
The product — a login-gated, document-grounded chat with a live security
firewall, answer citations, custom knowledge-base upload, and an admin
dashboard proving what the firewall blocked.

Routes:
  /login /register /logout      — accounts (auth.py)
  /                             — chat (grounded answers + citation chips)
  /kb                           — see what the assistant knows; upload your own doc
  /admin                        — firewall dashboard (audit_log summary)
  /api/chat                     — JSON chat endpoint
"""

import os
from dotenv import load_dotenv
from flask import (Flask, request, session, redirect, url_for,
                   render_template_string, jsonify)

from src.auth import register_user, verify_login
from src.document_extractor import extract_text, ExtractionError
from src.knowledge_base import build_custom_kb, default_kb_description
from src.pipeline import AIFirewallPipeline
from src import audit_log

load_dotenv()  

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET", "dev-secret-change-me")

# Per-user pipeline + KB state (in-memory; fine for a prototype/demo).
_pipelines = {}
_kb_desc = {}
_pdf_bytes = {}


def _get_pipeline(user):
    if user not in _pipelines:
        _pipelines[user] = AIFirewallPipeline(user=user)
        _kb_desc[user] = default_kb_description()
    return _pipelines[user]


def _require_login():
    return session.get("user")


BASE_CSS = """
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:0 auto;padding:24px;background:#0f1720;color:#e6edf3}
 a{color:#58a6ff} .card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin:12px 0}
 input,textarea,button{font:inherit;padding:10px;border-radius:8px;border:1px solid #30363d;background:#0d1117;color:#e6edf3}
 button{background:#238636;border:none;cursor:pointer;font-weight:600} button:hover{background:#2ea043}
 .bubble{padding:12px 14px;border-radius:12px;margin:8px 0;max-width:80%}
 .user{background:#1f6feb;margin-left:auto;text-align:right} .bot{background:#21262d}
 .cite{display:inline-block;background:#30363d;border-radius:6px;padding:1px 7px;margin:2px;font-size:12px}
 .nav{display:flex;gap:16px;margin-bottom:16px} .metric{display:inline-block;min-width:120px}
 .metric b{font-size:26px;display:block;color:#58a6ff} .row{border-bottom:1px solid #21262d;padding:6px 0;font-size:14px}
 .pill{padding:2px 8px;border-radius:10px;font-size:12px}
 .blocked{background:#8b1a1a} .ok{background:#1a5a2a} .ood{background:#6a4a1a}
</style>"""

NAV = """<div class="nav"><a href="/">Chat</a><a href="/kb">Knowledge base</a>
<a href="/admin">Admin</a><a href="/logout">Logout ({{user}})</a></div>"""


@app.route("/register", methods=["GET", "POST"])
def register():
    msg = ""
    if request.method == "POST":
        ok, msg = register_user(request.form.get("username", ""), request.form.get("password", ""))
        if ok:
            return redirect(url_for("login"))
    return render_template_string(BASE_CSS + """
    <h2>Create account</h2><div class="card">
    <form method="post"><input name="username" placeholder="username"><br><br>
    <input name="password" type="password" placeholder="password (6+ chars)"><br><br>
    <button>Register</button></form><p style="color:#f85149">{{msg}}</p>
    <p>Have an account? <a href="/login">Log in</a></p></div>""", msg=msg)


@app.route("/login", methods=["GET", "POST"])
def login():
    msg = ""
    if request.method == "POST":
        u, p = request.form.get("username", ""), request.form.get("password", "")
        if verify_login(u, p):
            session["user"] = u.strip()
            session.setdefault("history_" + u.strip(), [])
            return redirect(url_for("chat"))
        msg = "Invalid username or password."
    return render_template_string(BASE_CSS + """
    <h2>AI Firewall — Log in</h2><div class="card">
    <form method="post"><input name="username" placeholder="username"><br><br>
    <input name="password" type="password" placeholder="password"><br><br>
    <button>Log in</button></form><p style="color:#f85149">{{msg}}</p>
    <p>New here? <a href="/register">Create an account</a></p></div>""", msg=msg)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/")
def chat():
    user = _require_login()
    if not user:
        return redirect(url_for("login"))
    _get_pipeline(user)
    history = session.get("history_" + user, [])
    return render_template_string(BASE_CSS + NAV + """
    <h2>Ask about your document</h2>
    <div class="card"><small>{{desc}}</small></div>
    <div id="log">
    {% for turn in history %}
      <div class="bubble user">{{turn.q}}</div>
      <div class="bubble bot">{{turn.a}}
        {% if turn.cites %}<br>{% for c in turn.cites %}<span class="cite" title="{{c.text}}">source [{{c.marker}}]</span>{% endfor %}{% endif %}
      </div>
    {% endfor %}
    </div>
    <div class="card"><form method="post" action="/ask">
      <input name="q" style="width:78%" placeholder="e.g. Which Act was the company incorporated under?" autofocus>
      <button>Send</button></form></div>""",
    user=user, history=history, desc=_kb_desc.get(user, default_kb_description()))


@app.route("/ask", methods=["POST"])
def ask():
    user = _require_login()
    if not user:
        return redirect(url_for("login"))
    q = request.form.get("q", "").strip()
    if q:
        result = _get_pipeline(user).process_query(q)
        hist = session.get("history_" + user, [])
        hist.append({"q": q, "a": result["response"], "cites": result.get("citations", [])})
        session["history_" + user] = hist[-20:]
    return redirect(url_for("chat"))


@app.route("/api/chat", methods=["POST"])
def api_chat():
    user = _require_login()
    if not user:
        return jsonify({"error": "not authenticated"}), 401
    q = (request.json or {}).get("query", "").strip()
    result = _get_pipeline(user).process_query(q)
    return jsonify({"answer": result["response"], "status": result["status"],
                    "citations": result.get("citations", [])})


@app.route("/kb", methods=["GET", "POST"])
def kb():
    user = _require_login()
    if not user:
        return redirect(url_for("login"))
    msg = ""
    if request.method == "POST":
        f = request.files.get("file")
        pasted = request.form.get("text", "").strip()
        try:
            if f and f.filename:
                data = f.read()
                text = extract_text(f.filename, data)
                _pdf_bytes[user] = data if f.filename.lower().endswith(".pdf") else None
                src_name = f.filename
            elif pasted:
                text = pasted
                _pdf_bytes[user] = None
                src_name = "pasted text"
            else:
                raise ExtractionError("Provide a file or paste some text.")
            store, gate, desc = build_custom_kb(text, source_name=src_name)
            _pipelines[user] = AIFirewallPipeline(store=store, topic_gate=gate,
                                                  pdf_bytes=_pdf_bytes[user], user=user)
            _kb_desc[user] = desc
            session["history_" + user] = []
            msg = "Knowledge base replaced. Chat has been reset to the new document."
        except ExtractionError as e:
            msg = str(e)
    return render_template_string(BASE_CSS + NAV + """
    <h2>Knowledge base</h2>
    <div class="card"><b>Currently loaded:</b><br><small>{{desc}}</small></div>
    <div class="card"><b>Replace it with your own document</b>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".pdf,.docx,.txt,.md"><br><br>
      <textarea name="text" style="width:96%" rows="4" placeholder="...or paste text here"></textarea><br><br>
      <button>Rebuild knowledge base</button></form>
    <p style="color:#3fb950">{{msg}}</p></div>""",
    user=user, desc=_kb_desc.get(user, default_kb_description()), msg=msg)


@app.route("/admin")
def admin():
    user = _require_login()
    if not user:
        return redirect(url_for("login"))
    s = audit_log.summary()
    return render_template_string(BASE_CSS + NAV + """
    <h2>Firewall dashboard</h2>
    <div class="card">
      <span class="metric"><b>{{s.total_queries}}</b>queries</span>
      <span class="metric"><b>{{s.answered}}</b>answered</span>
      <span class="metric"><b>{{s.threats_blocked}}</b>threats blocked</span>
      <span class="metric"><b>{{s.out_of_domain}}</b>out-of-domain</span>
      <span class="metric"><b>{{s.avg_latency_ms}}</b>avg ms</span>
    </div>
    <div class="card"><b>Recent activity</b>
    {% for e in s.recent %}
      <div class="row">
        {% if e.status=='blocked_at_input' or e.status=='blocked_at_output' %}<span class="pill blocked">BLOCKED</span>
        {% elif e.status=='short_circuited_out_of_domain' %}<span class="pill ood">OUT-OF-DOMAIN</span>
        {% else %}<span class="pill ok">{{e.status}}</span>{% endif %}
        &nbsp;{{e.query}} &nbsp;<small>({{e.latency_ms}}ms{% if e.input_label %}, {{e.input_label}}{% endif %})</small>
      </div>
    {% endfor %}
    {% if not s.recent %}<small>No activity yet — ask something in the chat.</small>{% endif %}
    </div>""", user=user, s=s)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
