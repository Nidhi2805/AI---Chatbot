# AI Firewall — a document-grounded assistant with a built-in security firewall

A sellable, end-to-end product: real accounts, a real LLM (Gemini) generating
**grounded, cited** answers from a document a customer uploads, a security
firewall on both the input and the output, and an admin dashboard that proves
what the firewall blocked.

Runs fully without an API key (degrades to showing the top retrieved passage),
so a live demo never hard-fails — but set `GEMINI_API_KEY` for the real thing.

---

## What's new in this version

Beyond the original pipeline, this build adds the four things that turn a
prototype into something you'd put in front of a customer:

| Feature | Where | Why it matters to a buyer |
|---|---|---|
| **Grounded answers with citations** | `src/llm.py`, `/` chat | Every answer sentence is tagged with the source passage it came from ([1], [2]). A customer can audit *why* the assistant said something — the #1 trust requirement for document AI. |
| **Native PDF answering** | `src/llm.py` (`pdf_bytes`) | Scanned documents (like the sample certificate) have a garbled text layer. Instead of relying on broken OCR text, the actual PDF is sent to the model. This is the robust fix for the real-world scanned-doc case. |
| **Abstention / relevance floor** | `src/llm.py`, `src/pipeline.py` | If nothing relevant was retrieved, the assistant says "I don't have that information" instead of returning a confidently-wrong chunk. A wrong answer costs more than an honest miss. |
| **Audit log + admin dashboard** | `src/audit_log.py`, `/admin` | Every query, decision, latency, and block is logged. The dashboard rolls this into live metrics (threats blocked, out-of-domain rate, avg latency) — the evidence a security team needs. |

Plus the retrieval fixes from the debugging pass: custom KBs now chunk small
(45 words) so a short document is searchable per-question, and the relevance
floor uses cosine (stable) rather than BM25 rerank score (goes negative on
short corpora).

---

## The pipeline

```
query
  -> conversational gate      (greetings/small talk: instant reply, nothing else runs)
  -> input checkpoint         (benign / jailbreak / injection / pii — blocks threats)
  -> topic gate               (in-domain? else abstain, no retrieval cost)
  -> retriever                (top-K clean chunks, cosine)
  -> reranker                 (corpus-wide BM25)
  -> relevance floor          (nothing good enough? abstain)
  -> LLM generation           (grounded + cited, or native-PDF, or fallback)
  -> output checkpoint        (scan generated answer for leaked PII/injection)
  -> audit log                (record decision + latency)
```

Agent tool calls go through a separate **action firewall** (argument
inspection + escalation-then-destructive sequence detection).

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# one-time: generate synthetic data + train the classifiers
python -m src.generate_dataset
python -m src.input_checkpoint
python -m src.topic_gate
python -m src.context_scanner

# (optional but recommended) turn on real generation
cp .env.example .env                 # then paste your Gemini key into .env
export GEMINI_API_KEY="your-key"        # or load .env with python-dotenv

python app.py                        # the product at http://localhost:5000
python run_demo.py                   # batch demo in the terminal
python tests/test_pipeline.py        # 9 regression tests
```

## Using the product

1. Open `http://localhost:5000`, register, log in.
2. **`/kb`** — see what the assistant knows in plain language, then upload your
   own PDF / DOCX / TXT (or paste text) to replace it. The chat resets to the
   new document.
3. **`/`** — ask questions. With a key set, answers are generated and cited;
   hover a `source [n]` chip to see the exact passage.
4. **`/admin`** — the firewall dashboard: total queries, threats blocked,
   out-of-domain rate, average latency, and a live feed of recent decisions.

## Getting a Gemini API key

Create one (free tier) at https://aistudio.google.com/apikey and put it in
`.env` (never hardcode it in source — that file is in `.gitignore`).

## Going live (temporary public URL)

```bash
python app.py
cloudflared tunnel --url http://localhost:5000   # prints a public https URL
```
Keep both terminals open. For a stable URL, use a named Cloudflare tunnel or
real hosting (Render/Railway/Fly.io).

---

## Honest limitations (worth raising with a mentor)

- **Keyword-based threat detection has a structural ceiling.** There's always
  another jailbreak phrasing not yet in the list. Reliability scales with
  adversarial testing / training-set size, not with how finished the code
  looks. A production version would add an LLM-based classifier as a second
  layer behind the cheap classical one.
- **A single-document KB makes the topic gate weak.** With one uploaded doc,
  the gate can't learn a sharp in/out boundary; lexically-overlapping
  out-of-domain queries ("capital of Japan" vs "share capital") are the hard
  case. Grounded generation ("answer only from context") is the backstop.
- **The synthetic dataset proves the logic, not real-world accuracy.**
  Swapping in a real labelled attack set and the actual tenant documents is
  the natural next step.

## File map

```
ai_firewall/
├── app.py                  the product: login, chat (cited), /kb, /admin
├── run_demo.py             batch terminal demo
├── interactive_demo.py     live terminal demo
├── requirements.txt
├── .env.example            copy to .env, add your key
├── src/
│   ├── pipeline.py         orchestrator (+ latency, audit, abstention)
│   ├── llm.py              generation: grounded+cited, native-PDF, fallback
│   ├── audit_log.py        append-only JSONL + dashboard summary   [new]
│   ├── knowledge_base.py   upload → chunk → index → fresh topic gate
│   ├── input_checkpoint.py classical-ML threat classifier
│   ├── topic_gate.py       in-domain / answerability gate
│   ├── retriever.py reranker.py vector_store.py chunker.py
│   ├── context_scanner.py  ingestion-time poison/PII tagging
│   ├── action_firewall.py  agent tool-call firewall
│   ├── conversational_gate.py sentiment.py features.py
│   ├── document_extractor.py auth.py display.py generate_dataset.py
└── tests/test_pipeline.py  9 regression tests
```
