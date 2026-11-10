<div align="center">

# AskYourDoc

**Privacy-First, Zero-Retention RAG Platform & Embeddable Document Assistant.**  
Upload PDFs, ask questions, and receive grounded answers with exact page citations and live token streaming.

[**🌐 Live Application: https://askyourdoc-s0p9.onrender.com**](https://askyourdoc-s0p9.onrender.com)

<br/>

[![Live Demo](https://img.shields.io/badge/Render-Live%20Demo-46E3B7?style=flat-square&logo=render)](https://askyourdoc-s0p9.onrender.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-blue?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Free%20Tier-orange?style=flat-square)](https://console.groq.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 🌟 Key Highlights

- **🔒 Zero-Retention RAM Privacy**: Uploaded PDFs and extracted chunks are parsed in volatile memory (`io.BytesIO`). **Zero user files are written to server disk.**
- **⏱️ 30-Minute Idle Auto-Purge**: Inactive documents and in-memory embeddings are automatically purged after 30 minutes, or instantly via the **"Clear All"** button.
- **🛡️ Anti-Prompt-Leakage Guard**: Active guardrails block system prompt extraction attacks (e.g. *"give your system prompt"*), with zero citation pollution.
- **⚡ Real-Time Markdown Token Streaming**: Progressive streaming via FastAPI Server-Sent Events (SSE) and Groq LPU inference, rendering rich GitHub-flavored markdown (headings, bold text, lists, code, tables) live in the browser.
- **📚 Grounded Page Citations**: Exact PDF page citations, match relevance percentages, and clean text excerpts. Phantom citations are automatically suppressed when information is not in the document.
- **💬 Multi-Turn Follow-Up Memory**: Conversational context passing resolves follow-up referents (*"what does it mean?"*, *"explain that clause"*).
- **📋 1-Click Answer Copy**: Dedicated copy button on assistant answers copies clean markdown to your clipboard without dragging along citations.
- **🔑 Bring Your Own Key (BYOK)**: Users and third-party apps can provide their own Groq API key via HTTP headers (`X-Groq-API-Key`), request body, or UI settings modal to use their personal quota.
- **🧩 1-Line Embeddable Chat Widget**: Drop a single `<script>` tag into any website (Webflow, WordPress, React, Shopify) to display a floating AI document drawer.
- **📦 Python Package / CLI**: Installable directly via `pip` with an executable CLI entrypoint (`ask-your-doc`).

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/santoshkkashyap25/AskYourDoc.git
cd AskYourDoc

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # On Windows
# source .venv/bin/activate       # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Copy the example environment file and add your free [Groq API Key](https://console.groq.com/keys):

```bash
copy .env.example .env    # On Windows (or 'cp .env.example .env' on Linux/macOS)
```

Edit `.env`:
```env
GROQ_API_KEY=your_free_groq_api_key_here
```

### 3. Run Application

**Option A: Local Python**
```bash
python run.py
# Or via CLI entrypoint:
ask-your-doc
```

**Option B: Docker**
```bash
# Using Docker Compose (Recommended)
docker compose up --build

# Or standard Docker
docker build -t askyourdoc .
docker run -p 8000:8000 --env-file .env askyourdoc
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## ⚙️ Configuration & Privacy Limits

All parameters can be configured in `.env`:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ZERO_RETENTION` | `True` | Ingest and store PDFs purely in RAM without writing to disk |
| `DOC_INACTIVITY_TIMEOUT_SECONDS` | `1800` | Inactivity idle timeout before auto-purging documents (30 min) |
| `MAX_DOCUMENTS` | `5` | Maximum number of simultaneous active documents allowed in memory |
| `MAX_FILE_SIZE_MB` | `10` | Maximum allowed file size per uploaded PDF |
| `MAX_TOTAL_STORAGE_MB` | `10` | Maximum cumulative in-memory vault storage limit |
| `LLM_PROVIDER` | `groq` | Generator backend (`groq`, `grok`, `openai`, `gemini`, `mock`) |
| `GROQ_API_KEY` | `""` | Free API key from [Groq Console](https://console.groq.com/keys) |
| `GROQ_MODEL_NAME` | `openai/gpt-oss-20b` | Model name (`llama-3.3-70b-versatile`, `qwen/qwen3.6-27b`) |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-Transformers embedding model |
| `CHUNK_SIZE` | `500` | Text chunk character length |
| `CHUNK_OVERLAP` | `50` | Character overlap between adjacent chunks |
| `RETRIEVER_TOP_K` | `4` | Number of context chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.25` | Minimum cosine relevance score required for citations |
| `CACHE_TTL_SECONDS` | `86400` | Cache expiration time (24 hours) |

---

## 🔌 How to Embed in Your Website / App

### 1. Drop-in Floating Chat Widget (1 Line of HTML)
Add this script tag right before `</body>` on any website or CMS (Webflow, WordPress, Shopify, React):

```html
<script src="https://askyourdoc-s0p9.onrender.com/static/widget.js" 
        data-server="https://askyourdoc-s0p9.onrender.com"
        data-title="Ask Document"
        data-position="bottom-right">
</script>
```
*Creates a floating branded bubble button that opens a smooth, mobile-responsive document chat drawer.*

### 2. Headless REST & SSE Streaming API (with BYOK)
Any frontend (React, Next.js, Flutter, iOS/Android) or backend service can query AskYourDoc programmatically. Pass your own Groq key via `X-Groq-API-Key` to use your private quota:

```bash
# 1. Upload a PDF
curl -X POST "https://askyourdoc-s0p9.onrender.com/api/documents/upload" \
  -F "file=@sample_agreement.pdf"

# Response: {"doc_id": "981f5078", "filename": "sample_agreement.pdf", "total_pages": 4, "total_chunks": 12}

# 2. Stream Real-Time Answer via SSE
curl -N -X POST "https://askyourdoc-s0p9.onrender.com/api/query/stream" \
  -H "Content-Type: application/json" \
  -H "X-Groq-API-Key: gsk_your_groq_key_here" \
  -d '{
    "question": "What is the termination notice period?",
    "doc_id": "981f5078",
    "history": [
      {"role": "user", "content": "What is this agreement about?"},
      {"role": "assistant", "content": "It is a software consulting service agreement."}
    ]
  }'
```
Interactive Swagger API documentation is available at **[`/docs`](https://askyourdoc-s0p9.onrender.com/docs)**.

### 3. Install as a Python Package
Install AskYourDoc directly into any Python application:

```bash
pip install git+https://github.com/santoshkkashyap25/ask-your-doc.git
```

```python
from src.services.container import container

# Get workflow with custom or default LLM key
workflow = container.get_workflow(custom_api_key="gsk_...")
result = workflow.run("What are the core deliverables?")

print("Answer:", result.answer)
for c in result.citations:
    print(f"- {c.filename} (Page {c.page_number}): {c.excerpt}")
```

---

## 🧪 Testing

```bash
# Run automated test suite (20 unit tests covering domain, security & BYOK)
python -m unittest tests/test_rag.py

# Run live end-to-end integration tests
python tests/test_e2e.py
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
