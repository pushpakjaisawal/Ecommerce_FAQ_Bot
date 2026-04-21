# 🛒 E-Commerce FAQ Bot — Agentic AI Customer Support

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.4%2B-red?style=for-the-badge&logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-purple?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An Agentic AI-powered customer support bot that handles 500+ daily e-commerce queries using LangGraph, ChromaDB, and Groq LLaMA 3.**

[🚀 Live Demo](#live-demo) · [📖 Documentation](#documentation) · [🛠️ Installation](#installation) · [🤝 Contributing](#contributing)

</div>

---
Demo link: https://ecommercefaqbotsupport.streamlit.app/
## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Bot](#running-the-bot)
- [Deploying to Streamlit Cloud](#deploying-to-streamlit-cloud)
- [Agent Pipeline](#agent-pipeline)
- [Knowledge Base Topics](#knowledge-base-topics)
- [Testing](#testing)
- [Evaluation RAGAS](#evaluation-ragas)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 Overview

The **E-Commerce FAQ Bot** is a production-ready **Agentic AI** customer support assistant built with **LangGraph** stateful graphs. It intelligently handles the most common post-purchase queries that e-commerce customer support teams face daily — including payment failures, wrong items, refund tracking, delayed deliveries, delivery complaints, damaged products, and counterfeit product reports.

The bot is **grounded** in a structured knowledge base (ChromaDB + SentenceTransformers) and uses **faithfulness evaluation** to ensure every answer is factually accurate — never hallucinated.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔀 **Intelligent Routing** | LLM-based router classifies queries into `retrieve`, `skip`, or `tool` paths |
| 📚 **RAG Pipeline** | Semantic search over 12 KB documents using ChromaDB + all-MiniLM-L6-v2 |
| 🔧 **Tool Use** | Built-in datetime tool and return window calculator |
| 🧠 **Multi-turn Memory** | MemorySaver checkpointer maintains conversation context across turns |
| ⚖️ **Faithfulness Evaluation** | LLM-based scoring — auto-retries answers scoring below 0.6 |
| 💬 **Streamlit Chat UI** | Dark-themed professional chat interface with metadata badges |
| 🛡️ **Red-team Tested** | Guards against prompt injection and false policy claims |
| 📊 **RAGAS Evaluation** | Faithfulness · Answer Relevancy · Context Precision scoring |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────┐
│ memory_node │  — Sliding window (last 6 msgs), name/order extraction
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ router_node │  — LLM classifies: 'retrieve' | 'skip' | 'tool'
└──────┬──────┘
       │
   ┌───┴──────────────────────┐
   │              │           │
   ▼              ▼           ▼
┌──────────┐  ┌────────┐  ┌────────┐
│ retrieve │  │  skip  │  │  tool  │
│ ChromaDB │  │chitchat│  │datetime│
│  top-3   │  │        │  │calcul. │
└────┬─────┘  └───┬────┘  └───┬────┘
     └────────────┴───────────┘
                  │
                  ▼
         ┌───────────────┐
         │  answer_node  │  — LLM grounded strictly in context
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │   eval_node   │  — Faithfulness score 0.0 to 1.0
         └───────┬───────┘
                 │
        score >= 0.6?
         YES ────┴──── NO → retry answer (max 2x)
                 │
                 ▼
         ┌───────────────┐
         │   save_node   │  — Append to conversation history
         └───────┬───────┘
                 │
                END
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Agent Framework** | LangGraph 0.2+ (StateGraph + MemorySaver) |
| **LLM** | Groq LLaMA 3.3 70B Versatile |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) |
| **Vector Database** | ChromaDB (in-memory, cosine similarity) |
| **LLM Wrapper** | LangChain Core + LangChain Groq |
| **Frontend** | Streamlit 1.4+ |
| **Evaluation** | RAGAS / Manual LLM-based scoring |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
ecommerce-faq-bot/
│
├── ecommerce_faq_bot.py        ← Main agent (KB, graph, nodes, tests, RAGAS)
├── capstone_streamlit.py       ← Streamlit chat UI
│
├── requirements.txt            ← Python dependencies
├── .gitignore                  ← Excludes .env, venv, secrets
├── .env                        ← Local API keys (never pushed to GitHub)
│
├── .streamlit/
│   └── secrets.toml            ← Local Streamlit secrets (never pushed)
│
└── README.md                   ← This file
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- A free [Groq API key](https://console.groq.com) starting with `gsk_`
- Git

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-faq-bot.git
cd ecommerce-faq-bot
```

---

### Step 2 — Create a Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Verify Installation

```bash
python -c "
packages = {
    'langgraph'            : 'langgraph',
    'langchain-groq'       : 'langchain_groq',
    'langchain-core'       : 'langchain_core',
    'sentence-transformers': 'sentence_transformers',
    'chromadb'             : 'chromadb',
    'streamlit'            : 'streamlit',
}
for name, imp in packages.items():
    try:
        mod = __import__(imp)
        print(f'  OK  {name}')
    except ImportError:
        print(f'  MISSING  {name}')
"
```

---

## 🔑 Configuration

### Option A — `.env` File (Local Development)

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

### Option B — `.streamlit/secrets.toml` (Local Streamlit)

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_actual_key_here"
```

### Option C — Streamlit Cloud Dashboard

In your Streamlit Cloud app go to Settings → Secrets and paste:

```toml
GROQ_API_KEY = "gsk_your_actual_key_here"
```

> **Never commit `.env` or `secrets.toml` to GitHub. Both are excluded by `.gitignore`.**

---

## 🚀 Running the Bot

### Option A — Run Agent and Tests in Terminal

```bash
python ecommerce_faq_bot.py
```

This runs the full 8-part pipeline:
- Builds 12-document ChromaDB knowledge base
- Verifies semantic retrieval
- Assembles and compiles the LangGraph graph
- Runs 10 test questions (8 domain + 2 red-team)
- Runs RAGAS faithfulness evaluation

---

### Option B — Launch Streamlit Chat UI

```bash
streamlit run capstone_streamlit.py
```

Opens at: `http://localhost:8501`

---

### Option C — Interactive Python Session

```python
from ecommerce_faq_bot import (
    build_knowledge_base, verify_retrieval,
    get_llm, build_graph, ask
)

collection, embedder = build_knowledge_base()
verify_retrieval(collection, embedder)
llm = get_llm()
app = build_graph(collection, embedder, llm)

result = ask(app, "My payment failed but money was deducted")
print(result["answer"])
```

---

## ☁️ Deploying to Streamlit Cloud

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: E-Commerce FAQ Bot"
git remote add origin https://github.com/YOUR_USERNAME/ecommerce-faq-bot.git
git branch -M main
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Fill in:
   - Repository: `YOUR_USERNAME/ecommerce-faq-bot`
   - Branch: `main`
   - Main file: `capstone_streamlit.py`
4. Click **Advanced settings** → **Secrets** → paste your API key
5. Click **Deploy**

### Step 3 — Update After Code Changes

```bash
git add .
git commit -m "describe your change"
git push
# Streamlit Cloud auto-redeploys in about 60 seconds
```

---

## 🔄 Agent Pipeline

The bot follows an 8-part pipeline:

| Part | Component | Description |
|---|---|---|
| **Part 1** | Knowledge Base | 12 documents, ChromaDB, all-MiniLM-L6-v2 embeddings |
| **Part 2** | State Design | CapstoneState TypedDict with 12 fields |
| **Part 3** | Node Functions | 8 nodes: memory, router, retrieve, skip, tool, answer, eval, save |
| **Part 4** | Graph Assembly | LangGraph StateGraph with conditional edges and MemorySaver |
| **Part 5** | Testing | 8 domain tests + 2 red-team adversarial tests |
| **Part 6** | RAGAS Eval | Faithfulness, Answer Relevancy, Context Precision |
| **Part 7** | Streamlit UI | Dark-themed chat UI with metadata badges |
| **Part 8** | Summary | Printed runtime summary with scores and metrics |

---

## 📚 Knowledge Base Topics

| # | Topic |
|---|---|
| 1 | Return Policy (30-day window, process, exceptions) |
| 2 | Shipping Information (Standard / Express / Same-Day) |
| 3 | Payment Transaction Failed (reasons, resolution steps) |
| 4 | Wrong Item Received (reporting, reverse pickup, refund) |
| 5 | Refund Status Check (timelines by payment method) |
| 6 | Delayed Deliveries (causes, options, reship or refund) |
| 7 | Complaint About Delivery Boy (escalation process) |
| 8 | Damaged Product (unboxing video, report, replacement) |
| 9 | Fake or Counterfeit Product (detection, reporting, compensation) |
| 10 | Product Catalogue and Availability (search, notify me, wishlist) |
| 11 | Order Cancellation (pre-ship vs post-ship, partial cancel) |
| 12 | Exchange Policy (size or colour swap, eligibility, process) |

---

## 🧪 Testing

### Domain Tests

| ID | Question |
|---|---|
| T01 | Payment failed, money deducted |
| T02 | Wrong product received |
| T03 | Refund status for credit card |
| T04 | Order delayed by 5 days |
| T05 | Rude delivery executive complaint |
| T06 | Damaged product with unboxing video |
| T07 | Suspected fake branded product |
| T08 | Return policy and window |

### Red-Team Tests

| ID | Attack Type |
|---|---|
| R01 | False policy claim (90-day return window) |
| R02 | Prompt injection (ignore instructions) |

---

## 📊 Evaluation RAGAS

| Metric | Description | Target |
|---|---|---|
| **Faithfulness** | Is the answer grounded in KB context? | 0.90+ |
| **Answer Relevancy** | Does it answer the question asked? | 0.88+ |
| **Context Precision** | Was the right context retrieved? | 0.85+ |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent graph framework
- [Groq](https://groq.com) — Ultra-fast LLM inference
- [ChromaDB](https://www.trychroma.com) — Vector database
- [SentenceTransformers](https://www.sbert.net) — Semantic embeddings
- [Streamlit](https://streamlit.io) — Python web UI framework
- [RAGAS](https://github.com/explodinggradients/ragas) — RAG evaluation framework

---

<div align="center">

**Built with care as an Agentic AI Learning Project**

Star this repo if you found it helpful!

</div>
