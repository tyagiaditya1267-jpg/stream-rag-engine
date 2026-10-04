# Streaming Live RAG Engine

A high-performance, real-time streaming Retrieval-Augmented Generation (RAG) engine featuring a **5-Gate Deterministic Intent Controller**, **Dynamic State-Diffing**, **Asynchronous Multi-Intent Decomposition**, and **Hybrid Reciprocal Rank Fusion (BM25 + Dense RRF)**.

---

## 🏛️ System Architecture

```text
               STREAMRAG ENGINE PIPELINE

                     User Query
                         ↓
               Intent Controller (G1–G5)
                         ↓
               State Differ (&Delta;v_n → v_{n+1})
                         ↓
               Multi-Intent Decomposition
                         ↓
              Async Concurrent Search
               ↙                   ↘
       Dense Semantic              Sparse BM25
               ↘                   ↙
           Reciprocal Rank Fusion (RRF, k=60)
                         ↓
                   Deduplication
                         ↓
            Grounded Evidence / Citations
                         ↓
                UI / WebSocket / API
                         ↓
                Telemetry & Metrics
```

---

## 🚀 Key Features

### 1. 5-Gate Logic Controller (`intent_controller.py`)
Deterministic routing before firing vector searches:
* **G1: WAIT Gate**: Detects incomplete transcripts or low token entropy (< 3 words / dangling conjunctions) to prevent premature retrieval.
* **G2: PROVISIONAL RETRIEVE Gate**: Triggers speculative, background probes for evolving queries with moderate confidence.
* **G3: COMMIT Gate**: Confirms stable entity intent and executes multi-query hybrid retrieval.
* **G4: SUPPRESS Gate**: Halts vector retrieval on meta-directives ("bullet points", "summarize", "rephrase") or conversational filler.
* **G5: FALLBACK Gate**: Routes unresolvable domain queries to parametric fallback synthesis.

### 2. Dynamic State-Diffing (`state_differ.py`)
* Tracks active knowledge graph nodes and previous intent vectors.
* Calculates delta ($\Delta$) state mutations across multi-turn sessions ($v1 \rightarrow v2$).
* Executes **DELTA RETRIEVAL** exclusively for newly introduced sub-queries while reusing cached context.

### 3. Asynchronous Multi-Intent Hybrid Search (`search_engine.py`)
* Splits compound queries (`"What is X and what is Y?"`) into atomic sub-queries and executes them concurrently using `asyncio.gather`.
* Combines **Dense Cosine Similarity** + **Sparse BM25** with **Reciprocal Rank Fusion (RRF)**:
  $$RRF\_Score(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{60 + r_m(d)}$$
* Deduplicates overlapping chunks and extracts provenance metadata (`[Doc_ID §Section]`).

---

## 🖥️ Streamlit Interactive Demo (`ui.py`)

The Streamlit demo exposes all backend capabilities:
* **Logic Gate Inspector**: Displays exact gate decisions (`G1`–`G5`), routing actions, confidence percentages, and rationale.
* **State Evolution & Diffing**: Shows active session context, versioning ($v1 \rightarrow v2$), and delta mutations.
* **Multi-Intent Visualization**: Shows decomposed sub-queries with individual routing markers.
* **Hybrid Search Telemetry**: Live metrics for Dense probes, BM25 matches, Fused RRF candidates, unique chunks, and latency (ms).
* **Ranked Grounded Citations**: Collapsible evidence cards with source IDs, section numbers, RRF values, and dense/sparse ranks.
* **Preset Scenario Triggers**: Quick-test buttons for multi-intent, follow-up delta, meta suppression, and latency benchmarking.

---

## 🛠️ Getting Started

### 1. Installation
```bash
git clone https://github.com/tyagiaditya1267-jpg/stream-rag-engine.git
cd stream-rag-engine
pip install -r requirements.txt
```

### 2. Run Streamlit UI
```bash
streamlit run ui.py
```

### 3. Run FastAPI / WebSocket Backend (Optional)
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

---

## 📡 API Endpoints

* **`GET /api/health`**: Health status and pipeline metadata.
* **`POST /api/search`**: REST endpoint returning intent analysis, hybrid search citations, and latency telemetry.
* **`WS /ws/stream`**: Bidirectional real-time WebSocket pipe for streaming token payloads and state differencing.

---

## 🧪 Testing

Run the test suite:
```bash
python test_engine.py
```

Tests cover:
1. Intent Controller 5-Gate accuracy (G1 Wait, G3 Commit, G4 Suppress).
2. Hybrid Dense + BM25 search and RRF score calculation.
3. Asynchronous concurrent multi-intent search.
4. Multi-turn session state differencing (Fresh &rarr; Cache Hit &rarr; Delta Expansion &rarr; Suppression).
