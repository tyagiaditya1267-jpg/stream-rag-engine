# Streaming Live RAG Engine

A real-time Retrieval-Augmented Generation system for grounded Question Answering over live transcript and document data. The engine combines semantic retrieval, keyword matching, state-aware routing, and citation-backed responses to support fast and context-aware knowledge access.

---

## Project Overview

The system answers queries against a document corpus while reducing redundant retrieval across conversational turns. It can detect incomplete input, ignore formatting or conversational filler, split compound requests into sub-queries, and selectively fetch relevant context only when needed.

This makes it suitable for live assistants, support agents, knowledge copilots, and any workflow requiring fast grounding over a curated corpus.

---

## Why this project matters

Modern retrieval systems often over-query, re-fetch identical information, or respond without clear evidence. This project addresses those issues by combining:

- semantic retrieval for meaning-based matching
- sparse retrieval for exact term matching
- session-aware state tracking to reduce unnecessary recomputation
- citation-grounded results for explainability and trust
- streaming interfaces for real-time interactions

---

## Core Features

### 1. Five-gate intent routing
The project includes a deterministic gating system in `intent_controller.py` that evaluates each transcript turn before search is triggered:

- G1: Wait for incomplete sentences
- G2: Provisional retrieval trigger
- G3: Commit to valid retrieval flow
- G4: Suppress formatting or filler requests
- G5: Fallback when results are weak or empty

This avoids unnecessary vector queries and helps keep the system conversationally stable.

### 2. Hybrid retrieval with dense + sparse search
The retrieval engine uses Qdrant with a hybrid configuration:

- dense vector search with `sentence-transformers/all-MiniLM-L6-v2`
- sparse BM25-style keyword matching
- Reciprocal Rank Fusion (RRF) to combine results

The search logic is implemented in `search_engine.py` and supports both synchronous and asynchronous execution.

### 3. State-diffing across conversation turns
The `SessionStateDiffer` in `state_differ.py` tracks active entities and recent intent hashes. If a user asks a repeated or similar question, the system can reuse cached context instead of running fresh retrieval again.

This makes the engine significantly more efficient in streaming multi-turn situations.

### 4. Citation-based grounding
Ingested documents are split into chunks and stored with metadata such as:

- source doc ID
- section label
- citation tag
- chunk text

This allows the engine to return evidence-backed references for each result.

### 5. API + real-time interfaces
The project exposes:

- FastAPI REST endpoints
- WebSocket streaming support
- a static HTML page served from the backend
- a Streamlit dashboard UI

This makes it easy to demonstrate and integrate the engine in either a lightweight web prototype or a more advanced production workflow.

---

## System Architecture

The pipeline is structured as follows:

1. User input arrives through the backend API or WebSocket.
2. `StreamIntentController` decides whether the input is a valid retrieval trigger.
3. `SessionStateDiffer` checks for repeated or cached intents and updates state versioning.
4. `async_hybrid_search` executes dense and sparse retrieval concurrently.
5. Results are deduplicated and ranked.
6. Relevant citations are returned to the UI or API consumer.

High-level flow:

```text
User Transcript
    ↓
Intent Gate (WAIT / SUPPRESS / RETRIEVE)
    ↓
State Diff + Cache Evaluation
    ↓
Hybrid Qdrant Search (Dense + BM25)
    ↓
Deduplicate + Rank + Citation
    ↓
API / WebSocket / Dashboard Response
```

---

## Project Structure

```text
stream-rag-engine-1/
├── app.py                  # FastAPI server and WebSocket orchestration
├── ingest.py               # Document ingestion and chunking pipeline
├── intent_controller.py     # 5-gate retrieval routing logic
├── search_engine.py        # Hybrid retrieval, deduplication, Qdrant access
├── setup_vectorstore.py    # Vector collection initialization
├── state_differ.py         # Session state versioning and delta tracking
├── ui.py                   # Streamlit dashboard for real-time telemetry
├── test_pipeline.py        # Regression tests for routing and state logic
├── docker-compose.yml      # Docker orchestration for Qdrant + app
├── Dockerfile              # Container build for app + UI
├── requirements.txt        # Python dependencies
├── static/
│   └── index.html         # Lightweight static frontend served by FastAPI
├── qdrant_db/             # Local embedded Qdrant DB fallback
├── ARCHITECTURE.md        # Technical architecture notes
├── README.md              # Project documentation
├── LICENSE                # MIT license
```

---

## Tech Stack

- Python 3.10+
- FastAPI
- Streamlit
- Qdrant
- Sentence Transformers
- BM25 / sparse retrieval support
- Asyncio
- Docker

---

## Installation

### Option 1: Local Python environment

```bash
git clone <your-repository-url>
cd stream-rag-engine-1
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Option 2: Docker

```bash
docker-compose up --build
```

This will start:

- Qdrant on port `6333`
- the FastAPI app on port `8000`
- the Streamlit dashboard on port `8501`

---

## Running the app

### Start the API server

```bash
python app.py
```

This starts the FastAPI service at:

- `http://localhost:8000`
- WebSocket: `ws://localhost:8000/ws/stream`

The app automatically ensures the vector collection is created and populated if needed. If Qdrant is not running remotely, it falls back to the local embedded database at `./qdrant_db`.

### Start the dashboard

```bash
streamlit run ui.py
```

Open the Streamlit UI at:

- `http://localhost:8501`

---

## Example usage

The project includes a sample corpus for event and operational knowledge, including:

- venue capacity
- registration deadlines
- cancellation refund rules
- submission requirements
- support channels and policy constraints

Example queries you can test:

```text
What is the venue capacity and what is the cancellation fee?
What is the main event venue capacity in Pune?
When are submissions due?
Please summarize in bullet points
```

The system will either:

- wait for more context,
- suppress non-retrieval instructions,
- retrieve grounded docs,
- or fallback if the results are weak.

---

## API Reference

### Health check

```http
GET /api/health
```

Returns a basic service status payload including state version and graph checksum.

### Reset session state

```http
POST /api/reset
```

Resets the internal session memory and graph state to its default version.

### Stream a transcript

```http
POST /api/stream
{
  "transcript": "What is the venue capacity and what is the cancellation fee?"
}
```

Returns:

- action
- gate policy
- version
- reasons
- decomposed intents
- retrieved citations
- active session entities

### WebSocket streaming

Connect to:

```text
ws://localhost:8000/ws/stream
```

Send JSON such as:

```json
{"transcript": "What is the venue capacity and what is the cancellation fee?"}
```

---

## Testing

The project includes a regression suite for the routing and state logic:

```bash
python -m unittest test_pipeline.py
```

Current verification status:

- 6 tests passed
- all pipeline checks passed successfully

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

This project combines modern retrieval tooling, Qdrant vector search, and a structured conversational pipeline to demonstrate a realistic, production-ready RAG workflow for grounded knowledge access.
