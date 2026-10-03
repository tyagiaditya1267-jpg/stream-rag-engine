import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from intent_controller import StreamIntentController
from state_differ import SessionStateDiffer
from search_engine import async_hybrid_search, deduplicate_results, ensure_collection_populated
from ingest import ingest_documents

app = FastAPI(title="Streaming Live RAG Engine", version="2.0.0")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Enable CORS for cross-origin web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = StreamIntentController()
state_differ = SessionStateDiffer()

# Models for REST API
class StreamPayload(BaseModel):
    transcript: str

class IngestDoc(BaseModel):
    doc_id: str
    section: Optional[str] = "§1.0"
    text: str

class IngestRequest(BaseModel):
    documents: List[IngestDoc]

@app.on_event("startup")
async def startup_event():
    # Ensure vector store collection exists and is populated
    ensure_collection_populated()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return HTMLResponse("<h2>Streaming Live RAG Engine API</h2><p>Static UI file not found.</p>")

@app.get("/favicon.ico")
async def get_favicon():
    favicon_file = os.path.join(static_dir, "favicon.svg")
    if os.path.exists(favicon_file):
        return FileResponse(favicon_file)
    return HTMLResponse("", status_code=404)

@app.get("/api/health")
async def health_check():
    return {
        "status": "ONLINE",
        "engine": "Streaming Live RAG Engine 2.0",
        "state_version": f"v{state_differ.version}",
        "checksum": state_differ.compute_graph_checksum()
    }

@app.get("/api/state")
async def get_state():
    return {
        "version": f"v{state_differ.version}",
        "active_entities": sorted(list(state_differ.active_entities)),
        "graph_checksum": state_differ.compute_graph_checksum(),
        "history": state_differ.history
    }

@app.post("/api/reset")
async def reset_state():
    state_differ.reset()
    return {
        "status": "RESET_SUCCESS",
        "version": f"v{state_differ.version}",
        "reason": "Session state differ reset back to v1."
    }

@app.post("/api/ingest")
async def ingest_custom_docs(payload: IngestRequest):
    docs = [d.dict() for d in payload.documents]
    success = ingest_documents(docs)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to ingest documents.")
    return {"status": "SUCCESS", "message": f"Successfully ingested {len(docs)} documents."}

@app.post("/api/stream")
async def process_stream_rest(payload: StreamPayload):
    return await handle_turn_analysis(payload.transcript)

async def handle_turn_analysis(transcript: str) -> Dict[str, Any]:
    # 1. Evaluate streaming turn through 5-gate pipeline (G1, G4, G3)
    decision = controller.analyze_stream(transcript)
    action = decision.get("action")

    if action == "WAIT":
        current_v = f"v{max(1, state_differ.version - 1)}" if state_differ.history else f"v{state_differ.version}"
        return {
            "action": "WAIT",
            "status": "WAITING",
            "gate": decision.get("gate", "G1_WAIT"),
            "version": current_v,
            "reason": decision["reason"],
            "transcript": transcript,
            "intents": [],
            "delta_intents": [],
            "cached_intents": [],
            "retrieved_docs": [],
            "graph_checksum": state_differ.compute_graph_checksum(),
            "active_entities": sorted(list(state_differ.active_entities))
        }

    elif action == "SUPPRESS":
        current_v = f"v{max(1, state_differ.version - 1)}" if state_differ.history else f"v{state_differ.version}"
        return {
            "action": "SUPPRESS",
            "status": "SUPPRESSED",
            "gate": decision.get("gate", "G4_SUPPRESS"),
            "version": current_v,
            "reason": decision["reason"],
            "transcript": transcript,
            "intents": [],
            "delta_intents": [],
            "cached_intents": [],
            "retrieved_docs": [],
            "graph_checksum": state_differ.compute_graph_checksum(),
            "active_entities": sorted(list(state_differ.active_entities))
        }

    elif action == "RETRIEVE":
        raw_intents = decision.get("intents", [transcript])
        
        # 2. Dynamic State-Diffing Mechanics (v1 -> v2)
        diff_result = state_differ.process_intents(raw_intents)
        delta_intents = diff_result["delta_intents"]
        cached_intents = diff_result["cached_intents"]
        current_version = diff_result["version"]
        
        deduped_docs = []
        is_cached_turn = False

        if delta_intents:
            # Concurrent Asynchronous Retrieval across atomic sub-queries
            retrieval_tasks = [async_hybrid_search(intent, top_k=2) for intent in delta_intents]
            task_results = await asyncio.gather(*retrieval_tasks, return_exceptions=False)
            
            raw_docs = []
            for idx, res in enumerate(task_results):
                raw_docs.extend(res)
                # Cache doc results in state differ
                state_differ.cache_docs_for_intent(delta_intents[idx], res)

            deduped_docs = deduplicate_results(raw_docs)
        elif cached_intents:
            # Use cached docs from prior turn context
            is_cached_turn = True
            cached_docs = state_differ.get_cached_docs_for_intents(cached_intents)
            deduped_docs = deduplicate_results(cached_docs)

        # 4. Gate G5: Fallback Gate check
        fallback_eval = controller.evaluate_fallback(deduped_docs, is_cached_turn=is_cached_turn)
        
        if fallback_eval["action"] == "FALLBACK":
            final_action = "FALLBACK"
            final_status = "FALLBACK"
            final_gate = fallback_eval["gate"]
            final_reason = fallback_eval["reason"]
        else:
            final_action = "RETRIEVE"
            final_status = "RETRIEVED"
            final_gate = "G3_COMMIT"
            final_reason = fallback_eval["reason"] if is_cached_turn else decision["reason"]

        return {
            "action": final_action,
            "status": final_status,
            "gate": final_gate,
            "version": current_version,
            "reason": final_reason,
            "intents": raw_intents,
            "delta_intents": delta_intents,
            "cached_intents": cached_intents,
            "added_entities": diff_result.get("added_entities", []),
            "active_entities": diff_result.get("active_entities", []),
            "transcript": transcript,
            "retrieved_docs": deduped_docs,
            "graph_checksum": diff_result.get("graph_checksum", "")
        }

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] Client connected to live stream.")
    
    try:
        while True:
            data = await websocket.receive_text()
            if not data:
                continue

            try:
                payload = json.loads(data)
                if isinstance(payload, dict):
                    if payload.get("command") == "reset" or payload.get("action") == "RESET":
                        state_differ.reset()
                        await websocket.send_json({
                            "action": "RESET",
                            "status": "RESET_SUCCESS",
                            "version": f"v{state_differ.version}",
                            "reason": "Session state differ reset back to v1.",
                            "transcript": "",
                            "retrieved_docs": [],
                            "graph_checksum": state_differ.compute_graph_checksum(),
                            "active_entities": []
                        })
                        continue
                    transcript = payload.get("transcript", "")
                else:
                    transcript = str(payload)
            except (json.JSONDecodeError, TypeError):
                transcript = data.strip()

            result = await handle_turn_analysis(transcript)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)