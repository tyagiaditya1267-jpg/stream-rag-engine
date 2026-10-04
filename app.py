import time
import json
import asyncio
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from intent_controller import StreamIntentController
from search_engine import concurrent_multi_search, hybrid_search
from state_differ import SessionStateDiffer

app = FastAPI(
    title="Streaming Live RAG Engine",
    description="Real-Time 5-Gate Intent-Controlled Hybrid RAG Pipeline with Dynamic State-Diffing",
    version="2.0.0"
)

controller = StreamIntentController()

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "service": "StreamRAG Engine API",
        "pipeline": "IntentController (5-Gate) -> SessionStateDiffer -> HybridSearch (BM25+Dense RRF)"
    }

@app.post("/api/search")
async def api_search(req: QueryRequest):
    start_time = time.time()
    analysis = controller.analyze_stream(req.query)
    
    if analysis.get("action") in ["COMMIT", "RETRIEVE", "PROVISIONAL RETRIEVE"]:
        search_res = await concurrent_multi_search(analysis.get("intents", []))
        retrieved_docs = search_res["docs"]
        telemetry = {
            "dense_count": search_res["dense_count"],
            "sparse_count": search_res["sparse_count"],
            "rrf_count": search_res["rrf_count"],
            "dedup_count": search_res["dedup_count"]
        }
    else:
        retrieved_docs = []
        telemetry = {
            "dense_count": 0,
            "sparse_count": 0,
            "rrf_count": 0,
            "dedup_count": 0
        }

    latency = round((time.time() - start_time) * 1000, 2)
    
    return {
        "query": req.query,
        "intent_decision": analysis,
        "retrieved_docs": retrieved_docs,
        "telemetry": {
            **telemetry,
            "latency_ms": latency
        }
    }

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    differ = SessionStateDiffer()
    
    try:
        while True:
            data = await websocket.receive_text()
            start_time = time.time()
            
            # 1. Evaluate 5-Gate Intent Controller
            analysis = controller.analyze_stream(data)
            action = analysis.get("action", "WAIT")
            
            # 2. Asynchronous Multi-Intent Hybrid Retrieval (Dense + BM25 + RRF)
            if action in ["COMMIT", "RETRIEVE", "PROVISIONAL RETRIEVE"]:
                search_res = await concurrent_multi_search(analysis.get("intents", []))
                retrieved_docs = search_res["docs"]
                dense_count = search_res["dense_count"]
                sparse_count = search_res["sparse_count"]
                rrf_count = search_res["rrf_count"]
                dedup_count = search_res["dedup_count"]
            else:
                retrieved_docs = []
                dense_count = 0
                sparse_count = 0
                rrf_count = 0
                dedup_count = 0
                
            # 3. Dynamic State-Diffing (v_n -> v_{n+1})
            delta_info = differ.compute_delta(analysis, retrieved_docs)
            latency = round((time.time() - start_time) * 1000, 2)
            
            # 4. Synthesize complete telemetry payload
            payload = {
                "gate": analysis.get("gate", "G1"),
                "action": analysis.get("action", "WAIT"),
                "reason": analysis.get("reason", "N/A"),
                "confidence": analysis.get("confidence", 0.0),
                "intents": analysis.get("intents", []),
                "is_multi_intent": analysis.get("is_multi_intent", False),
                "version": delta_info.get("version", "v1"),
                "is_delta_update": delta_info.get("is_delta_update", False),
                "update_type": delta_info.get("update_type", "NO_CHANGE"),
                "retrieval_mode": delta_info.get("retrieval_mode", "FULL_RETRIEVAL"),
                "state_change": delta_info.get("state_change", "N/A"),
                "previous_intents": delta_info.get("previous_intents", []),
                "active_intents": delta_info.get("active_intents", []),
                "retrieved_docs": delta_info.get("docs", []),
                "telemetry": {
                    "latency_ms": latency,
                    "dense_candidates": dense_count,
                    "sparse_candidates": sparse_count,
                    "rrf_candidates": rrf_count,
                    "deduplicated_chunks": dedup_count,
                    "fusion_algorithm": "Reciprocal Rank Fusion (k=60)"
                }
            }
            
            await websocket.send_json(payload)
            
    except WebSocketDisconnect:
        print("Client disconnected cleanly.")
    except Exception as e:
        print(f"Error in websocket streaming loop: {e}")
        try:
            await websocket.close()
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)