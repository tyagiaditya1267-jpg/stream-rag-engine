import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from intent_controller import StreamIntentController
from search_engine import concurrent_multi_search
from state_differ import SessionStateDiffer

app = FastAPI(title="Streaming Intent & Retrieval Engine")
controller = StreamIntentController()

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Initialize session state differ for this client stream
    differ = SessionStateDiffer()
    
    try:
        while True:
            data = await websocket.receive_text()
            analysis = controller.analyze_stream(data)
            
            if analysis.get("action") == "RETRIEVE":
                retrieved_docs = await concurrent_multi_search(analysis.get("intents", []))
            else:
                retrieved_docs = []
                
            # Compute version delta (v1 -> v2)
            delta_info = differ.compute_delta(analysis, retrieved_docs)
            
            # Enrich analysis dictionary with telemetry & versioning
            analysis["retrieved_docs"] = delta_info["docs"]
            analysis["version"] = delta_info["version"]
            analysis["is_delta_update"] = delta_info["is_delta_update"]
            analysis["update_type"] = delta_info["update_type"]
            
            await websocket.send_json(analysis)
            
    except WebSocketDisconnect:
        print("Client disconnected cleanly.")
    except Exception as e:
        print(f"Error processing websocket stream: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)