import subprocess
import time
import httpx
import streamlit as st
import json
import asyncio
import websockets

# Automatically start FastAPI backend in background on Streamlit Cloud/Local
@st.cache_resource
def start_fastapi_backend():
    # Check if backend is already listening
    try:
        res = httpx.get("http://127.0.0.1:8000/docs", timeout=1.0)
        if res.status_code == 200:
            return
    except Exception:
        pass

    # Launch FastAPI (app.py) in background thread
    process = subprocess.Popen(
        ["python", "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"]
    )
    time.sleep(2)  # Give uvicorn a moment to initialize
    return process

# Trigger background process initialization
start_fastapi_backend()

# Page config for wide layout and dark theme
st.set_page_config(
    page_title="Streaming Live RAG Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# High-Tech Glassmorphism & Cyberpunk CSS Injection
st.markdown("""
<style>
    /* Dark grid background styling */
    .stApp {
        background-color: #0b0f19;
        background-image: radial-gradient(#1e293b 1px, transparent 1px);
        background-size: 24px 24px;
        color: #f8fafc;
    }
    
    /* Custom HUD Header */
    .hud-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.2rem;
        background: linear-gradient(90deg, #00f0ff, #7000ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    /* Live status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        color: #10b981;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.4; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.2); }
        100% { opacity: 0.4; transform: scale(0.9); }
    }

    /* Terminal-style stream display */
    .terminal-container {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.05);
        font-family: 'Consolas', 'Fira Code', monospace;
        color: #38bdf8;
        min-height: 120px;
        margin-top: 10px;
    }

    /* Telemetry cards */
    .telemetry-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(8px);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .citation-pill {
        background: rgba(112, 0, 255, 0.2);
        border: 1px solid rgba(112, 0, 255, 0.5);
        color: #e0e7ff;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-family: monospace;
        display: inline-block;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# HUD Header
st.markdown('<div class="hud-title">⚡ STREAMING LIVE RAG ENGINE</div>', unsafe_allow_html=True)
st.markdown("""
<div class="status-badge">
    <div class="pulse-dot"></div> ENGINE ONLINE | WEBSOCKET: WS://127.0.0.1:8000/WS/STREAM
</div>
""", unsafe_allow_html=True)

# Main UI split layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("🎙️ Live Speech Transcript Stream")
    
    # Preset triggers for rapid testing
    preset = st.selectbox(
        "Select sample transcript scenario:",
        [
            "What is the venue capacity and what is the cancellation fee?",
            "What is the venue capacity",
            "Please summarize in bullet points",
            "Type custom prompt..."
        ]
    )
    
    default_text = "" if preset == "Type custom prompt..." else preset
    user_input = st.text_area("Live Transcript Buffer:", value=default_text, height=100)
    
    simulate_btn = st.button("🚀 Stream Token Payload")

with col_right:
    st.subheader("🔍 Real-time Telemetry & Citations")
    telemetry_placeholder = st.empty()

# Execute WebSocket round-trip when button is pressed
if simulate_btn and user_input:
    async def stream_to_backend():
        uri = "ws://127.0.0.1:8000/ws/stream"
        start_time = time.time()
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(user_input)
                response = await websocket.recv()
                latency = round((time.time() - start_time) * 1000, 2)
                data = json.loads(response)
                
                with telemetry_placeholder.container():
                    # Metrics row
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Action Gate", data.get("action", "WAIT"))
                    m2.metric("State Version", data.get("version", "v1"))
                    m3.metric("Latency", f"{latency} ms")
                    
                    st.markdown("---")
                    
                    # Reason & Intents
                    st.markdown(f"**Gate Reasoning:** `{data.get('reason', 'N/A')}`")
                    
                    intents = data.get("intents", [])
                    if intents:
                        st.markdown("**Decomposed Intents:**")
                        for idx, intent in enumerate(intents, 1):
                            st.write(f"- Sub-query {idx}: `{intent}`")
                    
                    # Retrieved Citations
                    docs = data.get("retrieved_docs", [])
                    if docs:
                        st.markdown("**Grounded Context Citations:**")
                        for doc in docs:
                            citation = doc.get('citation', '[Doc_Ref]')
                            text = doc.get('text', '')
                            score = doc.get('score', 0.0)
                            st.markdown(f"""
                            <div class="telemetry-card">
                                <span class="citation-pill">{citation}</span> <b>Match Score: {score}</b>
                                <p style="margin-top: 8px; color: #cbd5e1; font-size: 0.9rem;">{text}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No vector search fired for current token payload.")
                        
        except Exception as e:
            telemetry_placeholder.error(f"WebSocket Connection Failed: {e}. Ensure backend is running.")

    asyncio.run(stream_to_backend())