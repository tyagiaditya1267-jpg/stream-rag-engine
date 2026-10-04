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

# Load theme.css styling
with open("theme.css", "r", encoding="utf-8") as f:
    theme_css = f.read()

st.markdown(f"<style>{theme_css}</style>", unsafe_allow_html=True)

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
    st.markdown('<div class="citations-section-header">Transcript Stream Buffer</div>', unsafe_allow_html=True)
    
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
    user_input = st.text_area("Live Transcript Buffer:", value=default_text, height=120, placeholder="Waiting for real-time speech token stream...")
    
    simulate_btn = st.button("Stream Token Payload")

with col_right:
    st.markdown('<div class="citations-section-header">Grounded Citations & RRF Results</div>', unsafe_allow_html=True)
    telemetry_placeholder = st.empty()

# Initial empty state before interaction
with telemetry_placeholder.container():
    st.markdown('<div class="empty-state-text">No active sub-queries evaluated yet.</div>', unsafe_allow_html=True)

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
                    st.markdown(f'<div class="sub-header-contrast">Gate Reasoning: <span style="font-weight: normal; font-family: monospace; color: #1E293B;">{data.get("reason", "N/A")}</span></div>', unsafe_allow_html=True)
                    
                    intents = data.get("intents", [])
                    st.markdown('<div class="sub-header-contrast">Decomposed Sub-Queries:</div>', unsafe_allow_html=True)
                    if intents:
                        for idx, intent in enumerate(intents, 1):
                            st.markdown(f'<div style="color: #334155; font-size: 0.9rem; padding: 2px 0 2px 8px;">• Sub-query {idx}: <code style="color: #0F3E3D; background: rgba(117,201,200,0.2);">{intent}</code></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="empty-state-text">No active sub-queries evaluated yet.</div>', unsafe_allow_html=True)
                    
                    # Retrieved Documents / Citations
                    docs = data.get("retrieved_docs", [])
                    st.markdown('<div class="sub-header-contrast" style="margin-top: 1rem;">Retrieved Documents:</div>', unsafe_allow_html=True)
                    if docs:
                        for doc in docs:
                            citation = doc.get('citation', '[Doc_Ref]')
                            text = doc.get('text', '')
                            score = doc.get('score', 0.0)
                            st.markdown(f"""
                            <div class="telemetry-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                    <span class="citation-pill">{citation}</span>
                                    <span style="font-size: 0.85rem; color: #475569; font-weight: 600;">Match Score: {score}</span>
                                </div>
                                <p style="margin: 0; color: #1E293B; font-size: 0.9rem; line-height: 1.5;">{text}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="empty-state-text">No vector search fired for current token payload.</div>', unsafe_allow_html=True)
                        
        except Exception as e:
            telemetry_placeholder.error(f"WebSocket Connection Failed: {e}. Ensure backend is running.")

    asyncio.run(stream_to_backend())