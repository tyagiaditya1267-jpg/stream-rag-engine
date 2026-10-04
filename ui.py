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
with open("theme.css", "r") as f:
    custom_theme_css = f.read()

st.markdown(f"""
<style>
{custom_theme_css}

/* Streamlit application specific overrides using the palette */
.stApp {{
    background: var(--page-bg-gradient);
    background-attachment: fixed;
    color: #2D3748;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

/* Custom HUD Header */
.hud-title {{
    font-family: 'Inter', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    background: linear-gradient(90deg, #4FA8A7, #5D82BA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}}

/* Live status badge using solid Pearl Aqua */
.status-badge {{
    display: inline-flex;
    align-items: center;
    background: rgba(var(--color-pearl-aqua-rgb), 0.2);
    border: 1px solid var(--color-pearl-aqua);
    padding: 6px 14px;
    border-radius: 20px;
    color: #134E4D;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 1rem;
}}

.pulse-dot {{
    width: 8px;
    height: 8px;
    background-color: var(--color-pearl-aqua);
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 8px var(--color-pearl-aqua);
    animation: pulse 1.5s infinite;
}}

@keyframes pulse {{
    0% { opacity: 0.4; transform: scale(0.9); }
    50% { opacity: 1; transform: scale(1.2); }
    100% { opacity: 0.4; transform: scale(0.9); }
}}

/* Frosted glass container panel */
.main-glass-panel {{
    background: var(--container-bg-glass);
    backdrop-filter: blur(var(--container-glass-blur));
    -webkit-backdrop-filter: blur(var(--container-glass-blur));
    border: 1px solid var(--container-border-glass);
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 8px 32px 0 rgba(128, 161, 212, 0.15);
    margin-bottom: 1rem;
}}

/* Telemetry cards using pastel cards from theme */
.telemetry-card {{
    background-color: var(--card-bg-project);
    border: 1px solid var(--card-border-subtle);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}}

.citation-pill {{
    background: var(--color-pearl-aqua);
    color: #0F3E3D;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-family: monospace;
    display: inline-block;
    margin-top: 6px;
}}

/* Input and select styling */
.stTextArea textarea, .stSelectbox > div {{
    background-color: rgba(255, 255, 255, 0.8) !important;
    border-radius: 8px !important;
}}

/* Button styling with interactive hover */
.stButton > button {{
    background-color: var(--color-pearl-aqua) !important;
    color: #0B3332 !important;
    font-weight: 600 !important;
    border: 1px solid transparent !important;
    transition: all 0.2s ease !important;
}}

.stButton > button:hover {{
    background-color: var(--color-wisteria-blue) !important;
    color: #FFFFFF !important;
}}
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
    st.markdown('<div class="citations-section-header">🎙️ Transcript Stream Buffer</div>', unsafe_allow_html=True)
    
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
    
    simulate_btn = st.button("🚀 Stream Token Payload")

with col_right:
    st.markdown('<div class="citations-section-header">🔍 Grounded Citations & RRF Results</div>', unsafe_allow_html=True)
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