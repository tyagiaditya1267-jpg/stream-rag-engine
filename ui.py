import subprocess
import time
import httpx
import streamlit as st
import json
import asyncio
import websockets
from typing import Dict, Any, List

# Core modules for direct in-process execution fallback
from intent_controller import StreamIntentController
from state_differ import SessionStateDiffer
from search_engine import concurrent_multi_search

# Automatically start FastAPI backend in background on local or Streamlit Cloud
@st.cache_resource
def start_fastapi_backend():
    try:
        res = httpx.get("http://127.0.0.1:8000/api/health", timeout=0.8)
        if res.status_code == 200:
            return None
    except Exception:
        pass

    try:
        process = subprocess.Popen(
            ["python", "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(1.5)
        return process
    except Exception:
        return None

# Trigger background process initialization
start_fastapi_backend()

# Page configuration
st.set_page_config(
    page_title="Streaming Live RAG Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load theme.css styling
try:
    with open("theme.css", "r", encoding="utf-8") as f:
        theme_css = f.read()
    st.markdown(f"<style>{theme_css}</style>", unsafe_allow_html=True)
except Exception:
    pass

# Initialize session state for multi-turn state-differing
if "state_differ" not in st.session_state:
    st.session_state.state_differ = SessionStateDiffer()

if "intent_controller" not in st.session_state:
    st.session_state.intent_controller = StreamIntentController()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "latest_payload" not in st.session_state:
    st.session_state.latest_payload = None

# HUD Header
st.markdown('<div class="hud-title">STREAMING LIVE RAG ENGINE</div>', unsafe_allow_html=True)
st.markdown("""
<div class="status-badge">
    <div class="pulse-dot"></div> ENGINE ONLINE | 5-GATE INTENT CONTROL | DYNAMIC STATE-DIFFING | HYBRID RRF
</div>
""", unsafe_allow_html=True)

# Pipeline Architecture Ribbon
st.markdown("""
<div style="background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(192, 185, 221, 0.5); border-radius: 12px; padding: 10px 18px; margin-bottom: 1.5rem; font-size: 0.85rem; color: #334155; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px;">
    <span><b>Query</b> &rarr; <b>Gate (G1–G5)</b> &rarr; <b>State Differ (&Delta;v)</b> &rarr; <b>Multi-Intent Async</b> &rarr; <b>Dense + BM25 RRF</b> &rarr; <b>Deduplication</b> &rarr; <b>Evidence & Citations</b></span>
    <span style="font-size: 0.78rem; font-weight: 700; color: #0F4C4B; background: rgba(117,201,200,0.25); padding: 3px 10px; border-radius: 8px;">Reciprocal Rank Fusion (k=60)</span>
</div>
""", unsafe_allow_html=True)

# Sidebar: System Diagnostics, Gate Architecture & Session State Inspector
with st.sidebar:
    st.markdown("### System Diagnostics")
    
    # State reset
    if st.button("Clear Session State", use_container_width=True):
        st.session_state.state_differ.reset_state()
        st.session_state.conversation_history = []
        st.session_state.latest_payload = None
        st.rerun()

    st.markdown("---")
    st.markdown("#### Logic Gate Reference")
    st.markdown("""
    - **G1 [WAIT]**: Turn entropy low / incomplete sentence
    - **G2 [PROVISIONAL]**: Speculative background probe
    - **G3 [COMMIT]**: Confirmed intent / multi-intent execution
    - **G4 [SUPPRESS]**: Meta-directive / formatting filter
    - **G5 [FALLBACK]**: Direct parametric fallback synthesis
    """)
    
    st.markdown("---")
    st.markdown("#### Active Session Context")
    st.markdown(f"**State Version:** `v{st.session_state.state_differ.context_version}`")
    st.markdown(f"**Total Turns Evaluated:** `{st.session_state.state_differ.turn_count}`")
    
    active_intents = st.session_state.state_differ.active_intents
    if active_intents:
        st.markdown("**Active Knowledge Graph Nodes:**")
        for i, intent in enumerate(active_intents, 1):
            st.markdown(f"<small>{i}. <code>{intent}</code></small>", unsafe_allow_html=True)
    else:
        st.markdown("<small style='color: #64748B;'>No active context nodes yet.</small>", unsafe_allow_html=True)

# Main UI split layout: Left (Transcript Buffer & Stream), Right (Grounded Citations & Telemetry)
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="citations-section-header">Transcript Stream Buffer</div>', unsafe_allow_html=True)
    
    # Preset triggers for testing all architectural gates and multi-intent scenarios
    preset_options = [
        "Select sample transcript scenario...",
        "Multi-Intent: What is the venue capacity and what is the cancellation fee?",
        "Single Query: What is the venue capacity?",
        "Delta Follow-Up: What are the cancellation terms and refund policy?",
        "Hackathon Specs: What are the hackathon submission guidelines and team size?",
        "SLA & Latency: What is the streaming latency SLA and RRF parameter?",
        "Gate G4 (Suppress): Please summarize previous points in bullet points",
        "Gate G1 (Wait): What is",
        "Type custom prompt..."
    ]
    
    selected_preset = st.selectbox("Sample Transcripts & Gate Scenarios:", preset_options)
    
    preset_text_map = {
        "Multi-Intent: What is the venue capacity and what is the cancellation fee?": "What is the venue capacity and what is the cancellation fee?",
        "Single Query: What is the venue capacity?": "What is the venue capacity?",
        "Delta Follow-Up: What are the cancellation terms and refund policy?": "What are the cancellation terms and refund policy?",
        "Hackathon Specs: What are the hackathon submission guidelines and team size?": "What are the hackathon submission guidelines and team size?",
        "SLA & Latency: What is the streaming latency SLA and RRF parameter?": "What is the streaming latency SLA and RRF parameter?",
        "Gate G4 (Suppress): Please summarize previous points in bullet points": "Please summarize previous points in bullet points",
        "Gate G1 (Wait): What is": "What is",
    }
    
    default_text = preset_text_map.get(selected_preset, "" if selected_preset == "Type custom prompt..." else "")
    user_input = st.text_area(
        "Live Transcript Buffer:",
        value=default_text,
        height=110,
        placeholder="Type or stream live speech transcript tokens here..."
    )
    
    stream_col1, stream_col2 = st.columns([2, 1])
    with stream_col1:
        simulate_btn = st.button("Stream Token Payload", use_container_width=True)
    with stream_col2:
        clear_input_btn = st.button("Reset Input", use_container_width=True)
        if clear_input_btn:
            user_input = ""

with col_right:
    st.markdown('<div class="citations-section-header">Grounded Citations & RRF Results</div>', unsafe_allow_html=True)
    results_container = st.container()

# Execution Pipeline Logic (WebSocket with seamless in-process Fallback)
async def execute_stream_query(query_text: str) -> Dict[str, Any]:
    start_time = time.time()
    
    # Strategy 1: Attempt WebSocket stream round-trip to FastAPI backend
    try:
        uri = "ws://127.0.0.1:8000/ws/stream"
        async with websockets.connect(uri, open_timeout=0.6, close_timeout=0.6) as ws:
            await ws.send(query_text)
            resp = await ws.recv()
            data = json.loads(resp)
            data["transport"] = "WebSocket (FastAPI Pipeline)"
            return data
    except Exception:
        pass

    # Strategy 2: Direct In-Process Pipeline Fallback (100% reliable on Streamlit Cloud)
    controller = st.session_state.intent_controller
    differ = st.session_state.state_differ
    
    # 1. 5-Gate Intent Controller Analysis
    analysis = controller.analyze_stream(query_text)
    action = analysis.get("action", "WAIT")
    
    # 2. Async Multi-Intent Hybrid Retrieval (Dense + BM25 + RRF)
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
    
    return {
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
        },
        "transport": "In-Process Async Event Loop"
    }

# Handle streaming execution
if simulate_btn and user_input:
    with st.spinner("Processing token payload through 5-gate intent engine..."):
        payload = asyncio.run(execute_stream_query(user_input))
        st.session_state.latest_payload = payload
        st.session_state.conversation_history.append({
            "query": user_input,
            "payload": payload,
            "timestamp": time.strftime("%H:%M:%S")
        })

# Render Results & Diagnostics in Right Column
with results_container:
    payload = st.session_state.latest_payload
    
    if payload is None:
        st.markdown('<div class="empty-state-text">No active sub-queries evaluated yet. Submit or stream a query to inspect live pipeline execution.</div>', unsafe_allow_html=True)
    else:
        gate = payload.get("gate", "G1")
        action = payload.get("action", "WAIT")
        reason = payload.get("reason", "N/A")
        confidence = payload.get("confidence", 0.0)
        version = payload.get("version", "v1")
        retrieval_mode = payload.get("retrieval_mode", "FULL_RETRIEVAL")
        telemetry = payload.get("telemetry", {})
        latency = telemetry.get("latency_ms", 0.0)
        
        # 1. Top Metrics Bar
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Logic Gate", f"{gate} [{action}]")
        m2.metric("State Version", version)
        m3.metric("Latency", f"{latency} ms")
        m4.metric("Deduplicated Chunks", telemetry.get("deduplicated_chunks", 0))

        st.markdown("---")

        # 2. Intent Decision Card
        gate_badge_colors = {
            "G1": ("#F59E0B", "rgba(245, 158, 11, 0.15)"),
            "G2": ("#3B82F6", "rgba(59, 130, 246, 0.15)"),
            "G3": ("#10B981", "rgba(16, 185, 129, 0.15)"),
            "G4": ("#8B5CF6", "rgba(139, 92, 246, 0.15)"),
            "G5": ("#EF4444", "rgba(239, 68, 68, 0.15)")
        }
        color_primary, color_bg = gate_badge_colors.get(gate, ("#75C9C8", "rgba(117, 201, 200, 0.2)"))

        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.85); border: 1px solid rgba(192, 185, 221, 0.6); border-radius: 12px; padding: 14px; margin-bottom: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="background: {color_bg}; color: {color_primary}; font-weight: 700; padding: 3px 10px; border-radius: 6px; font-size: 0.85rem; border: 1px solid {color_primary};">
                    Gate {gate}: {action}
                </span>
                <span style="font-size: 0.8rem; color: #64748B; font-weight: 600;">Confidence: {int(confidence * 100)}%</span>
            </div>
            <div style="color: #334155; font-size: 0.88rem; line-height: 1.4;">
                <b>Routing Rationale:</b> {reason}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3. Multi-Intent Decomposed Queries
        intents = payload.get("intents", [])
        st.markdown('<div class="sub-header-contrast">Decomposed Sub-Queries:</div>', unsafe_allow_html=True)
        if intents:
            for idx, intent in enumerate(intents, 1):
                st.markdown(f'<div style="color: #1E293B; font-size: 0.9rem; padding: 3px 0 3px 8px; border-left: 3px solid #75C9C8; margin-bottom: 4px; background: rgba(255,255,255,0.6); border-radius: 0 6px 6px 0;">Sub-Query {idx}: <code style="color: #0F3E3D; font-weight: 600; background: rgba(117,201,200,0.2);">{intent}</code></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state-text">No active sub-queries evaluated (Gate ' + gate + ').</div>', unsafe_allow_html=True)

        # 4. State Differencing & Reuse Diagnostics
        st.markdown('<div class="sub-header-contrast" style="margin-top: 0.9rem;">State-Diffing & Cache Mechanics:</div>', unsafe_allow_html=True)
        state_change = payload.get("state_change", "N/A")
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.7); border: 1px solid rgba(192, 185, 221, 0.4); border-radius: 10px; padding: 10px 14px; font-size: 0.85rem; color: #334155; margin-bottom: 12px;">
            <div><b>Retrieval Strategy:</b> <span style="color: #0F4C4B; font-weight: 700;">{retrieval_mode}</span></div>
            <div style="margin-top: 4px; color: #64748B;"><b>State Mutation (&Delta;):</b> {state_change}</div>
        </div>
        """, unsafe_allow_html=True)

        # 5. Hybrid Retrieval Telemetry
        st.markdown('<div class="sub-header-contrast">Hybrid Search Telemetry (BM25 + Dense RRF):</div>', unsafe_allow_html=True)
        t_dense = telemetry.get("dense_candidates", 0)
        t_sparse = telemetry.get("sparse_candidates", 0)
        t_rrf = telemetry.get("rrf_candidates", 0)
        t_dedup = telemetry.get("deduplicated_chunks", 0)
        
        st.markdown(f"""
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px;">
            <div style="flex: 1; min-width: 90px; background: rgba(255,255,255,0.85); border: 1px solid rgba(192, 185, 221, 0.5); padding: 8px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">DENSE PROBES</div>
                <div style="font-size: 1.1rem; color: #1E293B; font-weight: 700;">{t_dense}</div>
            </div>
            <div style="flex: 1; min-width: 90px; background: rgba(255,255,255,0.85); border: 1px solid rgba(192, 185, 221, 0.5); padding: 8px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">SPARSE BM25</div>
                <div style="font-size: 1.1rem; color: #1E293B; font-weight: 700;">{t_sparse}</div>
            </div>
            <div style="flex: 1; min-width: 90px; background: rgba(255,255,255,0.85); border: 1px solid rgba(192, 185, 221, 0.5); padding: 8px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">FUSED (RRF)</div>
                <div style="font-size: 1.1rem; color: #1E293B; font-weight: 700;">{t_rrf}</div>
            </div>
            <div style="flex: 1; min-width: 90px; background: rgba(255,255,255,0.85); border: 1px solid rgba(192, 185, 221, 0.5); padding: 8px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">UNIQUE CHUNKS</div>
                <div style="font-size: 1.1rem; color: #1E293B; font-weight: 700;">{t_dedup}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 6. Retrieved Grounded Context & Ranked Citations
        docs = payload.get("retrieved_docs", [])
        st.markdown('<div class="sub-header-contrast" style="margin-top: 0.8rem;">Grounded Evidence & Ranked Citations:</div>', unsafe_allow_html=True)
        
        if docs:
            for rank_idx, doc in enumerate(docs, 1):
                citation = doc.get('citation', '[Doc_Ref]')
                title = doc.get('title', 'Document Section')
                text = doc.get('text', '')
                score = doc.get('score', 0.0)
                rrf_val = doc.get('rrf_score', 0.0)
                sparse_rank = doc.get('sparse_rank', '-')
                dense_rank = doc.get('dense_rank', '-')
                
                with st.expander(f"Rank #{rank_idx} — {citation}: {title} (Score: {score})", expanded=(rank_idx <= 2)):
                    st.markdown(f"""
                    <div style="margin-bottom: 8px;">
                        <span class="citation-pill">{citation}</span>
                        <span style="font-size: 0.82rem; color: #475569; font-weight: 600; margin-left: 8px;">
                            RRF Value: <code>{rrf_val}</code> | Dense Rank: <code>#{dense_rank}</code> | BM25 Rank: <code>#{sparse_rank}</code>
                        </span>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.95); border: 1px solid rgba(192, 185, 221, 0.5); border-radius: 8px; padding: 12px; color: #1E293B; font-size: 0.92rem; line-height: 1.55;">
                        {text}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state-text">No vector retrieval required or fired for current token payload.</div>', unsafe_allow_html=True)

# Turn History Collapsible Section at the Bottom
if len(st.session_state.conversation_history) > 1:
    with st.expander("Session Turn History & State Evolution Log", expanded=False):
        for idx, turn in enumerate(st.session_state.conversation_history, 1):
            t_payload = turn.get("payload", {})
            st.markdown(f"""
            **Turn {idx} [{turn.get('timestamp')}]**: *"{turn.get('query')}"*  
            &rarr; Gate: `{t_payload.get('gate')}` | Action: `{t_payload.get('action')}` | Version: `{t_payload.get('version')}` | Strategy: `{t_payload.get('retrieval_mode')}` | Chunks: `{len(t_payload.get('retrieved_docs', []))}`
            """)
            st.markdown("---")