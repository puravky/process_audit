import streamlit as st
import google.generativeai as genai
import json
import os
import pandas as pd
from datetime import datetime
from io import BytesIO

st.set_page_config(
    page_title="DMart Process Audit",
    page_icon="🏪",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Load API key from Streamlit Secrets (never exposed to user) ──────────────
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ── Load checkpoints ─────────────────────────────────────────────────────────
@st.cache_data
def load_checkpoints():
    path = os.path.join(os.path.dirname(__file__), "checkpoints.json")
    with open(path) as f:
        return json.load(f)

checkpoints = load_checkpoints()

@st.cache_data
def build_checklist_text():
    lines = []
    for cp in checkpoints:
        for sp in cp["subpoints"]:
            lines.append(
                f"[{cp['id']}.{sp['sub']}] [{cp['section']}] {cp['title']} | "
                f"{sp['criteria'][:120]} | score={sp['score']}"
            )
    return "\n".join(lines)

CHECKLIST_TEXT = build_checklist_text()

SYSTEM_PROMPT = f"""You are a DMart store process audit assistant helping an auditor identify compliance violations.

The auditor describes what they observed on the store floor in plain English.
Map the observation to the correct checkpoint from the official audit checklist below.

RULES:
1. If ONE checkpoint clearly matches, return it directly as JSON.
2. If MULTIPLE checkpoints could match, ask exactly ONE short clarifying question (under 15 words). Return candidates as JSON.
3. Never guess when ambiguous. Never return more than one final confirmed answer.
4. Always respond ONLY with valid JSON. No extra text outside the JSON.

RESPONSE FORMAT when match is clear:
{{
  "status": "match",
  "point_id": 8,
  "sub": 2,
  "section": "FACILITY",
  "title": "GATE PASS FILE AND REGISTER",
  "criteria": "exact criteria text",
  "max_score": 0.05,
  "explanation": "one line why this matches"
}}

RESPONSE FORMAT when clarification needed:
{{
  "status": "clarify",
  "question": "Is this Core or Non Core section?",
  "candidates": [
    {{"point_id": 45, "sub": 2, "section": "GRN", "title": "SP CHANGE REGISTER : CORE", "criteria": "...", "max_score": 0.06}},
    {{"point_id": 46, "sub": 2, "section": "GRN", "title": "SP CHANGE REGISTER : NON CORE", "criteria": "...", "max_score": 0.06}}
  ]
}}

FULL AUDIT CHECKLIST:
{CHECKLIST_TEXT}
"""

# ── Gemini model ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return genai.GenerativeModel("gemini-1.5-flash", system_instruction=SYSTEM_PROMPT)

model = get_model()

def ask_gemini(messages):
    chat = model.start_chat(history=messages[:-1])
    response = chat.send_message(messages[-1]["parts"])
    raw = response.text.strip().strip("```json").strip("```").strip()
    return raw

# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "auditor_name": "",
        "store_name": "",
        "audit_date": datetime.today().strftime("%Y-%m-%d"),
        "session_started": False,
        "chat_history": [],
        "display_chat": [],
        "logged_findings": [],
        "pending_match": None,
        "total_deducted": 0.0,
        "input_key": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .stTextInput input { font-size: 16px; padding: 10px; }
    .stButton button { font-size: 16px; padding: 10px; border-radius: 8px; }
    .finding-card {
        background: #f0f9f0; border-left: 4px solid #2e7d32;
        padding: 10px 14px; border-radius: 6px; margin: 6px 0; font-size: 14px;
    }
    .chat-user {
        background: #e3f2fd; padding: 10px 14px;
        border-radius: 10px; margin: 6px 0; font-size: 15px;
    }
    .chat-ai {
        background: #f5f5f5; padding: 10px 14px; border-radius: 10px;
        margin: 6px 0; font-size: 15px; border-left: 3px solid #ff6b35;
    }
    .clarify-box {
        background: #fffde7; border-left: 4px solid #f9a825;
        padding: 10px 14px; border-radius: 6px; margin: 6px 0;
    }
    .score-box {
        background: #fff3e0; border: 2px solid #ff6b35;
        padding: 14px; border-radius: 10px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Setup screen ──────────────────────────────────────────────────────────────
if not st.session_state.session_started:
    st.title("🏪 DMart Process Audit")
    st.markdown("---")
    st.subheader("Start New Session")

    st.session_state.auditor_name = st.text_input("Your Name", value=st.session_state.auditor_name)
    st.session_state.store_name   = st.text_input("Store Name / Location", value=st.session_state.store_name)
    st.session_state.audit_date   = st.date_input("Date", value=datetime.today()).strftime("%Y-%m-%d")

    if st.button("▶ Start Audit", type="primary"):
        if not st.session_state.auditor_name:
            st.error("Enter your name.")
        elif not st.session_state.store_name:
            st.error("Enter store name.")
        else:
            st.session_state.session_started = True
            st.rerun()
    st.stop()

# ── Main audit screen ─────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"### 🏪 {st.session_state.store_name}")
    st.caption(f"{st.session_state.auditor_name} · {st.session_state.audit_date}")
with col2:
    remaining = round(30 - st.session_state.total_deducted, 2)
    st.markdown(
        f'<div class="score-box"><b style="font-size:24px">{remaining}</b><br>'
        f'<span style="font-size:12px">/ 30.0 remaining</span></div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# Chat display
for msg in st.session_state.display_chat:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">🧑 {msg["text"]}</div>', unsafe_allow_html=True)
    elif msg["type"] == "clarify":
        st.markdown(f'<div class="clarify-box">🤔 {msg["text"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-ai">✅ {msg["text"]}</div>', unsafe_allow_html=True)

# Input
st.markdown("**Describe what you observed:**")
observation = st.text_input(
    label="obs", label_visibility="collapsed",
    placeholder="e.g. spelling mistake on price board in garments section",
    key=f"obs_{st.session_state.input_key}"
)

c1, c2 = st.columns([5, 1])
with c1:
    send = st.button("🔍 Find Checkpoint", type="primary", use_container_width=True)
with c2:
    if st.button("🔄", use_container_width=True):
        st.session_state.display_chat = []
        st.session_state.chat_history = []
        st.session_state.pending_match = None
        st.session_state.input_key += 1
        st.rerun()

# ── Process input ─────────────────────────────────────────────────────────────
if send and observation.strip():
    user_text = observation.strip()
    st.session_state.chat_history.append({"role": "user", "parts": user_text})
    st.session_state.display_chat.append({"role": "user", "text": user_text, "type": "user"})

    with st.spinner("Finding checkpoint..."):
        try:
            raw = ask_gemini(st.session_state.chat_history)
            result = json.loads(raw)
            st.session_state.chat_history.append({"role": "model", "parts": raw})

            if result["status"] == "match":
                st.session_state.pending_match = result
                display_text = (
                    f"**Point {result['point_id']}.{result['sub']} — {result['title']}** "
                    f"[{result['section']}]\n\n"
                    f"{result['criteria'][:220]}\n\n"
                    f"*{result['explanation']}*\n\n"
                    f"**Max deductible: {result['max_score']} pts**"
                )
                st.session_state.display_chat.append({"role": "ai", "text": display_text, "type": "match"})

            elif result["status"] == "clarify":
                st.session_state.pending_match = None
                cands = "\n".join(
                    f"- {c['point_id']}.{c['sub']} {c['title']} [{c['section']}]"
                    for c in result["candidates"]
                )
                st.session_state.display_chat.append({
                    "role": "ai",
                    "text": f"**{result['question']}**\n\nPossible matches:\n{cands}",
                    "type": "clarify"
                })

        except Exception as e:
            st.session_state.display_chat.append({
                "role": "ai", "text": f"Could not parse response. Try rephrasing. ({e})", "type": "error"
            })

    st.session_state.input_key += 1
    st.rerun()

# ── Log finding ───────────────────────────────────────────────────────────────
if st.session_state.pending_match:
    m = st.session_state.pending_match
    st.markdown("---")
    st.markdown(f"**Log finding — Point {m['point_id']}.{m['sub']}** (max {m['max_score']} pts)")

    lc1, lc2, lc3 = st.columns([2, 3, 1])
    with lc1:
        deduct_val = st.number_input(
            "Deduct (pts)", min_value=0.0, max_value=float(m["max_score"]),
            value=float(m["max_score"]), step=0.01, format="%.2f", key="deduct_input"
        )
    with lc2:
        remark = st.text_input("Remark", placeholder="optional note", key="remark_input")
    with lc3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✔ Log", type="primary"):
            st.session_state.logged_findings.append({
                "point_id": m["point_id"], "sub": m["sub"],
                "section": m["section"], "title": m["title"],
                "criteria": m["criteria"], "max_score": m["max_score"],
                "deducted": round(deduct_val, 2), "remark": remark,
                "time": datetime.now().strftime("%H:%M")
            })
            st.session_state.total_deducted = round(st.session_state.total_deducted + deduct_val, 2)
            st.session_state.pending_match = None
            st.session_state.display_chat.append({
                "role": "ai",
                "text": f"✔ Logged {deduct_val} pts deducted — Point {m['point_id']}.{m['sub']}",
                "type": "match"
            })
            st.rerun()
        if st.button("✖ Skip"):
            st.session_state.pending_match = None
            st.rerun()

# ── Findings list ─────────────────────────────────────────────────────────────
if st.session_state.logged_findings:
    st.markdown("---")
    st.markdown(f"### 📋 Logged Findings ({len(st.session_state.logged_findings)})")
    for f in st.session_state.logged_findings:
        remark_str = f" — *{f['remark']}*" if f["remark"] else ""
        st.markdown(
            f'<div class="finding-card">'
            f'<b>{f["point_id"]}.{f["sub"]} {f["title"]}</b> [{f["section"]}]<br>'
            f'<span style="color:#555">{f["criteria"][:110]}...</span><br>'
            f'⬇ <b style="color:#c62828">{f["deducted"]} pts</b>{remark_str} '
            f'<span style="color:#aaa;font-size:12px">· {f["time"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("📥 End Audit & Download Report", type="primary"):
        rows = [{
            "Point No.": f"{f['point_id']}.{f['sub']}",
            "Section": f["section"], "Title": f["title"],
            "Criteria": f["criteria"], "Max Score": f["max_score"],
            "Deducted": f["deducted"], "Remark": f["remark"], "Time": f["time"]
        } for f in st.session_state.logged_findings]

        rows.append({
            "Point No.": "TOTAL", "Section": "", "Title": "", "Criteria": "",
            "Max Score": "", "Deducted": round(st.session_state.total_deducted, 2),
            "Remark": f"Final Score: {round(30 - st.session_state.total_deducted, 2)} / 30",
            "Time": ""
        })

        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Audit Report")
        buf.seek(0)

        st.download_button(
            label="⬇ Download Excel",
            data=buf,
            file_name=f"audit_{st.session_state.store_name.replace(' ','_')}_{st.session_state.audit_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )