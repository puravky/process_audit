import streamlit as st
from google import genai
from google.genai import types
import json
import os
import pandas as pd
from datetime import datetime
from io import BytesIO

st.set_page_config(
    page_title="CheckMate",
    page_icon="✅",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("Hi Manoj! 👋")
st.subheader("I'm CheckMate🕵🏻‍♂️, your DMart audit assistant.")

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

SYSTEM_PROMPT = f"""You are CheckMate, a DMart store process audit assistant.

The auditor describes observations from the store floor in plain English.
Your job is to map each observation to the correct checkpoint from the audit checklist.

RULES:
1. If ONE checkpoint clearly matches, return it directly as JSON.
2. If MULTIPLE checkpoints could match, ask exactly ONE short clarifying question (under 15 words) and return candidates as JSON.
3. Never guess when ambiguous. Never return more than one final confirmed answer.
4. Always respond ONLY with valid JSON. No extra text outside the JSON.

RESPONSE FORMAT when match is clear:
{{
  "status": "match",
  "point_id": 8,
  "sub": 2,
  "section": "FACILITY",
  "title": "GATE PASS FILE AND REGISTER",
  "criteria": "exact criteria text here",
  "max_score": 0.05,
  "explanation": "one line why this matches"
}}

RESPONSE FORMAT when clarification needed:
{{
  "status": "clarify",
  "question": "Is this a Core or Non Core section board?",
  "candidates": [
    {{"point_id": 45, "sub": 2, "section": "GRN", "title": "SP CHANGE REGISTER : CORE", "criteria": "...", "max_score": 0.06}},
    {{"point_id": 46, "sub": 2, "section": "GRN", "title": "SP CHANGE REGISTER : NON CORE", "criteria": "...", "max_score": 0.06}}
  ]
}}

FULL AUDIT CHECKLIST:
{CHECKLIST_TEXT}
"""

@st.cache_resource
def get_client():
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

client = get_client()

def ask_gemini(messages):
    conversation = []

    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        conversation.append(
            types.Content(
                role=role,
                parts=[types.Part(text=msg["parts"])]
            )
        )

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=conversation,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1
        )
    )

    return response.text.strip()

def init_state():
    defaults = {
        "chat_history": [],
        "messages": [],
        "logged_findings": [],
        "total_deducted": 0.0,
        "pending_match": None,
        "input_key": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

st.markdown("""
<style>
    .main > div { padding-top: 0.5rem; }
    .stButton button { border-radius: 8px; }

    .msg-user {
        display: flex; justify-content: flex-end; margin: 6px 0;
    }
    .msg-user span {
        background: #1a73e8; color: white;
        padding: 10px 14px; border-radius: 18px 18px 4px 18px;
        max-width: 80%; font-size: 15px; line-height: 1.5;
    }
    .msg-bot {
        display: flex; justify-content: flex-start; margin: 6px 0;
    }
    .msg-bot span {
        background: #f1f3f4; color: #1a1a1a;
        padding: 10px 14px; border-radius: 18px 18px 18px 4px;
        max-width: 85%; font-size: 15px; line-height: 1.5;
    }
    .msg-clarify span {
        background: #fffde7; border-left: 3px solid #f9a825;
        border-radius: 18px 18px 18px 4px;
    }
    .msg-match span {
        background: #e8f5e9; border-left: 3px solid #2e7d32;
        border-radius: 18px 18px 18px 4px;
    }
    .score-pill {
        display: inline-block; white-space: nowrap;
        background: #fff3e0; border: 1.5px solid #ff6b35;
        padding: 4px 14px; border-radius: 20px;
        font-size: 14px; font-weight: 600; color: #e64a19;
    }
    .finding-row {
        background: #f9f9f9; border-left: 3px solid #2e7d32;
        padding: 8px 12px; border-radius: 6px;
        margin: 4px 0; font-size: 13px;
    }
    .log-bar {
        background: #f1f3f4; border-radius: 12px;
        padding: 12px 16px; margin: 8px 0;
    }
    div[data-testid="stTextInput"] input {
        font-size: 16px; border-radius: 24px; padding: 10px 18px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([4, 1])
with hc1:
    st.caption("Describe what you see — I'll find the audit checkpoint.")
with hc2:
    st.markdown("<br>", unsafe_allow_html=True)
    remaining = round(30 - st.session_state.total_deducted, 2)
    deducted  = round(st.session_state.total_deducted, 2)
    st.markdown(
        f'<div class="score-pill">⬇ {deducted} &nbsp;|&nbsp; {remaining} / 30</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ── Chat messages ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user"><span>{msg["text"]}</span></div>',
            unsafe_allow_html=True
        )
    elif msg["type"] == "clarify":
        st.markdown(
            f'<div class="msg-bot msg-clarify"><span>🤔 {msg["text"]}</span></div>',
            unsafe_allow_html=True
        )
    elif msg["type"] == "match":
        st.markdown(
            f'<div class="msg-bot msg-match"><span>{msg["text"]}</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="msg-bot"><span>{msg["text"]}</span></div>',
            unsafe_allow_html=True
        )

# ── Log confirmation bar ──────────────────────────────────────────────────────
if st.session_state.pending_match:
    m = st.session_state.pending_match
    st.markdown('<div class="log-bar">', unsafe_allow_html=True)
    st.markdown(f"**Log this?** &nbsp; Point `{m['point_id']}.{m['sub']}` — max **{m['max_score']} pts**")
    lc1, lc2, lc3, lc4 = st.columns([2, 3, 1, 1])
    with lc1:
        deduct_val = st.number_input(
            "Deduct", min_value=0.0, max_value=float(m["max_score"]),
            value=float(m["max_score"]), step=0.01, format="%.2f",
            key="deduct_input", label_visibility="collapsed"
        )
    with lc2:
        remark = st.text_input(
            "Remark", placeholder="optional remark...",
            key="remark_input", label_visibility="collapsed"
        )
    with lc3:
        if st.button("✔ Log", type="primary", use_container_width=True):
            st.session_state.logged_findings.append({
                "point_id": m["point_id"], "sub": m["sub"],
                "section": m["section"],   "title": m["title"],
                "criteria": m["criteria"], "max_score": m["max_score"],
                "deducted": round(deduct_val, 2),
                "remark": remark,
                "time": datetime.now().strftime("%H:%M")
            })
            st.session_state.total_deducted = round(
                st.session_state.total_deducted + deduct_val, 2
            )
            st.session_state.messages.append({
                "role": "bot",
                "text": f"✔ Logged — {deduct_val} pts deducted from Point {m['point_id']}.{m['sub']}",
                "type": "match"
            })
            st.session_state.pending_match = None
            st.rerun()
    with lc4:
        if st.button("✖ Skip", use_container_width=True):
            st.session_state.pending_match = None
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── Input box ─────────────────────────────────────────────────────────────────
st.markdown("---")
ic1, ic2 = st.columns([6, 1])
with ic1:
    observation = st.text_input(
        label="input", label_visibility="collapsed",
        placeholder="e.g. trolley left unattended near billing counter",
        key=f"obs_{st.session_state.input_key}"
    )
with ic2:
    send = st.button("Send", type="primary", use_container_width=True)

# clear button
if st.button("🔄 Clear chat", use_container_width=False):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.pending_match = None
    st.session_state.input_key += 1
    st.rerun()

# ── Process input ─────────────────────────────────────────────────────────────
if send and observation.strip():
    user_text = observation.strip()
    st.session_state.chat_history.append({"role": "user", "parts": user_text})
    st.session_state.messages.append({"role": "user", "text": user_text, "type": "user"})

    with st.spinner("Finding checkpoint..."):
        try:
            raw    = ask_gemini(st.session_state.chat_history)
            result = json.loads(raw)
            st.session_state.chat_history.append({"role": "model", "parts": raw})

            if result["status"] == "match":
                st.session_state.pending_match = result
                text = (
                    f"📍 Point {result['point_id']}.{result['sub']} — {result['title']} "
                    f"[{result['section']}]\n\n"
                    f"{result['criteria'][:250]}\n\n"
                    f"_{result['explanation']}_\n\n"
                    f"Deductible: {result['max_score']} pts"
                )
                st.session_state.messages.append({"role": "bot", "text": text, "type": "match"})

            elif result["status"] == "clarify":
                st.session_state.pending_match = None
                cands = "\n".join(
                    f"• {c['point_id']}.{c['sub']} — {c['title']} [{c['section']}]"
                    for c in result["candidates"]
                )
                text = f"**{result['question']}**\n\nPossible matches:\n{cands}"
                st.session_state.messages.append({"role": "bot", "text": text, "type": "clarify"})

        except Exception as e:
            st.session_state.messages.append({
                "role": "bot",
                "text": f"Couldn't parse that. Try rephrasing. ({e})",
                "type": "error"
            })

    st.session_state.input_key += 1
    st.rerun()

# ── Findings + export ─────────────────────────────────────────────────────────
if st.session_state.logged_findings:
    st.markdown("---")
    with st.expander(f"📋 Findings ({len(st.session_state.logged_findings)})  —  ⬇ {deducted} pts deducted"):
        for f in st.session_state.logged_findings:
            remark_str = f" · *{f['remark']}*" if f["remark"] else ""
            st.markdown(
                f'<div class="finding-row">'
                f'<b>{f["point_id"]}.{f["sub"]} {f["title"]}</b> [{f["section"]}]<br>'
                f'<span style="color:#555">{f["criteria"][:110]}...</span><br>'
                f'<b style="color:#c62828">⬇ {f["deducted"]} pts</b>{remark_str} '
                f'<span style="color:#aaa">· {f["time"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        if st.button("📥 Download Report", type="primary"):
            rows = [{
                "Point No.": f"{f['point_id']}.{f['sub']}",
                "Section":   f["section"],
                "Title":     f["title"],
                "Criteria":  f["criteria"],
                "Max Score": f["max_score"],
                "Deducted":  f["deducted"],
                "Remark":    f["remark"],
                "Time":      f["time"]
            } for f in st.session_state.logged_findings]

            rows.append({
                "Point No.": "TOTAL", "Section": "", "Title": "", "Criteria": "",
                "Max Score": "",
                "Deducted":  round(st.session_state.total_deducted, 2),
                "Remark":    f"Final Score: {round(30 - st.session_state.total_deducted, 2)} / 30",
                "Time": ""
            })

            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Audit Report")
            buf.seek(0)

            today = datetime.today().strftime("%Y-%m-%d")
            st.download_button(
                label="⬇ Download Excel",
                data=buf,
                file_name=f"pa_report_{today}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )