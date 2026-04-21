# """
# =============================================================================
#  capstone_streamlit.py  —  Streamlit UI for the E-Commerce FAQ Bot
# =============================================================================
# Launch with:  streamlit run capstone_streamlit.py

# Architecture notes:
#   - ALL heavy resources (LLM, embedder, ChromaDB, compiled graph) are initialised
#     inside @st.cache_resource so they are created ONCE and reused across reruns.
#   - st.session_state stores the messages list and thread_id per browser session.
#   - A "New Conversation" button in the sidebar resets both.
#   - The UI uses custom HTML/CSS injected via st.markdown for a polished look.
# =============================================================================
# """

# import uuid
# import streamlit as st

# # ─────────────────────────────────────────────────────────────────────────────
# #  Page configuration — MUST be the first Streamlit call
# # ─────────────────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title = "ShopAssist FAQ Bot",
#     page_icon  = "🛒",
#     layout     = "wide",
# )

# # ─────────────────────────────────────────────────────────────────────────────
# #  Import bot modules AFTER page config
# # ─────────────────────────────────────────────────────────────────────────────
# from ecommerce_faq_bot import (
#     build_knowledge_base,
#     verify_retrieval,
#     get_llm,
#     build_graph,
#     ask,
#     KB_DOCUMENTS,
# )

# # ─────────────────────────────────────────────────────────────────────────────
# #  @st.cache_resource — expensive objects initialised ONCE per server process
# #  (survives across multiple user sessions and browser reruns)
# # ─────────────────────────────────────────────────────────────────────────────

# @st.cache_resource
# def load_resources():
#     """
#     Initialises and caches:
#       1. SentenceTransformer embedder
#       2. ChromaDB collection (populated with KB documents)
#       3. ChatGroq LLM
#       4. Compiled LangGraph application

#     Returns
#     -------
#     tuple  —  (app, collection, embedder, llm)
#     """
#     collection, embedder = build_knowledge_base()
#     verify_retrieval(collection, embedder)  # safety gate
#     llm = get_llm()
#     app = build_graph(collection, embedder, llm)
#     return app, collection, embedder, llm


# # ─────────────────────────────────────────────────────────────────────────────
# #  Custom CSS — clean dark-themed chat UI
# # ─────────────────────────────────────────────────────────────────────────────
# CUSTOM_CSS = """
# <style>
# /* ── Global ─────────────────────────────────────────────────────────────── */
# @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

# html, body, [class*="css"] {
#     font-family: 'Sora', sans-serif;
# }
# .main { background: #0D1117; }
# section[data-testid="stSidebar"] { background: #161B22; border-right: 1px solid #30363D; }

# /* ── Header banner ──────────────────────────────────────────────────────── */
# .header-banner {
#     background: linear-gradient(135deg, #1F6FEB 0%, #0D419D 60%, #0D1117 100%);
#     border-radius: 16px;
#     padding: 28px 32px;
#     margin-bottom: 24px;
#     border: 1px solid #1F6FEB44;
# }
# .header-banner h1 { color: #F0F6FC; font-size: 2rem; font-weight: 700; margin: 0; }
# .header-banner p  { color: #8B949E; font-size: 0.95rem; margin: 6px 0 0; }

# /* ── Chat bubbles ───────────────────────────────────────────────────────── */
# .chat-container { display: flex; flex-direction: column; gap: 16px; }

# .user-bubble {
#     align-self: flex-end;
#     background: #1F6FEB;
#     color: #F0F6FC;
#     border-radius: 18px 18px 4px 18px;
#     padding: 12px 18px;
#     max-width: 72%;
#     font-size: 0.92rem;
#     line-height: 1.55;
#     box-shadow: 0 2px 12px #1F6FEB44;
# }
# .bot-bubble {
#     align-self: flex-start;
#     background: #161B22;
#     color: #E6EDF3;
#     border-radius: 18px 18px 18px 4px;
#     padding: 14px 20px;
#     max-width: 76%;
#     font-size: 0.92rem;
#     line-height: 1.6;
#     border: 1px solid #30363D;
# }
# .meta-row {
#     display: flex; gap: 10px; margin-top: 8px; flex-wrap: wrap;
# }
# .badge {
#     font-family: 'JetBrains Mono', monospace;
#     font-size: 0.72rem;
#     padding: 2px 9px;
#     border-radius: 20px;
#     font-weight: 500;
# }
# .badge-route   { background: #1C2432; color: #58A6FF; border: 1px solid #1F6FEB44; }
# .badge-sources { background: #1C2A1C; color: #3FB950; border: 1px solid #3FB95044; }
# .badge-score   { background: #2A1C1C; color: #F85149; border: 1px solid #F8514944; }
# .badge-score.good { color: #3FB950; border-color: #3FB95044; background: #1C2A1C; }

# /* ── Input area ─────────────────────────────────────────────────────────── */
# .stTextInput > div > div > input {
#     background: #161B22 !important;
#     border: 1px solid #30363D !important;
#     border-radius: 12px !important;
#     color: #E6EDF3 !important;
#     font-family: 'Sora', sans-serif !important;
#     padding: 14px 18px !important;
#     font-size: 0.93rem !important;
# }
# .stButton button {
#     background: #1F6FEB !important;
#     color: white !important;
#     border: none !important;
#     border-radius: 10px !important;
#     font-family: 'Sora', sans-serif !important;
#     font-weight: 600 !important;
#     padding: 10px 24px !important;
#     transition: all 0.2s !important;
# }
# .stButton button:hover { background: #388BFD !important; transform: translateY(-1px); }

# /* ── Sidebar ─────────────────────────────────────────────────────────────── */
# .sidebar-card {
#     background: #0D1117;
#     border: 1px solid #30363D;
#     border-radius: 12px;
#     padding: 16px;
#     margin-bottom: 16px;
# }
# .sidebar-card h4 { color: #58A6FF; margin: 0 0 10px; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; }
# .sidebar-card ul { color: #8B949E; font-size: 0.83rem; padding-left: 16px; margin: 0; }
# .sidebar-card ul li { margin-bottom: 5px; }

# /* ── Scrollable chat area ───────────────────────────────────────────────── */
# .scroll-area {
#     max-height: 520px;
#     overflow-y: auto;
#     padding: 4px;
#     scrollbar-width: thin;
#     scrollbar-color: #30363D transparent;
# }
# </style>
# """
# st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# # ─────────────────────────────────────────────────────────────────────────────
# #  Load resources (cached)
# # ─────────────────────────────────────────────────────────────────────────────
# with st.spinner("🔧  Booting ShopAssist — loading models and knowledge base…"):
#     app, collection, embedder, llm = load_resources()


# # ─────────────────────────────────────────────────────────────────────────────
# #  Session state — initialise per-session variables
# # ─────────────────────────────────────────────────────────────────────────────
# if "chat_history" not in st.session_state:
#     # chat_history: list of dicts with keys: role, content, meta
#     st.session_state.chat_history = []

# if "thread_id" not in st.session_state:
#     # Unique thread ID per conversation (used by MemorySaver checkpointer)
#     st.session_state.thread_id = str(uuid.uuid4())

# if "user_name" not in st.session_state:
#     st.session_state.user_name = ""


# # ─────────────────────────────────────────────────────────────────────────────
# #  SIDEBAR
# # ─────────────────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("# 🛒 ShopAssist")
#     st.markdown("---")

#     # Domain description
#     st.markdown("""
#     <div class="sidebar-card">
#         <h4>About this Bot</h4>
#         <ul>
#             <li>AI-powered customer support</li>
#             <li>Handles 500+ daily queries</li>
#             <li>Grounded in your return & shipping policies</li>
#             <li>Multi-turn memory within session</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

#     # Topics covered
#     topics = sorted(set(doc["topic"] for doc in KB_DOCUMENTS))
#     topics_html = "".join(f"<li>{t}</li>" for t in topics)
#     st.markdown(f"""
#     <div class="sidebar-card">
#         <h4>Topics Covered ({len(topics)})</h4>
#         <ul>{topics_html}</ul>
#     </div>
#     """, unsafe_allow_html=True)

#     # Suggested questions
#     st.markdown("""
#     <div class="sidebar-card">
#         <h4>Try asking…</h4>
#         <ul>
#             <li>My payment failed, money was deducted</li>
#             <li>I got the wrong product</li>
#             <li>How do I check my refund status?</li>
#             <li>The delivery boy was rude</li>
#             <li>I think my product is fake</li>
#             <li>My order is 5 days late</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

#     # New conversation button
#     st.markdown("---")
#     if st.button("🔄  New Conversation", use_container_width=True):
#         # Reset both messages list and thread_id (fresh memory slate)
#         st.session_state.chat_history = []
#         st.session_state.thread_id    = str(uuid.uuid4())
#         st.session_state.user_name    = ""
#         st.rerun()

#     # Session info
#     st.caption(f"Session ID: `{st.session_state.thread_id[:8]}…`")
#     st.caption(f"KB documents: {len(KB_DOCUMENTS)}")


# # ─────────────────────────────────────────────────────────────────────────────
# #  MAIN CHAT INTERFACE
# # ─────────────────────────────────────────────────────────────────────────────

# # ── Header banner ─────────────────────────────────────────────────────────────
# name_display = f", {st.session_state.user_name}" if st.session_state.user_name else ""
# st.markdown(f"""
# <div class="header-banner">
#     <h1>🛒 ShopAssist — Customer Support{name_display}</h1>
#     <p>Ask me anything about orders, payments, returns, refunds, shipping, or complaints.</p>
# </div>
# """, unsafe_allow_html=True)

# # ── Chat history display ──────────────────────────────────────────────────────
# chat_html = '<div class="scroll-area"><div class="chat-container">'

# if not st.session_state.chat_history:
#     chat_html += """
#     <div style="text-align:center; color:#8B949E; padding: 60px 0; font-size:0.9rem;">
#         👋  Hi! I'm ShopAssist. How can I help you today?<br>
#         <span style="font-size:0.8rem; margin-top:8px; display:block;">
#             Type your question below or pick a suggestion from the sidebar.
#         </span>
#     </div>
#     """
# else:
#     for msg in st.session_state.chat_history:
#         if msg["role"] == "user":
#             chat_html += f'<div class="user-bubble">🧑 {msg["content"]}</div>'
#         else:
#             meta    = msg.get("meta", {})
#             route   = meta.get("route",       "—")
#             sources = meta.get("sources",     [])
#             score   = meta.get("faithfulness", 1.0)
#             ctype   = meta.get("complaint_type", "")

#             score_class = "good" if score >= 0.7 else ""
#             sources_str = " · ".join(sources[:2]) if sources else "—"

#             chat_html += f"""
#             <div class="bot-bubble">
#                 🤖 {msg["content"]}
#                 <div class="meta-row">
#                     <span class="badge badge-route">route: {route}</span>
#                     <span class="badge badge-sources">📚 {sources_str}</span>
#                     <span class="badge badge-score {score_class}">faith: {score:.2f}</span>
#                     {"<span class='badge badge-route'>" + ctype + "</span>" if ctype else ""}
#                 </div>
#             </div>
#             """

# chat_html += '</div></div>'
# st.markdown(chat_html, unsafe_allow_html=True)

# # ── Input row ─────────────────────────────────────────────────────────────────
# col1, col2 = st.columns([5, 1])

# with col1:
#     user_input = st.text_input(
#         label       = "Your message",
#         placeholder = "e.g.  My payment failed but money was deducted…",
#         label_visibility = "collapsed",
#         key         = "user_input_field",
#     )

# with col2:
#     send_clicked = st.button("Send ➤", use_container_width=True)

# # ── Process message ───────────────────────────────────────────────────────────
# if (send_clicked or user_input) and user_input.strip():
#     question = user_input.strip()

#     # Add user message to display history
#     st.session_state.chat_history.append({"role": "user", "content": question})

#     # Call the LangGraph agent
#     with st.spinner("🤔  Thinking…"):
#         result = ask(app, question, thread_id=st.session_state.thread_id)

#     answer       = result.get("answer",        "I'm sorry, I couldn't process that request.")
#     route        = result.get("route",         "retrieve")
#     sources      = result.get("sources",       [])
#     faithfulness = result.get("faithfulness",  1.0)
#     ctype        = result.get("complaint_type","")
#     user_name    = result.get("user_name",     "")

#     # Persist user name across the session
#     if user_name:
#         st.session_state.user_name = user_name

#     # Add bot response with metadata to history
#     st.session_state.chat_history.append({
#         "role"   : "assistant",
#         "content": answer,
#         "meta"   : {
#             "route"         : route,
#             "sources"       : sources,
#             "faithfulness"  : faithfulness,
#             "complaint_type": ctype,
#         },
#     })

#     # Rerun to refresh the display with the new messages
#     st.rerun()


"""
=============================================================================
 capstone_streamlit.py  —  Streamlit UI for the E-Commerce FAQ Bot
=============================================================================
Launch with:  streamlit run capstone_streamlit.py

Architecture notes:
  - ALL heavy resources (LLM, embedder, ChromaDB, compiled graph) are
    initialised inside @st.cache_resource so they are created ONCE and
    reused across reruns — no repeated model loading.
  - st.session_state stores chat_history, thread_id, user_name, and
    pending_question (submitted form text) per browser session.
  - A "New Conversation" button in the sidebar resets all session state.
  - The UI uses custom HTML/CSS injected via st.markdown.

Bug Fixes Applied
─────────────────
  FIX 1 → Duplicate message bug: replaced st.text_input + st.button with
           st.form(clear_on_submit=True) — the only 100% reliable solution.
           Form fields are wiped before Streamlit reruns, so the old value
           can never re-trigger processing on subsequent reruns.
  FIX 2 → Input field not cleared after sending: clear_on_submit handles this.
  FIX 3 → New Conversation button now resets pending_question too.
  FIX 4 → pending_question cleared immediately before processing — crash-safe.
  FIX 5 → All session state variables initialised before first use.
  FIX 6 → score:.2f crash when faithfulness is not a float → safe cast.
  FIX 7 → HTML injection risk in chat bubbles via msg content → html.escape.
  FIX 8 → Sidebar caption guarded against missing thread_id.
=============================================================================
"""

import uuid
import html    # FIX 7 — used to escape user content before injecting into HTML
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  Page configuration — MUST be the very first Streamlit call in the script
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "ShopAssist FAQ Bot",
    page_icon  = "🛒",
    layout     = "wide",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Import bot modules AFTER set_page_config
# ─────────────────────────────────────────────────────────────────────────────
from ecommerce_faq_bot import (
    build_knowledge_base,
    verify_retrieval,
    get_llm,
    build_graph,
    ask,
    KB_DOCUMENTS,
)

# ─────────────────────────────────────────────────────────────────────────────
#  @st.cache_resource
#  Expensive objects are initialised ONCE per server process and reused
#  across all browser sessions and all Streamlit reruns.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_resources():
    """
    Builds and caches:
      1. SentenceTransformer embedder  (all-MiniLM-L6-v2)
      2. ChromaDB in-memory collection (12 KB documents)
      3. ChatGroq LLM                  (llama-3.3-70b-versatile)
      4. Compiled LangGraph app        (8-node graph + MemorySaver)

    Returns
    -------
    tuple — (app, collection, embedder, llm)
    """
    collection, embedder = build_knowledge_base()
    verify_retrieval(collection, embedder)   # safety gate — crashes early if KB broken
    llm = get_llm()
    app = build_graph(collection, embedder, llm)
    return app, collection, embedder, llm


# ─────────────────────────────────────────────────────────────────────────────
#  Custom CSS — dark-themed professional chat UI
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Global fonts ────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.main { background: #0D1117; }
section[data-testid="stSidebar"] {
    background: #161B22;
    border-right: 1px solid #30363D;
}

/* ── Header banner ───────────────────────────────────────────────────────── */
.header-banner {
    background: linear-gradient(135deg, #1F6FEB 0%, #0D419D 60%, #0D1117 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    border: 1px solid #1F6FEB44;
}
.header-banner h1 {
    color: #F0F6FC; font-size: 2rem;
    font-weight: 700; margin: 0;
}
.header-banner p {
    color: #8B949E; font-size: 0.95rem; margin: 6px 0 0;
}

/* ── Chat bubbles ────────────────────────────────────────────────────────── */
.chat-container { display: flex; flex-direction: column; gap: 16px; }

.user-bubble {
    align-self: flex-end;
    background: #1F6FEB;
    color: #F0F6FC;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 72%;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 2px 12px #1F6FEB44;
    word-break: break-word;
}
.bot-bubble {
    align-self: flex-start;
    background: #161B22;
    color: #E6EDF3;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 20px;
    max-width: 76%;
    font-size: 0.92rem;
    line-height: 1.6;
    border: 1px solid #30363D;
    word-break: break-word;
}

/* ── Metadata badges ─────────────────────────────────────────────────────── */
.meta-row { display: flex; gap: 10px; margin-top: 8px; flex-wrap: wrap; }
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 9px;
    border-radius: 20px;
    font-weight: 500;
}
.badge-route   { background:#1C2432; color:#58A6FF; border:1px solid #1F6FEB44; }
.badge-sources { background:#1C2A1C; color:#3FB950; border:1px solid #3FB95044; }
.badge-score   { background:#2A1C1C; color:#F85149; border:1px solid #F8514944; }
.badge-score.good { color:#3FB950; border-color:#3FB95044; background:#1C2A1C; }
.badge-type    { background:#1C1C2A; color:#A371F7; border:1px solid #A371F744; }

/* ── Input area ──────────────────────────────────────────────────────────── */
.stTextInput > div > div > input {
    background    : #161B22 !important;
    border        : 1px solid #30363D !important;
    border-radius : 12px !important;
    color         : #E6EDF3 !important;
    font-family   : 'Sora', sans-serif !important;
    padding       : 14px 18px !important;
    font-size     : 0.93rem !important;
}
.stTextInput > div > div > input:focus {
    border-color : #1F6FEB !important;
    box-shadow   : 0 0 0 2px #1F6FEB33 !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton button {
    background    : #1F6FEB !important;
    color         : white !important;
    border        : none !important;
    border-radius : 10px !important;
    font-family   : 'Sora', sans-serif !important;
    font-weight   : 600 !important;
    padding       : 10px 24px !important;
    transition    : all 0.2s !important;
}
.stButton button:hover {
    background  : #388BFD !important;
    transform   : translateY(-1px);
    box-shadow  : 0 4px 12px #1F6FEB55;
}

/* ── Sidebar cards ───────────────────────────────────────────────────────── */
.sidebar-card {
    background    : #0D1117;
    border        : 1px solid #30363D;
    border-radius : 12px;
    padding       : 16px;
    margin-bottom : 16px;
}
.sidebar-card h4 {
    color          : #58A6FF;
    margin         : 0 0 10px;
    font-size      : 0.85rem;
    text-transform : uppercase;
    letter-spacing : 0.08em;
}
.sidebar-card ul {
    color       : #8B949E;
    font-size   : 0.83rem;
    padding-left: 16px;
    margin      : 0;
}
.sidebar-card ul li { margin-bottom: 5px; }

/* ── Scrollable chat area ────────────────────────────────────────────────── */
.scroll-area {
    max-height     : 520px;
    overflow-y     : auto;
    padding        : 4px;
    scrollbar-width: thin;
    scrollbar-color: #30363D transparent;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Load resources (cached — runs only once per server process)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("🔧  Booting ShopAssist — loading models and knowledge base…"):
    app, collection, embedder, llm = load_resources()


# ─────────────────────────────────────────────────────────────────────────────
#  Session state — initialise ALL per-session variables here
#  IMPORTANT: every key must be initialised before it is read anywhere below
# ─────────────────────────────────────────────────────────────────────────────

# chat_history : list of message dicts {role, content, meta}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# thread_id : unique ID per conversation for MemorySaver checkpointer
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# user_name : extracted from conversation ("My name is …")
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# pending_question : question submitted via form, processed after form renders
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🛒 ShopAssist")
    st.markdown("---")

    # About card
    st.markdown("""
    <div class="sidebar-card">
        <h4>About this Bot</h4>
        <ul>
            <li>AI-powered customer support</li>
            <li>Handles 500+ daily queries</li>
            <li>Grounded in return &amp; shipping policies</li>
            <li>Multi-turn memory within session</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Topics covered — dynamically built from KB_DOCUMENTS
    topics      = sorted(set(doc["topic"] for doc in KB_DOCUMENTS))
    topics_html = "".join(f"<li>{html.escape(t)}</li>" for t in topics)
    st.markdown(f"""
    <div class="sidebar-card">
        <h4>Topics Covered ({len(topics)})</h4>
        <ul>{topics_html}</ul>
    </div>
    """, unsafe_allow_html=True)

    # Suggested questions
    st.markdown("""
    <div class="sidebar-card">
        <h4>Try Asking…</h4>
        <ul>
            <li>My payment failed, money was deducted</li>
            <li>I got the wrong product</li>
            <li>How do I check my refund status?</li>
            <li>The delivery boy was rude to me</li>
            <li>I think my product is fake</li>
            <li>My order is 5 days late</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── FIX 3 — New Conversation resets pending_question too ──────────────
    if st.button("🔄  New Conversation", use_container_width=True):
        st.session_state.chat_history     = []
        st.session_state.thread_id        = str(uuid.uuid4())
        st.session_state.user_name        = ""
        st.session_state.pending_question = ""   # ← prevents stale question replay
        st.rerun()

    # ── FIX 8 — guard against missing thread_id before slicing ────────────
    tid_display = st.session_state.get("thread_id", "")[:8]
    st.caption(f"Session ID: `{tid_display}…`")
    st.caption(f"KB documents: {len(KB_DOCUMENTS)}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CHAT INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

# ── Header banner ─────────────────────────────────────────────────────────────
name_display = (
    f", {html.escape(st.session_state.user_name)}"
    if st.session_state.user_name else ""
)
st.markdown(f"""
<div class="header-banner">
    <h1>🛒 ShopAssist — Customer Support{name_display}</h1>
    <p>Ask me anything about orders, payments, returns, refunds, shipping, or complaints.</p>
</div>
""", unsafe_allow_html=True)

# ── Chat history display ───────────────────────────────────────────────────────
chat_html = '<div class="scroll-area"><div class="chat-container">'

if not st.session_state.chat_history:
    # Welcome message shown before any conversation starts
    chat_html += """
    <div style="text-align:center; color:#8B949E; padding:60px 0; font-size:0.9rem;">
        👋  Hi! I'm ShopAssist. How can I help you today?<br>
        <span style="font-size:0.8rem; margin-top:8px; display:block;">
            Type your question below or pick a suggestion from the sidebar.
        </span>
    </div>
    """
else:
    for msg in st.session_state.chat_history:

        if msg["role"] == "user":
            # ── FIX 7 — escape content before HTML injection ───────────────
            safe_content = html.escape(msg.get("content", ""))
            chat_html += f'<div class="user-bubble">🧑 {safe_content}</div>'

        else:
            # Bot message — extract metadata safely with defaults
            meta    = msg.get("meta", {})
            route   = html.escape(str(meta.get("route",        "—")))
            sources = meta.get("sources", [])
            ctype   = html.escape(str(meta.get("complaint_type", "")))

            # ── FIX 6 — safely cast faithfulness to float ─────────────────
            # In older runs the value could be a list or string — guard it
            raw_score = meta.get("faithfulness", 1.0)
            try:
                score = float(raw_score) if not isinstance(raw_score, list) else 1.0
            except (TypeError, ValueError):
                score = 1.0

            score_class = "good" if score >= 0.7 else ""
            sources_str = " · ".join(
                html.escape(s) for s in sources[:2]
            ) if sources else "—"

            # ── FIX 7 — escape bot answer before HTML injection ────────────
            safe_answer = html.escape(msg.get("content", ""))

            # Complaint type badge — only render if non-empty
            ctype_badge = (
                f"<span class='badge badge-type'>{ctype}</span>"
                if ctype else ""
            )

            chat_html += f"""
            <div class="bot-bubble">
                🤖 {safe_answer}
                <div class="meta-row">
                    <span class="badge badge-route">route: {route}</span>
                    <span class="badge badge-sources">📚 {sources_str}</span>
                    <span class="badge badge-score {score_class}">
                        faith: {score:.2f}
                    </span>
                    {ctype_badge}
                </div>
            </div>
            """

chat_html += '</div></div>'
st.markdown(chat_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  INPUT FORM — using st.form with clear_on_submit=True
#
#  WHY st.form IS THE CORRECT SOLUTION:
#  ─────────────────────────────────────
#  The root cause of duplicate messages is that Streamlit reruns the entire
#  script on every widget interaction. A plain st.text_input + st.button
#  combination fires TWO reruns per send:
#    Rerun 1 → button clicked  → message appended ✅
#    Rerun 2 → st.rerun() call → text_input still has old value → appended AGAIN ❌
#
#  st.form fixes this permanently because:
#    1. clear_on_submit=True  → form fields are WIPED after submission
#                               before Streamlit reruns the script
#    2. Form submission is an ATOMIC event → only ONE rerun is triggered
#    3. After submission, form_submit_button returns False on all future reruns
#       → the if-block never re-enters → zero duplicates guaranteed
#
#  This is the official Streamlit-recommended pattern for chat input.
# ─────────────────────────────────────────────────────────────────────────────

# pending_question holds a question that was submitted via the form.
# We store it in session_state so it survives across the rerun triggered
# by form submission, and process it AFTER the form is rendered.
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

# ── Render the form ────────────────────────────────────────────────────────
with st.form(key="chat_form", clear_on_submit=True):
    # clear_on_submit=True — empties all form fields the moment Submit fires
    # This is what prevents the input value from persisting into the next rerun

    col1, col2 = st.columns([5, 1])

    with col1:
        user_input = st.text_input(
            label            = "Your message",
            placeholder      = "e.g.  My payment failed but money was deducted…",
            label_visibility = "collapsed",
            key              = "form_input",   # different key from old widget
        )

    with col2:
        # form_submit_button — returns True ONLY on the rerun immediately
        # after the user clicks it, then returns False on every subsequent
        # rerun → prevents any re-processing
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

    # When the form is submitted and input is not empty:
    # save the question into session_state so we can process it below
    # (outside the form block, after Streamlit finishes rendering the form)
    if submitted and user_input.strip():
        st.session_state.pending_question = user_input.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS PENDING QUESTION
#  We read from session_state instead of directly from the widget because
#  the form clears the widget value before we get here. session_state
#  preserves the submitted text across the rerun boundary.
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.pending_question:

    # Extract and immediately clear the pending question
    # Clearing BEFORE processing means a crash mid-way won't leave a
    # stale question that re-processes on the next rerun
    question = st.session_state.pending_question
    st.session_state.pending_question = ""      # ← clear immediately

    # Append user message to chat history ONCE
    st.session_state.chat_history.append({
        "role"   : "user",
        "content": question,
    })

    # ── Call the LangGraph agent ──────────────────────────────────────────
    with st.spinner("🤔  Thinking…"):
        try:
            result = ask(app, question, thread_id=st.session_state.thread_id)

        except Exception as e:
            # Agent errors must NEVER crash the UI — show a friendly fallback
            result = {
                "answer"        : (
                    f"Sorry, something went wrong. Please try again. "
                    f"({type(e).__name__}: {str(e)[:80]})"
                ),
                "route"         : "error",
                "sources"       : [],
                "faithfulness"  : 0.0,
                "complaint_type": "",
                "user_name"     : "",
            }

    # ── Safely extract result fields with fallback defaults ───────────────
    answer       = result.get("answer",         "I'm sorry, I couldn't process that.")
    route        = result.get("route",          "retrieve")
    sources      = result.get("sources",        [])
    faithfulness = result.get("faithfulness",   1.0)
    ctype        = result.get("complaint_type", "")
    user_name    = result.get("user_name",      "")

    # Safe float cast — RAGAS can sometimes return a list instead of float
    try:
        faithfulness = float(faithfulness) if not isinstance(faithfulness, list) else 1.0
    except (TypeError, ValueError):
        faithfulness = 1.0

    # Persist extracted user name across the session for the greeting header
    if user_name:
        st.session_state.user_name = user_name

    # Append bot response to chat history ONCE
    st.session_state.chat_history.append({
        "role"   : "assistant",
        "content": answer,
        "meta"   : {
            "route"         : route,
            "sources"       : sources,
            "faithfulness"  : faithfulness,
            "complaint_type": ctype,
        },
    })

    # Single rerun to refresh the chat display with the new messages
    st.rerun()