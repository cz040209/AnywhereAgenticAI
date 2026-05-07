"""
UI Styles - Custom CSS for Anywhere Predictive Maintenance System
Industrial dark theme with electric cyan accents
"""

import streamlit as st


def inject_custom_css():
    st.markdown("""
    <style>
    /* ====== FONTS ====== */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;500;600&display=swap');

    /* ====== ROOT VARIABLES ====== */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #0f1628;
        --bg-card: #111827;
        --bg-card-hover: #1a2035;
        --accent-cyan: #00e5ff;
        --accent-amber: #ffb300;
        --accent-red: #ff3d57;
        --accent-green: #00e676;
        --text-primary: #e8eaf6;
        --text-secondary: #90a4ae;
        --text-muted: #546e7a;
        --border-color: rgba(0, 229, 255, 0.15);
        --glow-cyan: 0 0 20px rgba(0, 229, 255, 0.3);
        --glow-amber: 0 0 20px rgba(255, 179, 0, 0.3);
    }

    /* ====== GLOBAL RESET ====== */
    .stApp {
        background: var(--bg-primary) !important;
        background-image: 
            radial-gradient(ellipse at 20% 20%, rgba(0, 229, 255, 0.03) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(255, 179, 0, 0.02) 0%, transparent 50%);
        font-family: 'Exo 2', sans-serif !important;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}

    /* ====== MAIN HEADER ====== */
    .main-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 20px 0 8px 0;
        border-bottom: 1px solid var(--border-color);
        margin-right: calc(min(380px, 28vw) + 4px);
        margin-bottom: 24px;
    }
    .header-icon {
        font-size: 2.5rem;
        animation: spin-slow 8s linear infinite;
    }
    @keyframes spin-slow {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .header-text h1 {
        font-family: 'Orbitron', monospace !important;
        font-size: clamp(1.35rem, 1.95vw, 1.8rem) !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        letter-spacing: 2px;
        white-space: nowrap;
    }
    .header-text .accent {
        color: var(--accent-cyan);
        text-shadow: var(--glow-cyan);
    }
    .header-sub {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-muted);
        letter-spacing: 3px;
        text-transform: uppercase;
        margin: 4px 0 0 0;
    }
    .ai-highlight {
        color: var(--accent-amber);
        text-shadow: var(--glow-amber);
    }

    /* ====== SIDEBAR ====== */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary) !important;
    }

    /* ====== METRIC CARDS ====== */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-cyan), transparent);
    }
    .metric-card:hover {
        border-color: rgba(0, 229, 255, 0.4);
        box-shadow: var(--glow-cyan);
        transform: translateY(-2px);
    }
    .metric-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-cyan);
        text-shadow: var(--glow-cyan);
        line-height: 1;
    }
    .metric-value.amber { color: var(--accent-amber); text-shadow: var(--glow-amber); }
    .metric-value.red { color: var(--accent-red); text-shadow: 0 0 20px rgba(255,61,87,0.3); }
    .metric-value.green { color: var(--accent-green); text-shadow: 0 0 20px rgba(0,230,118,0.3); }
    .metric-delta {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 4px;
    }

    /* ====== SECTION HEADERS ====== */
    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--accent-cyan);
        text-transform: uppercase;
        letter-spacing: 3px;
        padding: 12px 0 8px 0;
        border-bottom: 1px solid var(--border-color);
        margin-right: calc(min(380px, 28vw) + 24px);
        margin-bottom: 16px;
    }

    @media (max-width: 1200px) {
        .main-header,
        .section-header {
            margin-right: 0;
        }
    }

    /* ====== CHAT INTERFACE ====== */
    .chat-container {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 8px;
        max-height: 500px;
        overflow-y: auto;
    }
    .chat-msg-user {
        background: rgba(0, 229, 255, 0.08);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-radius: 12px 12px 2px 12px;
        padding: 12px 16px;
        margin: 8px 0 8px 60px;
        color: var(--text-primary);
        font-size: 0.9rem;
    }
    .chat-msg-agent {
        background: rgba(255, 179, 0, 0.05);
        border: 1px solid rgba(255, 179, 0, 0.2);
        border-radius: 2px 12px 12px 12px;
        padding: 12px 16px;
        margin: 8px 60px 8px 0;
        color: var(--text-primary);
        font-size: 0.9rem;
    }
    .chat-role-user {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.65rem;
        color: var(--accent-cyan);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 4px;
    }
    .chat-role-agent {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.65rem;
        color: var(--accent-amber);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 4px;
    }
    .agent-thinking {
        background: rgba(0, 229, 255, 0.04);
        border-left: 3px solid var(--accent-cyan);
        padding: 8px 12px;
        border-radius: 0 8px 8px 0;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-muted);
        margin: 4px 0;
    }

    /* ====== ALERT CARDS ====== */
    .alert-card {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 14px 16px;
        margin: 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
        border: 1px solid var(--border-color);
    }
    .alert-card.critical { border-left: 4px solid var(--accent-red); }
    .alert-card.warning { border-left: 4px solid var(--accent-amber); }
    .alert-card.normal { border-left: 4px solid var(--accent-green); }
    .alert-icon { font-size: 1.2rem; }
    .alert-title {
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--text-primary);
        font-family: 'Exo 2', sans-serif;
    }
    .alert-desc {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-family: 'Share Tech Mono', monospace;
    }

    /* ====== PREDICTION BADGE ====== */
    .pred-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .pred-badge.fail { background: rgba(255,61,87,0.15); color: var(--accent-red); border: 1px solid var(--accent-red); }
    .pred-badge.ok { background: rgba(0,230,118,0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }

    /* ====== TOOL ACTIVITY ====== */
    .tool-log {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
        background: #050810;
        border: 1px solid rgba(0, 229, 255, 0.1);
        border-radius: 8px;
        padding: 12px;
        color: var(--accent-cyan);
        line-height: 1.8;
    }
    .tool-log .tool-call { color: var(--accent-amber); }
    .tool-log .tool-result { color: var(--accent-green); }

    /* ====== TABLES ====== */
    [data-testid="stDataFrame"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }

    /* ====== BUTTONS ====== */
    .stButton button {
        background: transparent !important;
        border: 1px solid var(--accent-cyan) !important;
        color: var(--accent-cyan) !important;
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        transition: all 0.2s ease !important;
        border-radius: 6px !important;
    }
    .stButton button:hover {
        background: rgba(0, 229, 255, 0.1) !important;
        box-shadow: var(--glow-cyan) !important;
    }
    .stButton.primary-btn button {
        background: var(--accent-cyan) !important;
        color: #000 !important;
        font-weight: 700 !important;
    }

    /* ====== INPUTS ====== */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        font-family: 'Exo 2', sans-serif !important;
        border-radius: 6px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: var(--glow-cyan) !important;
    }

    /* ====== PROGRESS ====== */
    .stProgress > div > div {
        background: var(--accent-cyan) !important;
    }

    /* ====== STATUS INDICATOR ====== */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    .status-dot.online { background: var(--accent-green); }
    .status-dot.warning { background: var(--accent-amber); }
    .status-dot.offline { background: var(--accent-red); }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }

    /* ====== EXPANDER ====== */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* ====== TABS ====== */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border-color) !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-muted) !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 1px !important;
        border: none !important;
        padding: 8px 16px !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-cyan) !important;
        border-bottom: 2px solid var(--accent-cyan) !important;
    }

    /* ====== SCROLLBAR ====== */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--accent-cyan); border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

    /* ====== INFO BOXES ====== */
    .info-box {
        background: rgba(0, 229, 255, 0.04);
        border: 1px solid rgba(0, 229, 255, 0.15);
        border-radius: 8px;
        padding: 14px;
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    .info-box strong { color: var(--accent-cyan); }

    /* ====== ARCHITECTURE DIAGRAM ====== */
    .arch-node {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-primary);
        margin: 4px;
        transition: all 0.3s ease;
    }
    .arch-node:hover {
        border-color: var(--accent-cyan);
        box-shadow: var(--glow-cyan);
    }
    .arch-node.core {
        border-color: var(--accent-cyan);
        color: var(--accent-cyan);
        background: rgba(0, 229, 255, 0.05);
    }
    .arch-node.tool {
        border-color: rgba(255, 179, 0, 0.3);
        color: var(--accent-amber);
        background: rgba(255, 179, 0, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)