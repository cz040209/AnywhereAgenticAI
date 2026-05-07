"""
Agent Config Panel - Settings for the AI Agent System
"""

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

def render_agent_sidebar():
    """Render agent configuration in sidebar"""
    with st.sidebar:
        # Header
        st.markdown("""
        <div style="padding: 16px 0 8px 0; border-bottom: 1px solid rgba(0,229,255,0.15); margin-bottom: 16px;">
            <div style="font-family: 'Orbitron', monospace; font-size: 0.8rem; font-weight: 700; 
                        color: #00e5ff; letter-spacing: 3px; text-transform: uppercase;">
                ⚙️ ANYWHERE
            </div>
            <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; 
                        color: #546e7a; letter-spacing: 2px; margin-top: 4px;">
                AGENTIC AI SYSTEM v1.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        # System status
        st.markdown("""
        <div style="margin-bottom: 16px; padding: 10px; background: #111827; 
                    border: 1px solid rgba(0,229,255,0.1); border-radius: 8px;">
            <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; 
                        color: #546e7a; letter-spacing: 2px; margin-bottom: 8px;">🟢 SYSTEM STATUS</div>
            <div style="font-size: 0.75rem; color: #e8eaf6; margin: 3px 0;">
                <span style="display:inline-block; width:7px; height:7px; 
                background:#00e676; border-radius:50%; margin-right:6px;"></span>AI Agent Online
            </div>
            <div style="font-size: 0.75rem; color: #e8eaf6; margin: 3px 0;">
                <span style="display:inline-block; width:7px; height:7px; 
                background:#00e676; border-radius:50%; margin-right:6px;"></span>ML Models Loaded
            </div>
            <div style="font-size: 0.75rem; color: #e8eaf6; margin: 3px 0;">
                <span style="display:inline-block; width:7px; height:7px; 
                background:#ffb300; border-radius:50%; margin-right:6px;"></span>LLM Connected
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Agent Configuration
        st.markdown("""
        <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; 
                    color: #546e7a; letter-spacing: 2px; margin-bottom: 8px;">🤖 AGENT CONFIG</div>
        """, unsafe_allow_html=True)

        llm_provider = st.selectbox(
            "LLM Provider",
            ["Groq (llama3.3-70b)", "Google Gemini", "DeepSeek (via OpenRouter)"],
            label_visibility="collapsed",
            key="llm_provider_select"
        )

        st.markdown("""
        <div style="font-size: 0.7rem; color: #546e7a; margin-top: 12px;">
            <strong>📋 Agent Capabilities (7 Tools):</strong>
        </div>
        """, unsafe_allow_html=True)

        capabilities = [
            "🔮 Failure Prediction",
            "📊 Pattern Analysis",
            "🔍 Diagnosis",
            "📋 Maintenance Schedule",
            "📈 KPI Calculation",
            "🧮 Statistics",
            "⚙️ Limits Validation"
        ]
        
        for cap in capabilities:
            st.markdown(f"""
            <div style="font-size: 0.7rem; color: #90a4ae; margin: 2px 0;">
                ✓ {cap}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; 
                    color: #37474f; text-align: center; line-height: 1.8;">
            <span style="color: #90a4ae;">powered by Anywhere</span><br>
            <span style="color: #37474f; font-size: 0.6rem;">Agentic AI System</span>
        </div>
        """, unsafe_allow_html=True)

        return llm_provider