"""
Anywhere Agentic AI System for Predictive Maintenance
========================================================
Single-Page AI Agent Interface

The system is built around a conversational AI agent that handles all user interactions.
Users ask questions, the agent selects appropriate tools and returns insights with visualizations.
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Anywhere | Agentic AI System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/anywhere/predictive-maintenance",
        "About": "Anywhere Agentic AI for Predictive Maintenance v1.0"
    }
)

from styles import inject_custom_css
from sidebar import render_agent_sidebar
from pages import render_agent_chat_page

def main():
    inject_custom_css()
    
    # Render agent configuration sidebar
    llm_provider = render_agent_sidebar()
    
    # Main header
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.markdown("""
        <div class="main-header">
            <div class="header-icon">⚙️</div>
            <div class="header-text">
                <h1>
                    A<span class="ai-highlight">ny</span>where 
                    <span class="accent">Agentic AI</span>
                </h1>
                <p class="header-sub">Conversational Intelligence for Predictive Maintenance</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(
        """
        <style>
        .main-chat-divider {
            height: 1px;
            background: rgba(0, 229, 255, 0.18);
            margin: 10px calc(min(380px, 28vw) + 24px) 18px 0;
        }

        @media (max-width: 1200px) {
            .main-chat-divider {
                margin-right: 0;
            }
        }
        </style>
        <div class="main-chat-divider"></div>
        """,
        unsafe_allow_html=True,
    )
    
    # Single agent interface - everything goes through the agent
    render_agent_chat_page(llm_provider)

if __name__ == "__main__":
    main()