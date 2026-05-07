"""
Agent Chat Page - Single interface for all user interactions with the AI agent
"""

import streamlit as st
import pandas as pd
import numpy as np
from data.loader import load_dataset
from agent.maintenance_agent import MaintenanceAgent


def dataframe_to_html_table(df, hide_index=True):
    """Convert DataFrame to styled HTML table for embedding in chat"""
    html = '<div style="overflow-x: auto; margin: 8px 0;">'
    html += df.to_html(escape=False, index=not hide_index, 
                       border=0, justify='left').replace('<table border="0">', 
                       '<table style="border-collapse: collapse; width: 100%; font-size: 0.9rem;">')
    html += """
    <style>
        table { font-family: 'Share Tech Mono', monospace; }
        th { background-color: #1a1a2e; color: #00e5ff; padding: 10px; text-align: left; 
             border-bottom: 2px solid #00e5ff; font-weight: bold; }
        td { padding: 8px; border-bottom: 1px solid #333; color: #e0e0e0; }
        tr:hover { background-color: #0f1419; }
    </style>
    """
    html += '</div>'
    return html


def _render_visualization(tool_name: str, visualization: dict):
    """
    Render visualization using native Streamlit components.
    Called once per tool call — each render is independent.
    """
    import plotly.graph_objects as go

    if not visualization:
        return

    # ── PREDICT FAILURE ──────────────────────────────────────
    if tool_name == "predict_failure" and visualization.get("failure_risk") is not None:
        risk       = visualization["failure_risk"]
        prediction = visualization["prediction"]
        status     = visualization["status"]
        confidence = visualization.get("confidence")
        confidence_label = visualization.get("confidence_label", "N/A")
        anomaly_detected = visualization.get("anomaly_detected", False)
        range_warnings = visualization.get("range_warnings", [])
        z_warnings = visualization.get("z_warnings", [])
        features   = visualization.get("features", {})

        # Backward compatibility for old cached responses without confidence fields.
        if confidence is None and prediction is not None:
            predicted_class_prob = (risk / 100.0) if prediction == 1 else (1.0 - risk / 100.0)
            confidence = float(np.clip(predicted_class_prob * 100.0, 0.0, 100.0))
            confidence_label = "EST."

        st.markdown("""
        <div style="background:rgba(0,229,255,0.04); border:1px solid rgba(0,229,255,0.2);
                    border-radius:8px; padding:12px; margin:8px 0;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem;
                        color:#00e5ff; letter-spacing:2px; margin-bottom:10px;">
                📊 FAILURE PREDICTION RESULTS
            </div>
        """, unsafe_allow_html=True)

        risk_color = "#ff3d57" if risk > 70 else "#ffb300" if risk > 40 else "#00e676"
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div style="background:#111827; border:1px solid {risk_color};
                        border-radius:6px; padding:12px; text-align:center;">
                <div style="font-size:0.65rem; color:#546e7a; font-family:'Share Tech Mono',monospace;">
                    FAILURE RISK</div>
                <div style="font-size:1.6rem; font-weight:700; color:{risk_color};
                            font-family:'Orbitron',monospace;">{risk:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            pred_color  = "#ff3d57" if prediction == 1 else "#00e676"
            pred_label  = "⚠ FAILURE" if prediction == 1 else "✓ NORMAL"
            st.markdown(f"""
            <div style="background:#111827; border:1px solid {pred_color};
                        border-radius:6px; padding:12px; text-align:center;">
                <div style="font-size:0.65rem; color:#546e7a; font-family:'Share Tech Mono',monospace;">
                    PREDICTION</div>
                <div style="font-size:1.1rem; font-weight:700; color:{pred_color};
                            font-family:'Orbitron',monospace;">{pred_label}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            conf_color = "#00e676" if (confidence or 0) >= 80 else "#ffb300" if (confidence or 0) >= 60 else "#ff3d57"
            conf_value = f"{confidence:.1f}%" if confidence is not None else "N/A"
            st.markdown(f"""
            <div style="background:#111827; border:1px solid {conf_color};
                        border-radius:6px; padding:12px; text-align:center;">
                <div style="font-size:0.65rem; color:#546e7a; font-family:'Share Tech Mono',monospace;">
                    CONFIDENCE</div>
                <div style="font-size:1.0rem; font-weight:700; color:{conf_color};
                            font-family:'Orbitron',monospace;">{conf_value}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div style="background:#111827; border:1px solid rgba(0,229,255,0.3);
                        border-radius:6px; padding:12px; text-align:center;">
                <div style="font-size:0.65rem; color:#546e7a; font-family:'Share Tech Mono',monospace;">
                    STATUS</div>
                <div style="font-size:0.9rem; font-weight:700; color:#00e5ff;
                            font-family:'Orbitron',monospace;">{status}</div>
                <div style="font-size:0.65rem; color:#546e7a; margin-top:2px;">{confidence_label}</div>
            </div>
            """, unsafe_allow_html=True)

        if anomaly_detected:
            warning_lines = []
            if range_warnings:
                warning_lines.append("Outside training min-max: " + "; ".join(range_warnings))
            if z_warnings:
                warning_lines.append("High z-score features: " + "; ".join(z_warnings))
            warning_text = "<br>".join(warning_lines) if warning_lines else "Input may be outside normal training distribution."

            st.markdown(f"""
            <div style="background:rgba(255,179,0,0.08); border:1px solid rgba(255,179,0,0.45);
                        border-radius:8px; padding:10px 12px; margin-top:10px;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem;
                            color:#ffb300; letter-spacing:1px; margin-bottom:6px;">
                    ⚠ INPUT RELIABILITY WARNING
                </div>
                <div style="font-size:0.78rem; color:#e0e0e0; line-height:1.5;">
                    {warning_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if features:
            st.markdown("<br>", unsafe_allow_html=True)
            # Format features safely - handle strings, None values, and numeric values
            features_data = []
            for k, v in features.items():
                if v is None:
                    formatted_value = "—"
                elif isinstance(v, str):
                    formatted_value = str(v)
                else:
                    try:
                        formatted_value = f"{float(v):.1f}"
                    except (ValueError, TypeError):
                        formatted_value = str(v)
                features_data.append({"Sensor": k, "Value": formatted_value})
            
            features_df = pd.DataFrame(features_data)
            st.dataframe(features_df, width="stretch", hide_index=True)
        
        # Display failure mode analysis if available
        failure_modes = visualization.get("failure_modes", [])
        if failure_modes:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem;
                        color:#ffb300; letter-spacing:1px; margin-bottom:8px;">
                ⚠️ POTENTIAL FAILURE MODES DETECTED
            </div>
            """, unsafe_allow_html=True)
            
            modes_data = []
            for mode in failure_modes:
                risk = mode.get('risk', 0)
                risk_color = "🔴" if risk > 70 else "🟡" if risk > 40 else "🟠"
                modes_data.append({
                    "Mode": f"{risk_color} {mode['mode']}",
                    "Name": mode.get('name', ''),
                    "Risk": f"{risk:.0f}%",
                    "Trigger": mode.get('trigger', '')
                })
            
            if modes_data:
                modes_df = pd.DataFrame(modes_data)
                st.dataframe(modes_df, width="stretch", hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── CALCULATE KPIs ───────────────────────────────────────
    elif tool_name == "calculate_kpis" and visualization.get("kpis"):
        kpis = visualization["kpis"]

        st.markdown("""
        <div style="background:rgba(0,229,255,0.04); border:1px solid rgba(0,229,255,0.2);
                    border-radius:8px; padding:12px; margin:8px 0;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem;
                        color:#00e5ff; letter-spacing:2px; margin-bottom:10px;">
                📈 OPERATIONAL KPIs
            </div>
        """, unsafe_allow_html=True)

        kpi_items = list(kpis.items())
        cols = st.columns(len(kpi_items))
        for col, (name, value) in zip(cols, kpi_items):
            with col:
                st.metric(label=name, value=str(value))

        total_ops = kpis.get("Total Operations")
        failures = kpis.get("Failures")
        if isinstance(total_ops, (int, float)) and isinstance(failures, (int, float)) and total_ops > 0:
            healthy = max(int(total_ops - failures), 0)
            fig = go.Figure(go.Bar(
                x=["Healthy Ops", "Failed Ops"],
                y=[healthy, int(failures)],
                marker_color=["#00e676", "#ff3d57"],
                text=[healthy, int(failures)],
                textposition="outside",
                textfont=dict(family="Share Tech Mono", size=10)
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.5)",
                height=230,
                font=dict(family="Share Tech Mono", size=9, color="#90a4ae"),
                margin=dict(l=20, r=20, t=44, b=20),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,229,255,0.05)")
            )
            fig.update_traces(cliponaxis=False)
            st.plotly_chart(fig, width="stretch")

        kpi_df = pd.DataFrame([{"Metric": k, "Value": str(v)} for k, v in kpis.items()])
        st.dataframe(kpi_df, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── ANALYZE PATTERNS ─────────────────────────────────────
    elif tool_name == "analyze_patterns" and visualization.get("patterns"):
        patterns = visualization["patterns"]

        st.markdown("""
        <div style="background:rgba(0,229,255,0.04); border:1px solid rgba(0,229,255,0.2);
                    border-radius:8px; padding:12px; margin:8px 0;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem;
                        color:#00e5ff; letter-spacing:2px; margin-bottom:10px;">
                📊 FAILURE MODE DISTRIBUTION
            </div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Bar(
            x=list(patterns.keys()),
            y=list(patterns.values()),
            marker_color=["#00e5ff","#ffb300","#ff3d57","#00e676","#b39ddb"],
            text=list(patterns.values()),
            textposition="outside",
            textfont=dict(family="Share Tech Mono", size=11)
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.5)", height=250,
            font=dict(family="Share Tech Mono", size=9, color="#90a4ae"),
            margin=dict(l=20, r=20, t=44, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,229,255,0.05)")
        )
        fig.update_traces(cliponaxis=False)
        st.plotly_chart(fig, width="stretch")

        patterns_df = pd.DataFrame([
            {"Failure Mode": k, "Count": v, "% of Failures": f"{v/sum(patterns.values())*100:.1f}%"}
            for k, v in sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(patterns_df, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── DIAGNOSE FAILURE ─────────────────────────────────────
    elif tool_name == "diagnose_failure" and visualization.get("diagnosis"):
        diagnosis = visualization["diagnosis"]

        st.markdown("""
        <div style="background:rgba(0,229,255,0.04); border:1px solid rgba(0,229,255,0.2);
                    border-radius:8px; padding:12px; margin:8px 0;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem;
                        color:#00e5ff; letter-spacing:2px; margin-bottom:10px;">
                🔍 FAILURE MODE DIAGNOSIS
            </div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Bar(
            x=list(diagnosis.values()),
            y=list(diagnosis.keys()),
            orientation="h",
            marker_color=["#ff3d57","#ffb300","#00e5ff","#00e676","#b39ddb"],
            text=list(diagnosis.values()),
            textposition="outside",
            textfont=dict(family="Share Tech Mono", size=10)
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.5)", height=260,
            font=dict(family="Share Tech Mono", size=9, color="#90a4ae"),
            margin=dict(l=20, r=40, t=44, b=20),
            xaxis=dict(showgrid=True, gridcolor="rgba(0,229,255,0.05)"),
            yaxis=dict(showgrid=False)
        )
        fig.update_traces(cliponaxis=False)
        st.plotly_chart(fig, width="stretch")

        diag_df = pd.DataFrame([
            {"Failure Mode": k, "Potential Cases": v} for k, v in diagnosis.items()
        ])
        st.dataframe(diag_df, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── MAINTENANCE SCHEDULE ─────────────────────────────────
    elif tool_name == "get_maintenance_schedule" and visualization.get("schedule"):
        schedule = visualization["schedule"]
        thresholds = visualization.get("thresholds", {})

        st.markdown("""
        <div style="background:rgba(0,229,255,0.04); border:1px solid rgba(0,229,255,0.2);
                    border-radius:8px; padding:12px; margin:8px 0;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem;
                        color:#00e5ff; letter-spacing:2px; margin-bottom:10px;">
                📋 MAINTENANCE SCHEDULE
            </div>
        """, unsafe_allow_html=True)

        if thresholds:
            method = str(thresholds.get("method", "quantile")).upper()
            q80 = thresholds.get("q80_high")
            q95 = thresholds.get("q95_urgent")
            if q80 is not None and q95 is not None:
                st.caption(f"Thresholds used ({method}): Q80 HIGH = {q80:.2f} min, Q95 URGENT = {q95:.2f} min")

        chart_df = pd.DataFrame(schedule)
        if not chart_df.empty and "Priority" in chart_df.columns and "Units" in chart_df.columns:
            fig = go.Figure(go.Bar(
                x=chart_df["Priority"],
                y=chart_df["Units"],
                marker_color=["#ff3d57", "#ffb300", "#00e676"],
                text=chart_df["Units"],
                textposition="outside",
                textfont=dict(family="Share Tech Mono", size=10)
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.5)",
                height=230,
                font=dict(family="Share Tech Mono", size=9, color="#90a4ae"),
                margin=dict(l=20, r=20, t=44, b=20),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,229,255,0.05)")
            )
            fig.update_traces(cliponaxis=False)
            st.plotly_chart(fig, width="stretch")

        schedule_df = pd.DataFrame(schedule)
        st.dataframe(schedule_df, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── CHECK OPERATING LIMITS ───────────────────────────────
    elif tool_name == "check_operating_limits" and visualization.get("limits"):
        limits = visualization["limits"]

        st.markdown("""
        <div style="background:rgba(0,229,255,0.04); border:1px solid rgba(0,229,255,0.2);
                    border-radius:8px; padding:12px; margin:8px 0;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem;
                        color:#00e5ff; letter-spacing:2px; margin-bottom:10px;">
                ⚙️ OPERATING LIMITS VALIDATION
            </div>
        """, unsafe_allow_html=True)

        limits_df = pd.DataFrame(limits)

        # If user passed sensor values, show Value/Min/Max comparison chart.
        if not limits_df.empty and all(col in limits_df.columns for col in ["Sensor", "Value", "Min", "Max"]):
            chart_df = limits_df.copy()
            chart_df["Value"] = pd.to_numeric(chart_df["Value"], errors="coerce")
            chart_df["Min"] = pd.to_numeric(chart_df["Min"], errors="coerce")
            chart_df["Max"] = pd.to_numeric(chart_df["Max"], errors="coerce")
            chart_df = chart_df.dropna(subset=["Value", "Min", "Max"])

            if not chart_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=chart_df["Sensor"],
                    y=chart_df["Min"],
                    name="Min",
                    marker_color="#546e7a"
                ))
                fig.add_trace(go.Bar(
                    x=chart_df["Sensor"],
                    y=chart_df["Value"],
                    name="Current",
                    marker_color="#00e5ff"
                ))
                fig.add_trace(go.Bar(
                    x=chart_df["Sensor"],
                    y=chart_df["Max"],
                    name="Max",
                    marker_color="#ffb300"
                ))
                fig.update_layout(
                    barmode="group",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(17,24,39,0.5)",
                    height=260,
                    font=dict(family="Share Tech Mono", size=9, color="#90a4ae"),
                    margin=dict(l=20, r=20, t=44, b=20),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(0,229,255,0.05)")
                )
                fig.update_traces(cliponaxis=False)
                st.plotly_chart(fig, width="stretch")

        # If no user values were passed, show violation counts chart.
        elif not limits_df.empty and all(col in limits_df.columns for col in ["Sensor", "Violations"]):
            chart_df = limits_df.copy()
            chart_df["Violations"] = pd.to_numeric(chart_df["Violations"], errors="coerce").fillna(0)
            fig = go.Figure(go.Bar(
                x=chart_df["Sensor"],
                y=chart_df["Violations"],
                marker_color="#ff3d57",
                text=chart_df["Violations"].astype(int),
                textposition="outside",
                textfont=dict(family="Share Tech Mono", size=10)
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.5)",
                height=230,
                font=dict(family="Share Tech Mono", size=9, color="#90a4ae"),
                margin=dict(l=20, r=20, t=44, b=20),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,229,255,0.05)")
            )
            fig.update_traces(cliponaxis=False)
            st.plotly_chart(fig, width="stretch")

        st.dataframe(limits_df, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── STATISTICAL SUMMARY ──────────────────────────────────
    elif tool_name == "statistical_summary" and visualization.get("statistics"):
        stats = visualization["statistics"]

        st.markdown("""
        <div style="background:rgba(0,229,255,0.04); border:1px solid rgba(0,229,255,0.2);
                    border-radius:8px; padding:12px; margin:8px 0;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem;
                        color:#00e5ff; letter-spacing:2px; margin-bottom:10px;">
                🧮 STATISTICAL SUMMARY
            </div>
        """, unsafe_allow_html=True)

        stats_df = pd.DataFrame(stats).T.reset_index()
        stats_df.columns = ["Sensor", "Mean", "Std Dev", "Min", "Max"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stats_df["Sensor"],
            y=stats_df["Mean"],
            error_y=dict(type="data", array=stats_df["Std Dev"], visible=True),
            marker_color="#00e5ff",
            text=[f"{v:.1f}" for v in stats_df["Mean"]],
            textposition="outside",
            textfont=dict(family="Share Tech Mono", size=10)
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.5)",
            height=280,
            font=dict(family="Share Tech Mono", size=9, color="#90a4ae"),
            margin=dict(l=20, r=20, t=44, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,229,255,0.05)", title="Mean Value")
        )
        fig.update_traces(cliponaxis=False)
        st.plotly_chart(fig, width="stretch")

        st.dataframe(stats_df, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_agent_chat_page(llm_provider: str = "Groq (llama3.3-70b)"):
    # Increment when prediction visualization schema changes.
    agent_schema_version = "2026-04-27-confidence-v1"

    st.markdown('<div class="section-header">// 🤖 MAINTENANCE INTELLIGENCE AGENT</div>',
                unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .quick-help-sticky {
            position: fixed;
            top: 92px;
            right: 24px;
            width: min(380px, 28vw);
            max-height: calc(100vh - 124px);
            overflow-y: auto;
            z-index: 50;
        }

        /* Match bottom input width to main chat column (leave room for fixed help panel). */
        [data-testid="stChatInput"] {
            margin-right: calc(min(380px, 28vw) + 36px) !important;
        }

        /* Keep layout usable on smaller screens */
        @media (max-width: 1200px) {
            .quick-help-sticky {
                position: static;
                width: 100%;
                max-height: none;
                overflow-y: visible;
            }

            [data-testid="stChatInput"] {
                margin-right: 0 !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_main, col_help = st.columns([3.5, 1.5])

    with col_main:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "agent_instance" not in st.session_state:
            st.session_state.agent_instance = None
        if "agent_provider" not in st.session_state:
            st.session_state.agent_provider = None
        if "agent_schema_version" not in st.session_state:
            # Existing sessions from previous code may hold stale agent objects.
            if st.session_state.agent_instance is not None:
                st.session_state.agent_instance = None
                st.session_state.agent_provider = None
            st.session_state.agent_schema_version = agent_schema_version

        # Force reinit after schema changes so tool outputs include new fields.
        if st.session_state.agent_schema_version != agent_schema_version:
            st.session_state.agent_instance = None
            st.session_state.agent_provider = None
            st.session_state.agent_schema_version = agent_schema_version

        # ── Empty state ──────────────────────────────────────
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#37474f;">
                <div style="font-size:3rem; margin-bottom:20px;">🤖</div>
                <div style="font-family:'Orbitron',monospace; font-size:1rem;
                            color:#00e5ff; letter-spacing:2px; margin-bottom:10px;">
                    AGENTIC AI READY
                </div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.85rem;
                            color:#546e7a; margin-top:15px;">
                    Ask me about machine failures, KPIs, maintenance schedules, or sensor analysis.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Render each message separately ───────────────────
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    # User bubble
                    st.markdown(f"""
                    <div class="chat-msg-user">
                        <div class="chat-role-user">▶ YOU</div>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    # Role header (HTML)
                    st.markdown("""
                    <div class="chat-msg-agent">
                        <div class="chat-role-agent">⚙ AI AGENT</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Content (MARKDOWN — no HTML wrapper)
                    st.markdown(msg['content'])

                    # Render each tool call visualization SEPARATELY (deduped)
                    if "tool_calls" in msg and msg["tool_calls"]:
                        rendered_visualizations = set()  # Track rendered tools to avoid duplicates
                        for tc in msg["tool_calls"]:
                            tool_name     = tc.get('tool', '')
                            visualization = tc.get('visualization', {})

                            # Skip if we've already rendered this tool's visualization in this message
                            if tool_name in rendered_visualizations:
                                continue
                            
                            # ── Render visualization using Streamlit native components ──
                            if visualization and visualization.get('tool'):
                                _render_visualization(tool_name, visualization)
                                rendered_visualizations.add(tool_name)

                            # Tool badge
                            st.markdown(f"""
                            <div class="agent-thinking" style="margin-top:4px;">
                                🔧 Used Tool: <span style="color:#ffb300;">{tool_name}</span>
                            </div>
                            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Help panel ────────────────────────────────────────────
    with col_help:
        st.markdown("""
        <div class="quick-help-sticky" style="background:#111827; border:1px solid rgba(0,229,255,0.15);
                    border-radius:8px; padding:12px;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem;
                        color:#546e7a; letter-spacing:2px; margin-bottom:8px;">📚 QUICK HELP</div>
            <div style="font-size:0.75rem; color:#90a4ae; line-height:1.9;">
                <div style="color:#00e5ff; margin-bottom:6px; font-size:0.7rem;">AGENT TOOLS</div>
                🔮 Failure Prediction<br>📊 Pattern Analysis<br>🔍 Failure Diagnosis<br>
                📋 Maintenance Schedule<br>📈 KPI Calculation<br>🧮 Statistical Summary<br>
                ⚙️ Limits Validation<br><br>
                <div style="color:#00e5ff; margin-bottom:6px; font-size:0.7rem;">TRY ASKING</div>
                • Predict failure for RPM=1500, Torque=45, ToolWear=210, AirTemp=298, ProcessTemp=308, Type=M<br>
                • Predict failure for RPM=1200, Torque=60, ToolWear=230, AirTemp=295, ProcessTemp=302, Type=H<br>
                • Predict failure for RPM=2000, Torque=50, ToolWear=180, AirTemp=300, ProcessTemp=310, Type=L<br>
                • Analyze failure patterns in the dataset<br>
                • Diagnose all failure modes<br>
                • Get maintenance schedule<br>
                • Calculate KPIs<br>
                • Are these safe values? RPM=5000, Torque=100
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Keep chat input at page bottom (ChatGPT-like) by rendering it at top-level, not inside columns.
    user_input = st.chat_input(
        "Ask about predictions, KPIs, patterns, schedules, statistics..."
    )

    if user_input:
        if (st.session_state.agent_instance is None or
                st.session_state.agent_provider != llm_provider):
            with st.spinner("⚙ Initializing agent..."):
                df = load_dataset()
                st.session_state.agent_instance = MaintenanceAgent(
                    df, api_key=None, provider=llm_provider
                )
                st.session_state.agent_provider = llm_provider

        with st.spinner("🔄 Agent processing your query..."):
            response = st.session_state.agent_instance.run(user_input)

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({
            "role":       "assistant",
            "content":    response["answer"],
            "tool_calls": response.get("tool_calls", [])
        })
        st.rerun()