# -----------------------------
# Imports
# -----------------------------
import streamlit as st
from auth import init_auth_state, login_page, signup_page, welcome_page
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
import os
import time
from datetime import datetime
from torch_geometric.nn import GCNConv, knn_graph
from torch_geometric.data import Data
from torchdiffeq import odeint


# -----------------------------
# Main Application
# -----------------------------
def main_app():
    # from twilio.rest import Client  # SMS functionality commented out

    # -----------------------------
    # Page config
    # -----------------------------
    st.set_page_config(
        page_title="InjuryGuard AI",
        page_icon="AILOGO.png",
        layout="wide"
    )

    # -----------------------------
    # App Header
    # -----------------------------
    import base64
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    image_base64 = get_base64_image("AILOGO.png")

    st.markdown(
        f"""
        <div style='display: flex; align-items: center; justify-content: center; gap: 15px;'>
            <img src="data:image/png;base64,{image_base64}" width="80" style="vertical-align: middle;">
            <h1 style='color: #1f77b4; font-size:60px; display: inline-block; margin: 0;'>InjuryGuard AI</h1>
        </div>
        <p style='text-align: center; font-style: italic; font-size:15px; margin-top:5px;'>
        Shifting from Reactive Treatment to Proactive Prevention
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # -----------------------------
    # Styling
    # -----------------------------
    st.markdown("""
        <style>
        h1, h2, h3, .stSubheader {
            color: #1f77b4 !important;
        }
        div.stButton > button:first-child {
            background-color: #1f77b4;
            color: white;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    # -----------------------------
    # Helper functions
    # -----------------------------
    def get_recommendations(risk_score):
        if risk_score > 0.7:
            return [
                "Immediate training intensity reduction",
                "Electrolyte supplementation required",
                "Consult sports physician within 24h",
                "Enhanced hydration protocol"
            ]
        elif risk_score > 0.4:
            return [
                "Modify training load by 30%",
                "Ensure 8+ hours recovery sleep",
                "Monitor hydration levels",
                "Re-assess in 48 hours"
            ]
        else:
            return [
                "Maintain current training regimen",
                "Continue standard monitoring",
                "Optimal recovery conditions"
            ]

    def display_results(risk_score, probabilities, biomarkers):
        risk_level = "HIGH" if risk_score > 0.7 else "MODERATE" if risk_score > 0.4 else "LOW"
        risk_color = "red" if risk_level == "HIGH" else "orange" if risk_level == "MODERATE" else "green"

        st.success("### Assessment Complete!")

        col1, col2, col3 = st.columns(3)

        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_score*100,
                domain={'x': [0,1], 'y': [0,1]},
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': risk_color},
                    'steps': [{'range': [0,30], 'color': "lightgreen"},
                              {'range': [30,70], 'color': "yellow"},
                              {'range': [70,100], 'color': "red"}]})
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("Risk Level", risk_level)
            st.metric("Injury Risk Probability", f"{risk_score:.1%}")
            st.info("**Risk Indicators:**")
            if biomarkers['rms_feat'] > 1.0: st.warning("• Elevated muscle fatigue")
            if biomarkers['tissue_sweat'] < -1.0: st.warning("• Abnormal sweat conductivity")
            if risk_level == "LOW": st.success("• Biomarkers within normal range")

        with col3:
            st.subheader("Recommendations")
            recommendations = get_recommendations(risk_score)
            for rec in recommendations:
                if risk_level == "HIGH":
                    st.error(f"• {rec}")
                elif risk_level == "MODERATE":
                    st.warning(f"• {rec}")
                else:
                    st.success(f"• {rec}")

    # -----------------------------
    # GNODE Model Definition + Load
    # -----------------------------
    class ODEFunc(nn.Module):
        def __init__(self, in_channels, hidden_channels, edge_index):
            super(ODEFunc, self).__init__()
            self.edge_index = edge_index
            self.gc1 = GCNConv(in_channels, hidden_channels)
            self.gc2 = GCNConv(hidden_channels, hidden_channels)

        def forward(self, t, x):
            x = self.gc1(x, self.edge_index)
            x = F.relu(x)
            x = self.gc2(x, self.edge_index)
            return x

    class GNODEModel(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, edge_index):
            super(GNODEModel, self).__init__()
            self.edge_index = edge_index
            self.input_proj = nn.Linear(in_channels, hidden_channels)
            self.odefunc = ODEFunc(hidden_channels, hidden_channels, edge_index)
            self.linear = nn.Linear(hidden_channels, out_channels)

        def forward(self, x):
            x = self.input_proj(x)
            x = F.relu(x)
            t = torch.tensor([0,1], dtype=torch.float)
            out = odeint(self.odefunc, x, t, method='rk4', options={'step_size':0.1})
            out = out[-1]
            out = self.linear(out)
            return out

    @st.cache_resource
    def load_gnode_model():
        model_path = "gnode_model.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            edge_index = torch.tensor([[0],[0]], dtype=torch.long)
            model = GNODEModel(in_channels=7, hidden_channels=32, out_channels=2, edge_index=edge_index)
            try:
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            except Exception as e:
                st.warning(f"State dict loading issue: {e}")
            model.eval()
            return model
        else:
            st.warning("gnode_model.pth not found.")
            return None

    model = load_gnode_model()

    # -----------------------------
    # Session State Init
    # -----------------------------
    if 'players' not in st.session_state: st.session_state.players = {}
    if 'current_player' not in st.session_state: st.session_state.current_player = None
    if 'live_data' not in st.session_state: st.session_state.live_data = {}
    if 'biosensor_running' not in st.session_state: st.session_state.biosensor_running = False

    # -----------------------------
    # Sidebar & Footer
    # -----------------------------
    st.sidebar.header("👥 Player Management")
    # ... your tabs, sidebar inputs, etc.

    st.markdown("---")
    st.markdown("*AI Sports Medicine Platform • Real-time Biosensor Monitoring • Professional Athlete Management*")


# -----------------------------
# Authentication Routing
# -----------------------------
init_auth_state()

if not st.session_state.logged_in:
    if st.session_state.auth_page == 'login':
        login_page()
    elif st.session_state.auth_page == 'signup':
        signup_page()
    else:
        st.session_state.auth_page = 'login'
    st.stop()
elif st.session_state.auth_page == 'welcome':
    welcome_page()
    st.stop()
elif st.session_state.auth_page == 'app':
    main_app()
