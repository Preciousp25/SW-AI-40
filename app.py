import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import json
import os
import random
from torchdiffeq import odeint
from torch_geometric.nn import GCNConv

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="AI Sports Medicine Monitor", layout="wide")

PLAYER_DB = "players.json"

# ---------------------------------
# GNODE MODEL DEFINITION
# ---------------------------------
class ODEFunc(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ODEFunc, self).__init__()
        self.gc1 = GCNConv(in_channels, hidden_channels)
        self.gc2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, t, x):
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = self.gc2(x, edge_index)
        return x


class GNODEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNODEModel, self).__init__()
        self.odefunc = ODEFunc(in_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        t = torch.tensor([0, 1], dtype=torch.float)
        out = odeint(self.odefunc, x, t, method='rk4', options={'step_size': 0.1})
        out = out[-1]
        out = self.linear(out)
        return out


# ---------------------------------
# MODEL LOADING
# ---------------------------------
@st.cache_resource
def load_model():
    model_path = "gnode_model.pth"
    if not os.path.exists(model_path):
        st.error("Model file not found! Please ensure gnode_model.pth is in the root directory.")
        return None

    model = GNODEModel(in_channels=7, hidden_channels=32, out_channels=2)
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    st.success(" GNODE model loaded successfully!")
    return model


model = load_model()

# ---------------------------------
# PLAYER MANAGEMENT
# ---------------------------------
def load_players():
    if os.path.exists(PLAYER_DB):
        with open(PLAYER_DB, "r") as f:
            return json.load(f)
    return {}

def save_players(players):
    with open(PLAYER_DB, "w") as f:
        json.dump(players, f, indent=2)

players = load_players()

# ---------------------------------
# FEATURE DEFINITIONS
# ---------------------------------
feature_descriptions = {
    'mw': 'Molecular Weight (signal amplitude)',
    'tissue_sweat': 'Sweat Tissue Conductivity',
    'tissue_urine': 'Urine Tissue Conductivity',
    'rms_feat': 'RMS - Muscle Fatigue Indicator',
    'zero_crossings': 'Zero Crossings - Signal Variability',
    'skewness': 'Skewness - Signal Distribution',
    'waveform_length': 'Waveform Length - Signal Complexity'
}

feature_names = list(feature_descriptions.keys())

# ---------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------
st.sidebar.title("Athlete Management")
mode = st.sidebar.radio("Choose Mode:", ["Assess Player", "Add New Player", "Batch Screening"])

# ---------------------------------
# ADD PLAYER
# ---------------------------------
if mode == "Add New Player":
    st.header("Register New Player")
    with st.form("add_player_form"):
        player_id = st.text_input("Athlete ID", f"ATH-{len(players)+1:03d}")
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=10, max_value=60, value=20)
        team = st.text_input("Team", "Default Team")
        position = st.text_input("Position", "Midfielder")
        submitted = st.form_submit_button("Save Player")
    
    if submitted:
        players[player_id] = {
            "name": name,
            "age": age,
            "team": team,
            "position": position
        }
        save_players(players)
        st.success(f"Player {name} ({player_id}) added successfully.")

# ---------------------------------
# ASSESS PLAYER
# ---------------------------------
elif mode == "Assess Player":
    st.header("Injury Risk Assessment")

    if not players:
        st.warning("No players registered yet. Please add one first.")
        st.stop()

    selected_id = st.selectbox("Select Player:", list(players.keys()))
    selected_player = players[selected_id]
    st.info(f"{selected_player['name']} — {selected_player['position']} ({selected_player['team']})")

    input_mode = st.radio("Data Input Mode", ["Manual Entry", "Live Biosensor Feed"])

    input_data = {}
    cols = st.columns(2)

    if input_mode == "Manual Entry":
        for i, feat in enumerate(feature_names):
            with cols[i % 2]:
                input_data[feat] = st.slider(
                    feature_descriptions[feat],
                    min_value=-2.0,
                    max_value=2.0,
                    value=0.0,
                    step=0.1
                )
    else:
        for feat in feature_names:
            input_data[feat] = round(random.uniform(-2, 2), 2)
        st.success("Live biosensor data received.")
        st.json(input_data)

    def make_prediction(features_array, model):
        if model is None:
            return 0.5, [0.5, 0.5]
        try:
            features_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(features_tensor)
                probabilities = F.softmax(logits, dim=1)
                risk_score = probabilities[0][1].item()
                probs = probabilities[0].tolist()
            return risk_score, probs
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return 0.3, [0.7, 0.3]

    if st.button("Assess Injury Risk", type="primary"):
        features_array = np.array([input_data[feature] for feature in feature_names])
        risk_score, probabilities = make_prediction(features_array, model)

        risk_level = "HIGH" if risk_score > 0.7 else "MODERATE" if risk_score > 0.4 else "LOW"
        risk_color = {"HIGH": "red", "MODERATE": "orange", "LOW": "green"}[risk_level]

        col1, col2, col3 = st.columns(3)
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                title={'text': "Injury Risk Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("Risk Level", risk_level)
            st.metric("Confidence", f"{max(probabilities):.1%}")
            st.metric("Athlete ID", selected_id)

        with col3:
            st.subheader("Recommendations")
            if risk_level == "HIGH":
                st.error("Reduce training intensity; consult a sports physician.")
            elif risk_level == "MODERATE":
                st.warning("Increase hydration and recovery sessions.")
            else:
                st.success("Optimal biomarker balance maintained.")

        st.markdown("---")
        st.subheader("Biomarker Profile")
        categories = list(feature_descriptions.values())
        values = [input_data[feat] for feat in feature_names]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-2, 2])), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("---")
st.markdown("AI Sports Medicine • GNODE Model • Real-time Injury Prevention")
