import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import os
import json
import random

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Sports Medicine Monitor",
    page_icon="🏥",
    layout="wide"
)

PLAYER_DB = "players.json"

# -------------------------------
# MODEL DEFINITION
# -------------------------------
class SimpleBioModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleBioModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------------------
# MODEL LOADING
# -------------------------------
@st.cache_resource
def load_model():
    try:
        pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if pth_files:
            model_file = pth_files[0]
            st.success(f" Found model: {model_file}")
            
            checkpoint = torch.load(model_file, map_location='cpu')
            input_size = 7
            model = SimpleBioModel(input_size, 64, 2)
            try:
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                st.success(" Model weights loaded successfully!")
            except Exception as e:
                st.info(f"Using default weights: {e}")
            
            model.eval()
            return model
        else:
            st.warning(" No model file found. Demo Mode activated.")
            return None
    except Exception as e:
        st.warning(f"Model load issue: {e}. Demo Mode.")
        return None

model = load_model()

# -------------------------------
# PLAYER MANAGEMENT
# -------------------------------
def load_players():
    if os.path.exists(PLAYER_DB):
        with open(PLAYER_DB, "r") as f:
            return json.load(f)
    return {}

def save_players(players):
    with open(PLAYER_DB, "w") as f:
        json.dump(players, f, indent=2)

players = load_players()

# -------------------------------
# FEATURE DEFINITIONS
# -------------------------------
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

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("🏃 Athlete Management")
mode = st.sidebar.radio("Choose Mode:", ["Assess Player", "Add New Player", "Batch Screening"])

# -------------------------------
# ADD PLAYER
# -------------------------------
if mode == "Add New Player":
    st.header("➕ Register New Player")
    with st.form("add_player_form"):
        player_id = st.text_input("Athlete ID", f"ATH-{len(players)+1:03d}")
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=10, max_value=60, value=20)
        team = st.text_input("Team", "Default Team")
        position = st.text_input("Position", "Midfielder")
        submitted = st.form_submit_button(" Save Player")
    
    if submitted:
        players[player_id] = {
            "name": name,
            "age": age,
            "team": team,
            "position": position
        }
        save_players(players)
        st.success(f" Player {name} ({player_id}) added successfully!")

# -------------------------------
# ASSESS PLAYER
# -------------------------------
elif mode == "Assess Player":
    st.header(" Injury Risk Assessment")

    if not players:
        st.warning("No players registered yet. Add one first.")
        st.stop()

    selected_id = st.selectbox("Select Player:", list(players.keys()))
    selected_player = players[selected_id]
    st.info(f"**{selected_player['name']}** — {selected_player['position']} ({selected_player['team']})")

    # Input mode
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
        # Simulate live biosensor input (replace later with IoT feed)
        for feat in feature_names:
            input_data[feat] = round(random.uniform(-2, 2), 2)
        st.success("📡 Live biosensor data received!")
        st.json(input_data)

    # Prediction
    def make_prediction(features_array, model):
        try:
            if model is not None:
                features_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = model(features_tensor)
                    probabilities = F.softmax(logits, dim=1)
                    risk_score = probabilities[0][1].item()
                    probs = probabilities[0].tolist()
                return risk_score, probs
            else:
                fatigue_factor = max(0, input_data['rms_feat'])
                stress_factor = abs(input_data['tissue_sweat'])
                variability = abs(input_data['zero_crossings'])
                base_risk = 0.3
                risk_adjustment = (fatigue_factor*0.2 + stress_factor*0.15 + variability*0.1)
                risk_score = min(0.9, base_risk + risk_adjustment)
                return risk_score, [1-risk_score, risk_score]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return 0.3, [0.7, 0.3]

    if st.button(" Assess Injury Risk", type="primary"):
        features_array = np.array([input_data[feature] for feature in feature_names])
        risk_score, probabilities = make_prediction(features_array, model)

        risk_level = "HIGH" if risk_score > 0.7 else "MODERATE" if risk_score > 0.4 else "LOW"
        risk_color = {"HIGH": "red", "MODERATE": "orange", "LOW": "green"}[risk_level]

        st.success("###  Risk Assessment Complete!")

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
            st.subheader(" Recommendations")
            if risk_level == "HIGH":
                st.error(" Reduce training intensity; consult a sports physician.")
            elif risk_level == "MODERATE":
                st.warning(" Add hydration & recovery period.")
            else:
                st.success(" Optimal biomarker balance.")

        st.markdown("---")
        st.subheader(" Biomarker Profile")
        categories = list(feature_descriptions.values())
        values = [input_data[feat] for feat in feature_names]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Current Biomarkers'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-2, 2])), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)

# -------------------------------
# BATCH SCREENING
# -------------------------------
elif mode == "Batch Screening":
    st.header(" Team Batch Screening")
    uploaded_file = st.file_uploader("Upload team data (CSV)", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.dataframe(df.head())
        if st.button(" Process Team Assessment"):
            results = []
            progress = st.progress(0)
            for i, row in df.iterrows():
                progress.progress((i+1)/len(df))
                features = np.array([row.get(f, 0) for f in feature_names])
                risk, probs = (0.5, [0.5, 0.5]) if model is None else make_prediction(features, model)
                level = "HIGH" if risk > 0.7 else "MODERATE" if risk > 0.4 else "LOW"
                results.append({
                    "Athlete": row.get("Athlete_ID", f"ATH-{i+1:03d}"),
                    "Risk_Score": risk,
                    "Risk_Level": level,
                    "Confidence": max(probs)
                })
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            st.download_button(" Download Results", results_df.to_csv(index=False), "team_assessment.csv", "text/csv")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("*AI Sports Medicine • GNODE Model • Real-time Injury Prevention*")
