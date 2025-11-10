# -----------------------------
# Imports
# -----------------------------
import streamlit as st
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
# from twilio.rest import Client  # SMS functionality commented out

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="AI Sports Medicine Monitor",
    page_icon="AILOGO.png",
    layout="wide"
)

# -----------------------------
# App Header
# -----------------------------
col1, col2 = st.columns([1, 4])
with col1:
    st.image("AILOGO.png", width=80)
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: #1f77b4;'>InjuryGuard AI</h1>
        <p style='font-size:18px;'>AI-powered Real-time Athlete Injury Monitoring Platform</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# -----------------------------
# Twilio SMS function - COMMENTED OUT
# -----------------------------
# def send_sms(to_number, message):
#     account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
#     auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
#     from_number = st.secrets["TWILIO_PHONE_NUMBER"]
# 
#     client = Client(account_sid, auth_token)
#     client.messages.create(
#         body=message,
#         from_=from_number,
#         to=to_number
#     )

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
                             {'range': [70,100], 'color': "red"}]}
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Risk Level", risk_level)
        st.metric("Injury Risk Probability", f"{risk_score:.1%}")
        st.info("**Risk Indicators:**")
        if biomarkers['rms_feat'] > 1.0: st.warning("• Elevated muscle fatigue")
        if biomarkers['tissue_sweat'] < -1.0: st.warning("• Abnormal sweat conductivity")
        if risk_level == "LOW": st.success("• Biomarkers within normal range")

    with col3:
        st.subheader(" Recommendations")
        recommendations = get_recommendations(risk_score)
        for rec in recommendations:
            if risk_level == "HIGH":
                st.error(f"• {rec}")
            elif risk_level == "MODERATE":
                st.warning(f"• {rec}")
            else:
                st.success(f"• {rec}")

# -----------------------------
# GNODE Model Definition
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

# -----------------------------
# Load GNODE model
# -----------------------------
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
# Session State Initialization
# -----------------------------
if 'players' not in st.session_state: st.session_state.players = {}
if 'current_player' not in st.session_state: st.session_state.current_player = None
if 'live_data' not in st.session_state: st.session_state.live_data = {}
if 'biosensor_running' not in st.session_state: st.session_state.biosensor_running = False

# -----------------------------
# Feature Info
# -----------------------------
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

# -----------------------------
# Predict Risk
# -----------------------------
def predict_risk(features_array, model, k=3):
    try:
        features_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)
        num_nodes = features_tensor.shape[0]
        if num_nodes == 1:
            edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        else:
            edge_index = knn_graph(features_tensor, k=k, loop=True)
        gnode = GNODEModel(features_tensor.shape[1], 32, 2, edge_index)
        if model is not None:
            gnode.load_state_dict(model.state_dict())
            gnode.eval()
            with torch.no_grad():
                logits = gnode(features_tensor)
                probs = F.softmax(logits, dim=1)
                risk_score = probs[0][1].item()
                return risk_score, probs[0].tolist()
        else:
            fatigue = max(0, features_array[3])
            stress = abs(features_array[1])
            risk_score = min(0.9, 0.3 + fatigue*0.2 + stress*0.15)
            return risk_score, [1-risk_score, risk_score]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.3, [0.7,0.3]

# -----------------------------
# Generate Live Biosensor Data
# -----------------------------
def generate_live_biosensor_data():
    return {
        'mw': np.random.normal(0,0.5),
        'tissue_sweat': np.random.normal(-0.2,0.3),
        'tissue_urine': np.random.normal(0.1,0.4),
        'rms_feat': np.random.normal(0.5,0.6),
        'zero_crossings': np.random.normal(0.3,0.5),
        'skewness': np.random.normal(0,0.4),
        'waveform_length': np.random.normal(0.2,0.5)
    }

# -----------------------------
# Sidebar - Player Management (ONLY ONE)
# -----------------------------
st.sidebar.header("👥 Player Management")

with st.sidebar.expander("➕ Add New Player", expanded=True):
    new_player_id = st.text_input("Player ID", "ATH-001", key="new_player_id")
    new_player_name = st.text_input("Player Name", "John Peterson", key="new_player_name")
    new_player_age = st.number_input("Age", 18, 40, 25, key="new_player_age")
    new_player_position = st.selectbox(
        "Position", ["Forward", "Midfielder", "Defender", "Goalkeeper"], key="new_player_position"
    )
    # Coach phone number input commented out
    # new_coach_number = st.text_input(
    #     "📱 Coach/Recipient Phone Number (with country code, e.g., +2567XXXXXXX)",
    #     value="+2567",
    #     key="new_coach_number"
    # )

    if st.button("Add Player", key="add_player_btn"):
        # SMS validation removed
        # if new_coach_number.strip() == "" or not new_coach_number.startswith("+"):
        #     st.warning("Please enter a valid coach phone number (include country code).")
        if new_player_id.strip() == "":
            st.warning("Please enter a valid Player ID.")
        else:
            player_data = {
                'name': new_player_name,
                'age': new_player_age,
                'position': new_player_position,
                # 'coach_number': new_coach_number,  # Removed from player data
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'assessment_history': []
            }
            st.session_state.players[new_player_id] = player_data
            st.session_state.current_player = new_player_id
            # SMS reference removed from success message
            st.success(f"Added {new_player_name} ({new_player_id})")

# Player Selection
if st.session_state.players:
    player_options = [f"{pid} - {data['name']}" for pid, data in st.session_state.players.items()]
    selected_player = st.sidebar.selectbox("Select Player", player_options)
    st.session_state.current_player = selected_player.split(" - ")[0]

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Risk Assessment", "Player Management", "Live Biosensors", "History & Analytics"])

# -----------------------------
# Tab1 - Risk Assessment
# -----------------------------
with tab1:
    st.header("Real-time Risk Assessment")
    
    if not st.session_state.players:
        st.warning("⚠️ Please add a player first in the sidebar")
    else:
        current_player_id = st.session_state.current_player
        player_data = st.session_state.players[current_player_id]
        st.info(f"**Assessing:** {player_data['name']} ({current_player_id}) | {player_data['position']} | Age: {player_data['age']}")

        col1, col2 = st.columns([2, 1])
        # Manual input
        with col1:
            st.subheader("🔬 Manual Biomarker Input")
            input_data = {}
            cols = st.columns(2)
            for i, feature in enumerate(feature_names):
                with cols[i % 2]:
                    input_data[feature] = st.slider(
                        feature_descriptions[feature],
                        min_value=-2.0, max_value=2.0,
                        value=0.0, step=0.01,
                        format="%.2f",
                        key=f"manual_{feature}"
                    )
        # Quick Actions
        with col2:
            st.subheader("⚡ Quick Actions")
            if st.button("Assess Current Biomarkers", type="primary", use_container_width=True):
                features_array = np.array([input_data[f] for f in feature_names])
                risk_score, probabilities = predict_risk(features_array, model)
                assessment = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'risk_score': risk_score,
                    'biomarkers': input_data.copy(),
                    'recommendations': get_recommendations(risk_score)
                }
                st.session_state.players[current_player_id]['assessment_history'].append(assessment)

                # SMS alert functionality commented out
                # coach_number = player_data.get('coach_number', '')
                # if risk_score > 0.7 and coach_number:
                #     alert_msg = f"🚨 {player_data['name']} has HIGH injury risk ({risk_score:.1%})! Take immediate action."
                #     send_sms(coach_number, alert_msg)
                #     st.success(f"SMS alert sent to {coach_number}")

                display_results(risk_score, probabilities, input_data)

# -----------------------------
# Tab2 - Player Management (View/Edit only)
# -----------------------------
with tab2:
    st.header("Player Management")
    if not st.session_state.players:
        st.info("No players added yet.")
    else:
        for pid, pdata in st.session_state.players.items():
            st.subheader(f"{pdata['name']} ({pid})")
            st.write(f"Position: {pdata['position']} | Age: {pdata['age']}")
            # Coach phone display commented out
            # st.write(f"Coach Phone: {pdata.get('coach_number','N/A')}")
            st.write(f"Assessments: {len(pdata['assessment_history'])}")

# -----------------------------
# Tab3 - Live Biosensors
# -----------------------------
with tab3:
    st.header("📡 Live Biosensor Monitoring")
    st.info("Simulates real-time biosensor data")
    col1, col2 = st.columns([3,1])
    with col2:
        if st.button("🎬 Start Live Monitoring", disabled=st.session_state.biosensor_running):
            st.session_state.biosensor_running = True
            st.rerun()
        if st.button("⏹️ Stop Monitoring", disabled=not st.session_state.biosensor_running):
            st.session_state.biosensor_running = False
            st.rerun()
    with col1:
        if st.session_state.biosensor_running:
            st.success("🟢 Live monitoring ACTIVE")
            live_placeholder = st.empty()
            for _ in range(10):
                if not st.session_state.biosensor_running:
                    break
                live_data = generate_live_biosensor_data()
                st.session_state.live_data = live_data
                with live_placeholder.container():
                    st.subheader("Current Biosensor Readings")
                    cols = st.columns(4)
                    for j,(feature,value) in enumerate(live_data.items()):
                        with cols[j%4]:
                            st.metric(feature_descriptions[feature], f"{value:.3f}")
                    features_array = np.array([live_data[f] for f in feature_names])
                    risk_score, probabilities = predict_risk(features_array, model)
                    st.metric("🎯 Live Risk Score", f"{risk_score:.1%}")
                    if risk_score > 0.7: st.error("🚨 High Risk")
                    elif risk_score > 0.4: st.warning("⚠️ Moderate Risk")
                    else: st.success(" Low Risk")
                time.sleep(5)
                st.rerun()
        else:
            st.warning("🔴 Live monitoring INACTIVE")

# -----------------------------
# Tab4 - Analytics & History
# -----------------------------
with tab4:
    st.header("Analytics & History")
    if not st.session_state.players:
        st.info("No data available.")
    else:
        risk_data = []
        for pid, data in st.session_state.players.items():
            if data['assessment_history']:
                latest = data['assessment_history'][-1]
                risk_data.append({
                    'Player': f"{data['name']} ({pid})",
                    'Risk Score': latest['risk_score'],
                    'Risk Level': 'HIGH' if latest['risk_score']>0.7 else 'MODERATE' if latest['risk_score']>0.4 else 'LOW'
                })
        if risk_data:
            risk_df = pd.DataFrame(risk_data)
            col1,col2 = st.columns(2)
            with col1:
                fig = px.bar(risk_df, x='Player', y='Risk Score', color='Risk Level',
                             color_discrete_map={'HIGH':'red','MODERATE':'orange','LOW':'green'})
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig_pie = px.pie(risk_df, names='Risk Level', values='Risk Score',
                                 color='Risk Level',
                                 color_discrete_map={'HIGH':'red','MODERATE':'orange','LOW':'green'})
                st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("*AI Sports Medicine Platform • Real-time Biosensor Monitoring • Professional Athlete Management*")