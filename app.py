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
    # -----------------------------
    # Page config
    # -----------------------------
    st.set_page_config(
        page_title="InjuryGuard AI",
        page_icon="AILOGO.png",
        layout="wide"
    )

    # -----------------------------
    # Header
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
        h1, h2, h3, .stSubheader { color: #1f77b4 !important; }
        div.stButton > button:first-child { background-color: #1f77b4; color: white; border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)

    # -----------------------------
    # Helper Functions
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
                domain={'x':[0,1],'y':[0,1]},
                gauge={'axis': {'range':[None,100]}, 'bar': {'color': risk_color},
                    'steps':[{'range':[0,30],'color':'lightgreen'},
                             {'range':[30,70],'color':'yellow'},
                             {'range':[70,100],'color':'red'}]}
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
            st.subheader("Recommendations")
            recommendations = get_recommendations(risk_score)
            for rec in recommendations:
                if risk_level == "HIGH": st.error(f"• {rec}")
                elif risk_level == "MODERATE": st.warning(f"• {rec}")
                else: st.success(f"• {rec}")

    # -----------------------------
    # GNODE Model
    # -----------------------------
    class ODEFunc(nn.Module):
        def __init__(self, in_channels, hidden_channels, edge_index):
            super().__init__()
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
            super().__init__()
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
            model = GNODEModel(7, 32, 2, edge_index)
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
    # Predict Risk
    # -----------------------------
    def predict_risk(features_array, model, k=3):
        try:
            features_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)
            num_nodes = features_tensor.shape[0]
            if num_nodes == 1: edge_index = torch.tensor([[0],[0]], dtype=torch.long)
            else: edge_index = knn_graph(features_tensor, k=k, loop=True)
            gnode = GNODEModel(features_tensor.shape[1], 32, 2, edge_index)
            if model is not None: gnode.load_state_dict(model.state_dict())
            gnode.eval()
            with torch.no_grad():
                logits = gnode(features_tensor)
                probs = F.softmax(logits, dim=1)
                risk_score = probs[0][1].item()
                return risk_score, probs[0].tolist()
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return 0.3, [0.7,0.3]

    # -----------------------------
    # Live Biosensor Data
    # -----------------------------
    def generate_live_biosensor_data():
        return {f: np.random.normal(0,0.5) for f in feature_names}

    # -----------------------------
    # Session State Init
    # -----------------------------
    if 'players' not in st.session_state: st.session_state.players = {}
    if 'current_player' not in st.session_state: st.session_state.current_player = None
    if 'live_data' not in st.session_state: st.session_state.live_data = {}
    if 'biosensor_running' not in st.session_state: st.session_state.biosensor_running = False

    # -----------------------------
    # Sidebar - Player Management
    # -----------------------------
    
    st.sidebar.header("👥 Player Management")
    with st.sidebar.expander("➕ Add New Player", expanded=True):
        new_player_id = st.text_input("Player ID", "ATH-001", key="new_player_id")
        new_player_name = st.text_input("Player Name", "John Peterson", key="new_player_name")
        new_player_age = st.number_input("Age", 18, 40, 25, key="new_player_age")
        new_player_position = st.selectbox("Position", ["Forward","Midfielder","Defender","Goalkeeper"], key="new_player_position")
        if st.button("Add Player", key="add_player_btn"):
            if new_player_id.strip() == "": st.warning("Please enter a valid Player ID.")
            else:
                player_data = {
                    'name': new_player_name,
                    'age': new_player_age,
                    'position': new_player_position,
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'assessment_history': []
                }
                st.session_state.players[new_player_id] = player_data
                st.session_state.current_player = new_player_id
                st.success(f"Added {new_player_name} ({new_player_id})")

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
        if not st.session_state.players: st.warning("⚠️ Please add a player first in the sidebar")
        else:
            current_player_id = st.session_state.current_player
            player_data = st.session_state.players[current_player_id]
            st.info(f"**Assessing:** {player_data['name']} ({current_player_id}) | {player_data['position']} | Age: {player_data['age']}")
            col1, col2 = st.columns([2,1])
            # Manual Input
            input_data = {}
            cols = st.columns(2)
            for i, feature in enumerate(feature_names):
                with cols[i%2]:
                    input_data[feature] = st.slider(feature_descriptions[feature], -2.0,2.0,0.0,0.01, format="%.2f", key=f"manual_{feature}")
            # Quick Actions
            with col2:
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
                    display_results(risk_score, probabilities, input_data)

    # -----------------------------
    # Tab2 - Player Management
    # -----------------------------
    with tab2:
        st.header("Player Management")
        if not st.session_state.players: st.info("No players added yet.")
        else:
            for pid, pdata in st.session_state.players.items():
                st.subheader(f"{pdata['name']} ({pid})")
                st.write(f"Position: {pdata['position']} | Age: {pdata['age']}")
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
                    if not st.session_state.biosensor_running: break
                    live_data = generate_live_biosensor_data()
                    st.session_state.live_data = live_data
                    with live_placeholder.container():
                        st.subheader("Current Biosensor Readings")
                        cols = st.columns(4)
                        for j,(feature,value) in enumerate(live_data.items()):
                            with cols[j%4]: st.metric(feature_descriptions[feature], f"{value:.3f}")
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
        if not st.session_state.players: st.info("No data available.")
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

