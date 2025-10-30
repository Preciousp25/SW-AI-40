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
from datetime import datetime, timedelta
import json

# Set page config
st.set_page_config(
    page_title="AI Sports Medicine Monitor",
    page_icon="🏥",
    layout="wide"
)

# Simple neural network model
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

# Load model
@st.cache_resource
def load_model():
    try:
        pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if pth_files:
            model_file = pth_files[0]
            checkpoint = torch.load(model_file, map_location='cpu')
            input_size = 7
            model = SimpleBioModel(input_size, 64, 2)
            try:
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            except:
                pass
            model.eval()
            return model
        else:
            return None
    except:
        return None

# Initialize session state for player management
if 'players' not in st.session_state:
    st.session_state.players = {}
if 'current_player' not in st.session_state:
    st.session_state.current_player = None
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}
if 'biosensor_running' not in st.session_state:
    st.session_state.biosensor_running = False

# Feature configurations
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
feature_ranges = {name: (-2.0, 2.0) for name in feature_names}

model = load_model()

# Prediction function
def predict_risk(features_array, model):
    try:
        if model:
            features_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(features_tensor)
                probs = F.softmax(logits, dim=1)
                risk_score = probs[0][1].item()
                return risk_score, probs[0].tolist()
        else:
            # Smart demo based on biomarker patterns
            fatigue = max(0, features_array[3])  # rms_feat
            stress = abs(features_array[1])      # tissue_sweat
            risk_score = min(0.9, 0.3 + fatigue * 0.2 + stress * 0.15)
            return risk_score, [1-risk_score, risk_score]
    except:
        return 0.3, [0.7, 0.3]

# Generate simulated live biosensor data
def generate_live_biosensor_data():
    base_values = {
        'mw': np.random.normal(0, 0.5),
        'tissue_sweat': np.random.normal(-0.2, 0.3),
        'tissue_urine': np.random.normal(0.1, 0.4),
        'rms_feat': np.random.normal(0.5, 0.6),
        'zero_crossings': np.random.normal(0.3, 0.5),
        'skewness': np.random.normal(0, 0.4),
        'waveform_length': np.random.normal(0.2, 0.5)
    }
    return base_values

# Main UI
st.title("🏥 AI Sports Medicine Monitor")
st.markdown("### Advanced Cortisol & Electrolyte Based Injury Risk Prediction")

# Sidebar - Player Management
st.sidebar.header("👥 Player Management")

# Add new player section
with st.sidebar.expander("➕ Add New Player", expanded=True):
    new_player_id = st.text_input("Player ID", "ATH-001")
    new_player_name = st.text_input("Player Name", "John Doe")
    new_player_age = st.number_input("Age", 18, 40, 25)
    new_player_position = st.selectbox("Position", ["Forward", "Midfielder", "Defender", "Goalkeeper"])
    
    if st.button("Add Player", key="add_player"):
        player_data = {
            'name': new_player_name,
            'age': new_player_age,
            'position': new_player_position,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'assessment_history': []
        }
        st.session_state.players[new_player_id] = player_data
        st.session_state.current_player = new_player_id
        st.success(f"✅ Added {new_player_name} ({new_player_id})")

# Player selection
if st.session_state.players:
    player_options = [f"{pid} - {data['name']}" for pid, data in st.session_state.players.items()]
    selected_player = st.sidebar.selectbox("Select Player", player_options)
    st.session_state.current_player = selected_player.split(" - ")[0]

# Main tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Risk Assessment", "👥 Player Management", "📊 Live Biosensors", "📈 History & Analytics"])

with tab1:
    st.header("Real-time Risk Assessment")
    
    if not st.session_state.players:
        st.warning("⚠️ Please add a player first in the sidebar")
    else:
        current_player_id = st.session_state.current_player
        player_data = st.session_state.players[current_player_id]
        
        st.info(f"**Assessing:** {player_data['name']} ({current_player_id}) | {player_data['position']} | Age: {player_data['age']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔬 Manual Biomarker Input")
            input_data = {}
            cols = st.columns(2)
            for i, feature in enumerate(feature_names):
                with cols[i % 2]:
                    input_data[feature] = st.slider(
                        feature_descriptions[feature],
                        min_value=-2.0,
                        max_value=2.0,
                        value=0.0,
                        step=0.1,
                        key=f"manual_{feature}"
                    )
        
        with col2:
            st.subheader("⚡ Quick Actions")
            if st.button("🎯 Assess Current Biomarkers", type="primary", use_container_width=True):
                features_array = np.array([input_data[f] for f in feature_names])
                risk_score, probabilities = predict_risk(features_array, model)
                
                # Save assessment to player history
                assessment = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'risk_score': risk_score,
                    'biomarkers': input_data.copy(),
                    'recommendations': get_recommendations(risk_score)
                }
                st.session_state.players[current_player_id]['assessment_history'].append(assessment)
                
                display_results(risk_score, probabilities, input_data)
            
            if st.button("🔄 Use Live Biosensor Data", use_container_width=True):
                if st.session_state.live_data:
                    features_array = np.array([st.session_state.live_data[f] for f in feature_names])
                    risk_score, probabilities = predict_risk(features_array, model)
                    display_results(risk_score, probabilities, st.session_state.live_data)
                else:
                    st.warning("No live biosensor data available. Start live monitoring in the Biosensors tab.")

with tab2:
    st.header("Player Database")
    
    if not st.session_state.players:
        st.info("No players added yet. Use the sidebar to add players.")
    else:
        # Display players table
        player_list = []
        for pid, data in st.session_state.players.items():
            player_list.append({
                'Player ID': pid,
                'Name': data['name'],
                'Age': data['age'],
                'Position': data['position'],
                'Assessments': len(data['assessment_history']),
                'Last Assessment': data['assessment_history'][-1]['timestamp'] if data['assessment_history'] else 'Never'
            })
        
        players_df = pd.DataFrame(player_list)
        st.dataframe(players_df, use_container_width=True)
        
        # Player details
        if st.session_state.current_player:
            st.subheader(f"Player Details: {st.session_state.current_player}")
            player_data = st.session_state.players[st.session_state.current_player]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Name", player_data['name'])
                st.metric("Position", player_data['position'])
            with col2:
                st.metric("Age", player_data['age'])
                st.metric("Total Assessments", len(player_data['assessment_history']))
            
            # Assessment history
            if player_data['assessment_history']:
                st.subheader("Assessment History")
                history_df = pd.DataFrame(player_data['assessment_history'])
                st.dataframe(history_df, use_container_width=True)

with tab3:
    st.header("📡 Live Biosensor Monitoring")
    
    st.info("This simulates real-time biosensor data from wearable devices")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🎬 Start Live Monitoring", type="primary", disabled=st.session_state.biosensor_running):
            st.session_state.biosensor_running = True
            st.rerun()
        
        if st.button("⏹️ Stop Monitoring", disabled=not st.session_state.biosensor_running):
            st.session_state.biosensor_running = False
            st.rerun()
    
    with col1:
        if st.session_state.biosensor_running:
            st.success("🟢 Live monitoring ACTIVE - Data updating every 5 seconds")
            
            # Create placeholder for live data
            live_placeholder = st.empty()
            
            # Simulate live data updates
            for i in range(10):  # Show 10 updates
                if not st.session_state.biosensor_running:
                    break
                    
                live_data = generate_live_biosensor_data()
                st.session_state.live_data = live_data
                
                # Display current live values
                with live_placeholder.container():
                    st.subheader(" Current Biosensor Readings")
                    cols = st.columns(4)
                    for j, (feature, value) in enumerate(live_data.items()):
                        with cols[j % 4]:
                            st.metric(
                                feature_descriptions[feature],
                                f"{value:.3f}",
                                delta=None
                            )
                    
                    # Make prediction with live data
                    features_array = np.array([live_data[f] for f in feature_names])
                    risk_score, probabilities = predict_risk(features_array, model)
                    
                    st.metric(" Live Risk Score", f"{risk_score:.1%}")
                    
                    if risk_score > 0.7:
                        st.error("🚨 High Risk - Consider immediate action")
                    elif risk_score > 0.4:
                        st.warning("⚠️ Moderate Risk - Monitor closely")
                    else:
                        st.success("✅ Low Risk - Normal range")
                
                time.sleep(5)  # Update every 5 seconds
                st.rerun()
        else:
            st.warning("🔴 Live monitoring INACTIVE - Click 'Start Live Monitoring' to begin")

with tab4:
    st.header("Analytics & History")
    
    if not st.session_state.players:
        st.info("No data available. Add players and conduct assessments first.")
    else:
        # Risk distribution across players
        risk_data = []
        for pid, data in st.session_state.players.items():
            if data['assessment_history']:
                latest_assessment = data['assessment_history'][-1]
                risk_data.append({
                    'Player': f"{data['name']} ({pid})",
                    'Risk Score': latest_assessment['risk_score'],
                    'Risk Level': 'HIGH' if latest_assessment['risk_score'] > 0.7 else 'MODERATE' if latest_assessment['risk_score'] > 0.4 else 'LOW'
                })
        
        if risk_data:
            risk_df = pd.DataFrame(risk_data)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(risk_df, x='Player', y='Risk Score', 
                           color='Risk Level', title='Player Risk Scores',
                           color_discrete_map={'HIGH': 'red', 'MODERATE': 'orange', 'LOW': 'green'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig_pie = px.pie(risk_df, names='Risk Level', title='Risk Level Distribution',
                               color='Risk Level',
                               color_discrete_map={'HIGH': 'red', 'MODERATE': 'orange', 'LOW': 'green'})
                st.plotly_chart(fig_pie, use_container_width=True)

# Helper functions
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
    
    st.success("###  Assessment Complete!")
    
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
        st.metric("Confidence", f"{max(probabilities):.1%}")
        
        st.info("** Risk Indicators:**")
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

# Footer
st.markdown("---")
st.markdown("*AI Sports Medicine Platform • Real-time Biosensor Monitoring • Professional Athlete Management*")
