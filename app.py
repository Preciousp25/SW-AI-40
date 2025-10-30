import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torchdiffeq import odeint
import plotly.graph_objects as go
import plotly.express as px
import os

# Set page config - MUST be first Streamlit command
st.set_page_config(
    page_title="AI Sports Medicine Monitor",
    page_icon="🏥",
    layout="wide"
)

# Define your GNODE model architecture (same as your training)
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
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.odefunc = ODEFunc(hidden_channels, hidden_channels, edge_index)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.input_proj(x)
        x = F.relu(x)
        t = torch.tensor([0, 1], dtype=torch.float)
        out = odeint(self.odefunc, x, t, method='rk4', options={'step_size':0.1})
        out = out[-1]
        out = self.linear(out)
        return out

# Load your trained model
@st.cache_resource
def load_model():
    try:
        # Find the .pth file
        pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if pth_files:
            model_file = pth_files[0]
            st.success(f" Found model file: {model_file}")
            
            # Load the checkpoint
            checkpoint = torch.load(model_file, map_location='cpu')
            
            # Create model with correct architecture
            model = GNODEModel(
                in_channels=checkpoint['in_channels'],
                hidden_channels=checkpoint['hidden_channels'],
                out_channels=checkpoint['out_channels'],
                edge_index=checkpoint['edge_index']
            )
            
            # Load trained weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            st.success(" GNODE Model loaded successfully!")
            return model, checkpoint
        else:
            st.warning(" No .pth file found. Using demo mode.")
            return None, None
    except Exception as e:
        st.error(f" Error loading model: {e}")
        return None, None

# Initialize
model, checkpoint = load_model()

# Feature descriptions
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

# Prediction function
def make_prediction(features_array, model, checkpoint):
    try:
        if model is None:
            # Demo mode
            risk_score = np.clip(np.random.normal(0.3, 0.2), 0.1, 0.9)
            return risk_score, [1-risk_score, risk_score]
        
        # Convert to tensor
        features_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(features_tensor)
            probabilities = F.softmax(logits, dim=1)
            risk_score = probabilities[0][1].item()  # Probability of class 1
            probs = probabilities[0].tolist()
        
        return risk_score, probs
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        risk_score = np.clip(np.random.normal(0.3, 0.2), 0.1, 0.9)
        return risk_score, [1-risk_score, risk_score]

# UI Components
st.title(" AI-Powered Sports Medicine Monitor")
st.markdown("### Cortisol & Electrolyte Based Injury Risk Prediction")

# Sidebar for input
st.sidebar.header(" Biomarker Input Parameters")

# Create input fields for each feature
input_data = {}
cols = st.sidebar.columns(2)

for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        input_data[feature] = st.slider(
            feature_descriptions[feature],
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            key=feature
        )

# Contextual factors
st.sidebar.header("🏃 Athlete Context")
athlete_id = st.sidebar.text_input("Athlete ID", "ATH-001")
session_type = st.sidebar.selectbox("Session Type", ["Light", "Moderate", "Intense", "Recovery"])

# Prediction button
if st.sidebar.button(" Assess Injury Risk", type="primary"):
    
    # Create feature array
    features_array = np.array([input_data[feature] for feature in feature_names])
    
    # Make prediction
    risk_score, probabilities = make_prediction(features_array, model, checkpoint)
    
    # Determine risk level
    if risk_score > 0.7:
        risk_level = "HIGH"
        risk_color = "red"
    elif risk_score > 0.4:
        risk_level = "MODERATE" 
        risk_color = "orange"
    else:
        risk_level = "LOW"
        risk_color = "green"
    
    # Display results
    st.success("###  Risk Assessment Complete!")
    
    # Results columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Injury Risk Score"},
            gauge = {
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
        st.metric("Athlete ID", athlete_id)
        
        # Risk factors
        st.info("**🔍 Risk Indicators:**")
        if input_data['rms_feat'] > 1.0:
            st.warning("• Elevated muscle fatigue")
        if input_data['tissue_sweat'] < -1.0:
            st.warning("• Abnormal sweat conductivity")
        if risk_level == "LOW":
            st.success("• Biomarkers within normal range")
    
    with col3:
        st.subheader(" Recommendations")
        
        if risk_level == "HIGH":
            st.error("""
            **Immediate Action:**
            • Reduce training intensity
            • Electrolyte supplementation
            • Consult sports physician
            • Close monitoring needed
            """)
        elif risk_level == "MODERATE":
            st.warning("""
             **Precautions:**
            • Modify training load
            • Ensure proper hydration
            • Extra recovery time
            • Re-assess in 48h
            """)
        else:
            st.success("""
             **Optimal State:**
            • Maintain current regimen
            • Standard monitoring
            • Continue good practices
            """)

    # Biomarker visualization
    st.markdown("---")
    st.subheader(" Biomarker Profile")
    
    # Radar chart
    categories = list(feature_descriptions.values())
    values = [input_data[feat] for feat in feature_names]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Biomarkers'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-2, 2])),
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# Batch processing
st.markdown("---")
st.header(" Team Batch Screening")

uploaded_file = st.file_uploader("Upload team data (CSV)", type=['csv'])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.dataframe(df.head())
        
        if st.button(" Process Team Assessment"):
            results = []
            for i, row in df.iterrows():
                features_array = np.array([row.get(feat, 0) for feat in feature_names])
                risk_score, probabilities = make_prediction(features_array, model, checkpoint)
                
                risk_level = "HIGH" if risk_score > 0.7 else "MODERATE" if risk_score > 0.4 else "LOW"
                
                results.append({
                    'Athlete_ID': f'ATH-{i+1:03d}',
                    'Risk_Score': risk_score,
                    'Risk_Level': risk_level,
                    'Confidence': max(probabilities)
                })
            
            results_df = pd.DataFrame(results)
            st.success(f" Processed {len(results_df)} athletes!")
            
            # Display and download
            st.dataframe(results_df)
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label=" Download Results",
                data=csv,
                file_name="team_assessment.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*AI Sports Medicine • GNODE Model • Real-time Injury Prevention*")

# Model status in sidebar
with st.sidebar:
    st.markdown("---")
    if model is not None:
        st.success(" GNODE Model: ACTIVE")
        if checkpoint:
            st.write(f"Input dim: {checkpoint['in_channels']}")
            st.write(f"Hidden dim: {checkpoint['hidden_channels']}")
    else:
        st.warning(" GNODE Model: DEMO MODE")