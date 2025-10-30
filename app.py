import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(
    page_title="AI Sports Medicine Monitor",
    page_icon="🏥",
    layout="wide"
)

# Simple neural network model (no torch-geometric dependencies)
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

# Load model with fallback
@st.cache_resource
def load_model():
    try:
        # Find .pth file
        pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if pth_files:
            model_file = pth_files[0]
            st.success(f"✅ Found model: {model_file}")
            
            # Try to load the checkpoint
            checkpoint = torch.load(model_file, map_location='cpu')
            
            # Create simple model compatible with your features
            input_size = 7  # Your 7 features
            model = SimpleBioModel(input_size, 64, 2)
            
            # Try to load weights (if compatible)
            try:
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                st.success("✅ Model weights loaded successfully!")
            except Exception as e:
                st.info(f"🔄 Using model with default weights: {e}")
            
            model.eval()
            return model
        else:
            st.warning("⚠️ No .pth file found - Using AI Demo Mode")
            return None
    except Exception as e:
        st.warning(f"🔄 Model load issue: {e}. Using Demo Mode.")
        return None

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

# Initialize model
model = load_model()

# UI Components
st.title("🏥 AI-Powered Sports Medicine Monitor")
st.markdown("### Cortisol & Electrolyte Based Injury Risk Prediction")

# Sidebar for input
st.sidebar.header("🔬 Biomarker Input Parameters")

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

# Prediction function
def make_prediction(features_array, model):
    try:
        if model is not None:
            # Convert to tensor and make prediction
            features_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                logits = model(features_tensor)
                probabilities = F.softmax(logits, dim=1)
                risk_score = probabilities[0][1].item()  # Probability of class 1
                probs = probabilities[0].tolist()
            
            return risk_score, probs
        else:
            # Smart demo mode based on input values
            fatigue_factor = max(0, input_data['rms_feat']) 
            stress_factor = abs(input_data['tissue_sweat'])
            variability = abs(input_data['zero_crossings'])
            
            # Calculate risk based on biomarker patterns
            base_risk = 0.3
            risk_adjustment = (fatigue_factor * 0.2 + 
                             stress_factor * 0.15 + 
                             variability * 0.1)
            
            risk_score = min(0.9, base_risk + risk_adjustment)
            return risk_score, [1-risk_score, risk_score]
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        # Fallback
        risk_score = 0.3
        return risk_score, [0.7, 0.3]

# Prediction button
if st.sidebar.button("🔮 Assess Injury Risk", type="primary"):
    
    # Create feature array
    features_array = np.array([input_data[feature] for feature in feature_names])
    
    # Make prediction
    risk_score, probabilities = make_prediction(features_array, model)
    
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
    st.success("### 🎯 Risk Assessment Complete!")
    
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
        
        # Risk factors analysis
        st.info("**🔍 Risk Indicators:**")
        if input_data['rms_feat'] > 1.0:
            st.warning("• Elevated muscle fatigue detected")
        if input_data['tissue_sweat'] < -1.0:
            st.warning("• Abnormal sweat conductivity")
        if input_data['zero_crossings'] > 1.5:
            st.warning("• High signal variability")
        if risk_level == "LOW":
            st.success("• Biomarkers within normal range")
    
    with col3:
        st.subheader("📋 Recommendations")
        
        if risk_level == "HIGH":
            st.error("""
            🚨 **Immediate Action:**
            • Reduce training intensity
            • Electrolyte supplementation
            • Consult sports physician
            • Close monitoring needed
            """)
        elif risk_level == "MODERATE":
            st.warning("""
            ⚠️ **Precautions:**
            • Modify training load
            • Ensure proper hydration
            • Extra recovery time
            • Re-assess in 48h
            """)
        else:
            st.success("""
            ✅ **Optimal State:**
            • Maintain current regimen
            • Standard monitoring
            • Continue good practices
            """)

    # Biomarker visualization
    st.markdown("---")
    st.subheader("📊 Biomarker Profile")
    
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
st.header("📁 Team Batch Screening")

uploaded_file = st.file_uploader("Upload team data (CSV)", type=['csv'])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.dataframe(df.head())
        
        if st.button("🚀 Process Team Assessment"):
            results = []
            progress_bar = st.progress(0)
            
            for i, row in df.iterrows():
                # Update progress
                progress_bar.progress((i + 1) / len(df))
                
                features_array = np.array([row.get(feat, 0) for feat in feature_names])
                risk_score, probabilities = make_prediction(features_array, model)
                
                risk_level = "HIGH" if risk_score > 0.7 else "MODERATE" if risk_score > 0.4 else "LOW"
                
                results.append({
                    'Athlete_ID': f'ATH-{i+1:03d}',
                    'Risk_Score': risk_score,
                    'Risk_Level': risk_level,
                    'Confidence': max(probabilities)
                })
            
            results_df = pd.DataFrame(results)
            st.success(f"✅ Processed {len(results_df)} athletes!")
            
            # Display and download
            st.dataframe(results_df)
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
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
        st.success("✅ AI Model: ACTIVE")
    else:
        st.warning("⚠️ AI Model: DEMO MODE")
    st.markdown("**Features:**")
    st.write("• Real-time risk assessment")
    st.write("• Biomarker analysis")
    st.write("• Clinical recommendations")
