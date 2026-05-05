import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# Load Model & Scaler
model = joblib.load('decision_tree_model.pkl')
scaler = joblib.load('data_scaler.pkl')

st.set_page_config(page_title="ECMO Hemolysis Predictor", layout="wide")

# Header
st.title("🩺 ECMO/Microfluidic Blood Damage Predictor")
st.subheader("Internal Artificial Organ Design Optimization Tool")

# Layout: Two Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Design Parameters")
    flow = st.slider("Flow Rate (ml/min)", 10.0, 150.0, 70.0)
    inlet_dia = st.number_input("Inlet Diameter (mm)", 1.0, 5.0, 1.6)
    max_shear = st.number_input("Max Wall Shear Rate (1/s)", 3000.0, 200000.0, 50000.0)
    throat_shear = st.number_input("Throat Avg Shear Rate (1/s)", 100.0, 20000.0, 3000.0)
    dev_type = st.selectbox("Device Architecture", ["Baseline", "Herringbone", "Reduced Gap", "Reduced Port"])
    
    # Mapping
    dev_map = {"Baseline":0, "Herringbone":1, "Reduced Gap":2, "Reduced Port":3}

with col2:
    st.markdown("### Performance Analysis")
    if st.button("Generate Prediction Report"):
        # Predict Logic
        data = np.array([[flow, inlet_dia, 400, max_shear, 20000, throat_shear, 1e-10, 1e-7, 1e-12, 2.08, dev_map[dev_type]]])
        scaled_data = scaler.transform(data)
        damage = 10**model.predict(scaled_data)[0]
        
        # Risk Visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=damage*1000000,
            title={'text': "Hemolysis Index (PPM)"},
            gauge={'axis': {'range': [0, 10]}, 'bar': {'color': "red"}}
        ))
        st.plotly_chart(fig)
        
        # Decision
        if damage < 0.0005:
            st.success("Design Pass: Hemolysis within clinical limits.")
        else:
            st.warning("Design Warning: High Shear Stress detected. Optimization Required.")

# Image of Hemolysis Impact
st.markdown("---")
st.write("Professional Analysis for Biomedical Engineers.")
