import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page config
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }

    .main {
        background-color: #f0f4f8;
    }

    .stApp {
        background: linear-gradient(135deg, #e8f0fe 0%, #f0f4f8 100%);
    }

    h1 {
        color: #1a237e;
        font-weight: 700;
        font-size: 2rem;
    }

    .subtitle {
        color: #5c6bc0;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .result-box {
        background: linear-gradient(135deg, #1a237e, #3949ab);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px rgba(26,35,126,0.3);
    }

    .result-label {
        font-size: 0.9rem;
        font-weight: 300;
        opacity: 0.85;
        margin-bottom: 0.5rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1a237e, #3949ab);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-family: 'Sora', sans-serif;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #3949ab, #1a237e);
        box-shadow: 0 4px 15px rgba(26,35,126,0.4);
    }

    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and features
@st.cache_resource
def load_model():
    with open('insurance_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('selected_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

model, selected_features = load_model()

# Header
st.markdown("## 🏥 Insurance Premium Predictor")
st.markdown('<p class="subtitle">Fill in your health details to get an estimated insurance premium</p>', unsafe_allow_html=True)

# Input form
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 👤 Personal Details")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
with col2:
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
with col3:
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)

# Auto calculate BMI
bmi = weight / ((height / 100) ** 2)
st.info(f"📊 Your calculated BMI: **{bmi:.1f}**")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🏨 Medical History")

col3, col4 = st.columns(2)
with col3:
    any_transplants = st.selectbox("Any Transplants?", ["No", "Yes"])
    any_chronic = st.selectbox("Any Chronic Diseases?", ["No", "Yes"])
    known_allergies = st.selectbox("Known Allergies?", ["No", "Yes"])
with col4:
    cancer_history = st.selectbox("Family History of Cancer?", ["No", "Yes"])
    major_surgeries = st.number_input("Number of Major Surgeries", min_value=0, max_value=10, value=0)

st.markdown('</div>', unsafe_allow_html=True)

# Encode inputs
def encode(val):
    return 1 if val == "Yes" else 0

# Predict button
if st.button("💰 Predict My Premium"):
    input_dict = {
        'Age': age,
        'AnyTransplants': encode(any_transplants),
        'AnyChronicDiseases': encode(any_chronic),
        'KnownAllergies': encode(known_allergies),
        'HistoryOfCancerInFamily': encode(cancer_history),
        'NumberOfMajorSurgeries': major_surgeries,
        'BMI': bmi
    }

    input_df = pd.DataFrame([input_dict])[selected_features]
    prediction = model.predict(input_df)[0]

    st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Annual Insurance Premium</div>
            ₹ {prediction:,.2f}
        </div>
    """, unsafe_allow_html=True)

    st.info("💡 This is an estimate based on your inputs. Actual premiums may vary by insurer.")

# Footer
st.markdown("---")
st.markdown('<p style="text-align:center; color:#9e9e9e; font-size:0.8rem;">Built with Streamlit • Insurance Premium Prediction ML Model</p>', unsafe_allow_html=True)
