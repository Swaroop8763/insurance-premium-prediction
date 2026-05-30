import streamlit as st
import joblib
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

    html, body, [class*="css"], .stApp {
        font-family: 'Sora', sans-serif !important;
        color: #1a1a2e !important;
    }

    .stApp {
        background: linear-gradient(135deg, #e8f0fe 0%, #f0f4f8 100%) !important;
    }

    h1, h2, h3 {
        color: #1a237e !important;
        font-weight: 700 !important;
    }

    p, label, div {
        color: #1a1a2e !important;
    }

    .subtitle {
        color: #3949ab !important;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Input labels */
    .stNumberInput label, .stSelectbox label {
        color: #1a237e !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Cards */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e7ff;
    }

    .card h3 {
        color: #1a237e !important;
        margin-bottom: 1rem;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #1a237e, #3949ab);
        color: white !important;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px rgba(26,35,126,0.3);
    }

    .result-label {
        font-size: 0.9rem;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 0.5rem;
        color: white !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #1a237e, #3949ab) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #3949ab, #1a237e) !important;
        box-shadow: 0 4px 15px rgba(26,35,126,0.4) !important;
    }

    /* Info box */
    .stAlert {
        color: #1a237e !important;
        font-weight: 500 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280 !important;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Hardcoded selected features
selected_features = ['Age', 'AnyTransplants', 'AnyChronicDiseases', 'KnownAllergies',
                     'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries', 'BMI']

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('insurance_model.joblib')
    return model

model = load_model()

# Header
st.markdown("## 🏥 Insurance Premium Predictor")
st.markdown('<p class="subtitle">Fill in your health details to get an estimated insurance premium</p>', unsafe_allow_html=True)

# Personal Details Card
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

if bmi < 18.5:
    bmi_category = "Underweight 🟡"
elif bmi < 25:
    bmi_category = "Normal ✅"
elif bmi < 30:
    bmi_category = "Overweight 🟠"
else:
    bmi_category = "Obese 🔴"

st.info(f"📊 Your calculated BMI: **{bmi:.1f}** — {bmi_category}")
st.markdown('</div>', unsafe_allow_html=True)

# Medical History Card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🏨 Medical History")

col4, col5 = st.columns(2)
with col4:
    any_transplants = st.selectbox("Any Transplants?", ["No", "Yes"])
    any_chronic = st.selectbox("Any Chronic Diseases?", ["No", "Yes"])
    known_allergies = st.selectbox("Known Allergies?", ["No", "Yes"])
with col5:
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
st.markdown('<p class="footer">Built with Streamlit • Insurance Premium Prediction ML Model</p>', unsafe_allow_html=True)
