import streamlit as st
import torch
import numpy as np
import torch.nn as nn

# Model Structure
class YieldPredictor(nn.Module):
    def __init__(self):
        super(YieldPredictor, self).__init__()
        self.layer1 = nn.Linear(3, 16) 
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1) 

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x) 
        return x

# Load Model
# Cache the model to avoid reloading on every interaction
@st.cache_resource
def load_model():
    model = YieldPredictor()
    model.load_state_dict(torch.load('crop_yield.pth'))
    model.eval()
    return model

model = load_model()

# --- Custom CSS for "Apple-like" Aesthetic ---
st.markdown("""
<style>
    /* Global Styles & Font */
    @import url('https://fonts.googleapis.com/css2?family=sf-pro-display:wght@400;600&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1d1d1f;
    }
    
    .stApp {
        background-color: #f5f5f7;
        background-image: linear-gradient(135deg, #f5f5f7 0%, #ffffff 100%);
    }
    
    /* Vertically center the main content */
    .stMain {
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        min-height: 100vh !important;
    }

    /* Main Container (The "Card") */
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 3rem 4rem !important; /* Generous padding */
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.6);
        max-width: 950px;
        width: 100%;
        margin: auto;
        /* Remove default Streamlit top padding/margin adjustments if any */
        padding-top: 2rem !important; 
        padding-bottom: 2rem !important;
    }

    /* Mobile Responsive Adjustments */
    @media (max-width: 600px) {
        .block-container {
            padding: 1.5rem !important;
            max-width: 100%;
            margin: 1rem;
        }
        /* Allow scroll on mobile */
        .stMain {
            justify-content: flex-start !important; 
            padding-top: 2rem !important;
        }
    }

    /* Typography */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        font-size: 2.2rem !important;
        color: #1d1d1f !important;
        margin-bottom: 0.2rem !important;
        padding-top: 0 !important;
        text-align: center !important;
    }
    
    p {
        font-size: 1rem !important;
        color: #86868b !important;
        margin-bottom: 2rem !important;
        text-align: center !important;
    }

    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: #0071e3 !important;
    }

    /* Center the label text above the slider */
    .stSlider label {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
        text-align: center !important;
        font-size: 0.95rem !important;
        color: #86868b !important;
    }

    /* Button Styling */
    .stButton > button {
        background-color: #0071e3 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 980px !important;
        padding: 16px 32px !important; /* Larger touch area */
        font-size: 17px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(0, 113, 227, 0.3) !important;
        margin-top: 20px;
        
        /* Flexbox for perfect centering */
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin-left: auto !important;
        margin-right: auto !important;
        line-height: normal !important;
    }
    
    /* Fix internal paragraph margin inside Streamlit button */
    .stButton > button p {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
        line-height: 1.2 !important;
        padding: 0 !important;
    }
    
    /* Force text color inside button */
    .stButton > button p, .stButton > button span, .stButton > button:hover, .stButton > button:focus {
        color: #ffffff !important; 
    }

    .stButton > button:hover {
        background-color: #0077ed !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(0, 113, 227, 0.4) !important;
    }

    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #e0fcdc 0%, #d4f7cd 100%);
        color: #155724;
        border-radius: 12px;
        padding: 20px;
        margin-top: 24px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        box-shadow: 0 4px 20px rgba(76, 175, 80, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.6);
        animation: fadeIn 0.5s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Hide the top header decoration */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- App Interface ---

# Header (Compact & Centered)
st.title("🌾 Wheat Yield Predictor")
st.markdown("<p>Predict optimal wheat production based on real-time environmental factors.</p>", unsafe_allow_html=True)

# Main Input Section
# Using columns for wide layout on desktop (side-by-side saves vertical space)
# On mobile, these will naturally stack.
c1, c2, c3 = st.columns(3)

with c1:
    rain = st.slider("Average Rainfall (mm)", 50, 300, 150)

with c2:
    temp = st.slider("Temperature (°C)", 20, 40, 30)

with c3:
    fert = st.slider("Fertilizer Usage (kg/hectare)", 20, 100, 50)

st.write("") # Minimal Spacer

# Center the button contextually
# We use one central column with a specific width ratio to align the button or let it center via CSS
b1, b2, b3 = st.columns([1, 2, 1])

with b2:
    if st.button("Predict Yield", use_container_width=True): 
        # Manual rough scaling
        input_data = np.array([[(rain - 175)/75, (temp - 30)/10, (fert - 60)/20]]) 
        tensor_input = torch.tensor(input_data, dtype=torch.float32)
        
        with torch.no_grad():
            prediction = model(tensor_input).item()
            
        real_prediction = (prediction * 1.0) + 3.0 
        
        st.markdown(f"""
        <div class="result-card">
            Predicted: {real_prediction:.2f} Tonnes/Hectare
        </div>
        """, unsafe_allow_html=True)
