import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pickle
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

# --- Helper Functions ---

def to_int(x):
    """Converts a pandas Series to integer type."""
    return x.astype(int)

def load_preprocessor(path):
    """Loads the saved preprocessor object (ColumnTransformer)."""
    with open(path, 'rb') as f:
        preprocessor = joblib.load(f)
    return preprocessor

def load_model(path):
    """Loads the saved XGBoost model."""
    model = xgb.Booster()
    model.load_model(path)
    return model

def prepare_data(sample, preprocessor, columns):
    """Prepares the user's input data for prediction."""
    input_df = pd.DataFrame([sample])
    
    # Ensure boolean types are correct
    input_df['road_signs_present'] = input_df['road_signs_present'].astype(bool)
    input_df['public_road'] = input_df['public_road'].astype(bool)
    input_df['school_season'] = input_df['school_season'].astype(bool)

    # Ensure the column order is the same as during training
    input_df = input_df[columns]
    
    # Transform the data using the loaded preprocessor
    input_processed = preprocessor.transform(input_df)
    
    return input_processed

def predict(model, preprocessor, sample, columns):
    """Makes a prediction based on user input."""
    input_data = prepare_data(sample, preprocessor, columns)
    dmatrix = xgb.DMatrix(input_data)
    prediction = model.predict(dmatrix)
    return prediction[0]


# --- Main Application ---

# 1. Define consistent file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'xgb_model.json')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'preprocessor.pkl')

# 2. Load the model and the preprocessor
try:
    model = load_model(MODEL_PATH)
    preprocessor = load_preprocessor(PREPROCESSOR_PATH)
except FileNotFoundError:
    st.error(
        "Model or preprocessor file not found. "
        "Please ensure 'xgb_model.json' and 'preprocessor.pkl' "
        "are in the same directory as this script."
    )
    st.stop() 

# 3. Define the columns used by the model (must match the preprocessor)
columns = [
    'road_type', 'num_lanes', 'speed_limit', 'weather', 
    'road_signs_present', 'time_of_day', 'curvature',
    'holiday', 'lighting', 'public_road', 'school_season', 
    'num_reported_accidents'
]

# 4. Streamlit app interface
st.set_page_config(page_title="Road Accident Prediction", layout="wide")
st.title("Welcome to the Road Accident Prediction App")
st.markdown("Select the conditions below to predict the risk of an accident.")

st.sidebar.header("Your Accident Risk Guess")
user_pred = st.sidebar.slider('Guess the risk level:', 0.0, 1.0, 0.5, 0.01)

st.header("Enter Road and Environmental Conditions")


col1, col2 = st.columns(2)

with col1:
    sample = {}
    sample['road_type'] = st.selectbox("Road Type", ['urban', 'rural', 'highway'])
    sample['num_lanes'] = st.slider("Number of Lanes", 1, 4, 2, 1)
    sample['speed_limit'] = st.slider("Speed Limit", 25, 70, 30, step=5)
    sample['curvature'] = st.slider("Road Curvature", 0.0, 1.0, 0.5, 0.01)
    sample['weather'] = st.selectbox("Weather", ['clear', 'rainy', 'foggy'])
    sample['num_reported_accidents'] = st.slider("Number of Prior Reported Accidents in Area", 0, 100, 10, 1)

with col2:
    sample['time_of_day'] = st.selectbox("Time of Day", ['morning', 'afternoon', 'evening'])
    sample['lighting'] = st.selectbox("Lighting Conditions", ['daylight', 'dim', 'night'])
    sample['road_signs_present'] = st.toggle("Road Signs Present", value=True)
    sample['public_road'] = st.toggle("Is it a public road?", value=True)
    sample['school_season'] = st.toggle("Is it school season?", value=True)
    sample['holiday'] = st.toggle("Is it a holiday?", value=False)


# 5. The Predict Button and Results
if st.button('Predict Accident Risk', type="primary"):
    prediction = predict(model, preprocessor, sample, columns)
    st.markdown("---") 
    st.subheader("Prediction Results")
    
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.metric(label="Predicted Accident Risk", value=f"{prediction:.2f}")

    with res_col2:
        st.metric(label="Your Guess", value=f"{user_pred:.2f}")

    with res_col3:
        difference = abs(prediction - user_pred)
        st.metric(label="Difference", value=f"{difference:.2f}", delta=f"{-difference:.2f}")

    st.progress(float(prediction))
    
    if prediction > 0.7:
        st.error("High risk of accident detected. Please be extra cautious.")
    elif prediction > 0.4:
        st.warning("Moderate risk of accident. Stay alert.")
    else:
        st.success("Low risk of accident. Drive safely!")
