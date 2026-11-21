"""
California Home Price Prediction App

This Streamlit application uses a trained XGBoost pipeline
to predict closing prices for California single-family homes.

Users can:
1. Select an existing property from the April 2026 test dataset.
2. Enter basic property details and use precomputed features
   from a selected reference property.

Run from the project directory:
    streamlit run app.py
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Define project directories
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

# Allow Python to import custom modules from src (preprocessing.py)
sys.path.insert(0, str(SRC_DIR))

# Load saved model and test data set
MODEL_PATH = BASE_DIR / "models" / "xgboost_school_hoa_flooring_tuned.joblib"

FEATURES_PATH = BASE_DIR / "results" / "streamlit_test_features.csv"

ACTUAL_PRICES_PATH = BASE_DIR / "results" / "streamlit_test_actual_prices.csv"

ADDRESSES_PATH = BASE_DIR / "results" / "streamlit_test_addresses.csv"

# Load fitted model and test data
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    features = pd.read_csv(FEATURES_PATH, dtype={"PostalCode": str})
    actual_prices = pd.read_csv(ACTUAL_PRICES_PATH)
    addresses = pd.read_csv(ADDRESSES_PATH)

    return features, actual_prices, addresses

model = load_model()
X_test, actual_prices, addresses = load_data()

# Verify that the test features, actual prices, and addresses have matching row counts
assert len(X_test) == len(actual_prices) == len(addresses), ("The number of test properties, actual prices, and addresses must match.")

###Streamlit Interface###
st.title("California Home Price Predictor")

st.write("Estimate closing prices for California single-family homes"
         " using a trained XGBoost model.")

# Choose prediction method
option = st.radio("Choose a prediction method:",
    ["Select an existing property", "Enter property details"])

if option == "Select an existing property":
    st.subheader("Select an Existing Property")

    property_index = st.selectbox(
    "Choose a property:",options=range(len(X_test)),
    format_func=lambda i: (f"Property {i + 1} - "
                           f"{addresses.iloc[i]['UnparsedAddress']}, "
                           f"{addresses.iloc[i]['City']}, "
                           f"{addresses.iloc[i]['StateOrProvince']} "
                           f"{addresses.iloc[i]['PostalCode']}"))
    
    selected_property = X_test.iloc[[property_index]]

    if st.button("Predict Closing Price"):
        # Predict log closing price
        predicted_log_price = model.predict(selected_property)[0]

        # Convert prediction to dollars
        predicted_price = np.expm1(predicted_log_price)

        # Get actual closing price
        actual_price = actual_prices.iloc[property_index]["ActualClosePrice"]

        # Calculate absolute percentage error
        ape = abs(actual_price - predicted_price) / actual_price * 100

        # Display prediction results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Predicted Closing Price", value=f"${predicted_price:,.2f}")

        with col2:
            st.metric(label="Actual Closing Price", value=f"${actual_price:,.2f}")

        with col3:
            st.metric(label="Absolute Percentage Error", value=f"{ape:.2f}%")

else:
    st.subheader("Enter Property Details")

    # Select a reference property for precomputed features
    ref_index = st.selectbox(
    "Choose a reference property:",
    options=range(len(X_test)),
    format_func=lambda i: (f"Property {i + 1} - "
                           f"{addresses.iloc[i]['UnparsedAddress']}, "
                           f"{addresses.iloc[i]['City']}, "
                           f"{addresses.iloc[i]['StateOrProvince']} "
                           f"{addresses.iloc[i]['PostalCode']}"))
    
    ref_property = X_test.iloc[ref_index]

    st.info("Modify the property characteristics below. "
            "Other features are taken from the selected reference property.")

   # Collect user inputs using ranges consistent with preprocessing
    living_area = st.number_input("Living Area (sq ft)", min_value=1.0, value=float(ref_property["LivingArea"]))
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=20, value=int(ref_property["BedroomsTotal"]))
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=20, value=int(ref_property["BathroomsTotalInteger"]))
    lot_size = st.number_input("Lot Size (sq ft)", min_value=1.0, value=float(ref_property["LotSizeSquareFeet"]))

    # Create a copy of the reference property
    input_property = X_test.iloc[[ref_index]].copy()

    # Update the reference property with user-entered values
    input_property["LivingArea"] = living_area
    input_property["BedroomsTotal"] = bedrooms
    input_property["BathroomsTotalInteger"] = bathrooms
    input_property["LotSizeSquareFeet"] = lot_size

    if st.button("Predict Closing Price"):
        # Predict log closing price
        predicted_log_price = model.predict(input_property)[0]

        # Convert prediction to dollars
        predicted_price = np.expm1(predicted_log_price)

        # Display the prediction results
        st.metric(label="Predicted Closing Price", value=f"${predicted_price:,.2f}")