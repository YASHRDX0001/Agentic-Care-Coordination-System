import streamlit as st
import pandas as pd
import os
import sys
import importlib.util
from pathlib import Path

# Resolve path to preprocessor.py relative to this file
_current_dir = Path(__file__).resolve().parent          # src/app/
_preprocessor_path = _current_dir.parent / "data" / "preprocessor.py"  # src/data/preprocessor.py

# Load the module directly using importlib (works on Streamlit Cloud)
_spec = importlib.util.spec_from_file_location("preprocessor", str(_preprocessor_path))
_preprocessor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_preprocessor)

load_data = _preprocessor.load_data
preprocess_data = _preprocessor.preprocess_data

# Page configuration
st.set_page_config(
    page_title="No-Show Prediction System",
    page_icon="🏥",
    layout="wide"
)

def main():
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.info("Upload the appointment dataset to get started.")

    # Main content
    st.title("🏥 Clinical Appointment No-Show Prediction System")
    st.markdown("""
    Welcome to the **Agentic Care Coordination System**. 
    This tool predicts patient no-show risks and generates intervention strategies.
    """)

    # File Uploader
    st.header("1. Data Upload & Preprocessing")
    dataset_file = st.file_uploader("Upload your appointment data (CSV)", type=["csv"])

    if dataset_file is not None:
        try:
            # Load data
            raw_appointment_data = load_data(dataset_file)
            
            # Preprocess
            processed_data = preprocess_data(raw_appointment_data)
            
            # Success message
            st.success(f"Successfully loaded and preprocessed dataset.")
            
            records_col, features_col = st.columns(2)
            with records_col:
                st.metric("Total Records", processed_data.shape[0])
            with features_col:
                st.metric("Total Features", processed_data.shape[1])

            # Data Preview
            st.subheader("Preprocessed Data Preview")
            st.dataframe(processed_data.head())
            
            # Show new feature
            if 'LeadTime' in processed_data.columns:
                st.info("✨ **Feature Engineering**: Calculated `LeadTime` (days between scheduling and appointment).")
                st.bar_chart(processed_data['LeadTime'].value_counts().head(20))
            
            # Data Statistics
            with st.expander("View Detailed Statistics"):
                st.write(processed_data.describe())
            
            # Missing Values
            if processed_data.isnull().sum().sum() > 0:
                st.warning("⚠️ This dataset contains missing values.")
                st.write(processed_data.isnull().sum()[processed_data.isnull().sum() > 0])
            else:
                st.info("✅ No missing values detected.")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
