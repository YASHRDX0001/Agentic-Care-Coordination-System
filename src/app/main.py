import streamlit as st
import pandas as pd
import os
import sys

# Add the project root to the python path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.preprocessor import load_data, preprocess_data

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
    uploaded_file = st.file_uploader("Upload your appointment data (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load data
            raw_df = load_data(uploaded_file)
            
            # Preprocess
            df = preprocess_data(raw_df)
            
            # Success message
            st.success(f"Successfully loaded and preprocessed dataset.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", df.shape[0])
            with col2:
                st.metric("Total Features", df.shape[1])

            # Data Preview
            st.subheader("Preprocessed Data Preview")
            st.dataframe(df.head())
            
            # Show new feature
            if 'LeadTime' in df.columns:
                st.info("✨ **Feature Engineering**: Calculated `LeadTime` (days between scheduling and appointment).")
                st.bar_chart(df['LeadTime'].value_counts().head(20))
            
            # Data Statistics
            with st.expander("View Detailed Statistics"):
                st.write(df.describe())
            
            # Missing Values
            if df.isnull().sum().sum() > 0:
                st.warning("⚠️ This dataset contains missing values.")
                st.write(df.isnull().sum()[df.isnull().sum() > 0])
            else:
                st.info("✅ No missing values detected.")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
