import streamlit as st
import pandas as pd
import os

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
    st.header("1. Data Upload")
    uploaded_file = st.file_uploader("Upload your appointment data (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Success message
            st.success(f"Successfully loaded dataset with {df.shape[0]} records and {df.shape[1]} columns.")
            
            # Data Preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Data Statistics
            st.subheader("Dataset Statistics")
            st.write(df.describe())
            
            # Missing Values
            if df.isnull().sum().sum() > 0:
                st.warning("⚠️ This dataset contains missing values.")
                st.write(df.isnull().sum()[df.isnull().sum() > 0])
            else:
                st.info("✅ No missing values detected in the dataset.")
                
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
