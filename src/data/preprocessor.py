import pandas as pd
import numpy as np


def load_data(file):
    """
    Load appointment data from a CSV file or file-like object.
    
    Args:
        file: A file path (str) or file-like object (e.g., Streamlit UploadedFile).
    
    Returns:
        pd.DataFrame: The raw appointment data.
    """
    df = pd.read_csv(file)
    return df


def preprocess_data(df):
    """
    Clean and preprocess the raw appointment dataset.

    Steps:
        1. Drop duplicate rows.
        2. Parse date columns (ScheduledDay, AppointmentDay) to datetime.
        3. Engineer the LeadTime feature (days between scheduling and appointment).
        4. Encode the target variable (No-show) as binary (1 = No-show, 0 = Show).
        5. Drop columns that are not useful for modelling.
        6. Encode categorical variables.
        7. Handle any remaining missing values.

    Args:
        df (pd.DataFrame): Raw appointment data.

    Returns:
        pd.DataFrame: Cleaned and feature-engineered dataframe.
    """
    df = df.copy()

    # --- 1. Remove duplicates ---
    df.drop_duplicates(inplace=True)

    # --- 2. Parse date columns ---
    date_columns = ['ScheduledDay', 'AppointmentDay']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # --- 3. Feature Engineering: LeadTime ---
    if 'ScheduledDay' in df.columns and 'AppointmentDay' in df.columns:
        df['LeadTime'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
        # Negative lead times are likely data errors; clip to 0
        df['LeadTime'] = df['LeadTime'].clip(lower=0)

    # --- 4. Encode target variable ---
    if 'No-show' in df.columns:
        df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})
    elif 'No_show' in df.columns:
        df['No_show'] = df['No_show'].map({'Yes': 1, 'No': 0})

    # --- 5. Drop non-predictive / ID columns ---
    columns_to_drop = ['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay']
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)

    # --- 6. Encode categorical variables ---
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Exclude target if it's still object type
    categorical_cols = [c for c in categorical_cols if c not in ['No-show', 'No_show']]

    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    # --- 7. Handle missing values ---
    df.dropna(inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df
