# Clinical Appointment No-Show Prediction and Agentic Care Coordination System

An AI-powered healthcare operations system that predicts patient appointment no-shows using machine learning and extends into an intelligent agent-based assistant for generating actionable care coordination strategies.

## Project Overview

Patient no-shows are a significant issue in healthcare, leading to wasted resources and longer wait times for other patients. Manual risk assessment is inefficient and often inconsistent.

This system automates risk assessment using traditional Machine Learning models and provides a scalable architecture for intelligent health guidance.

The project is divided into two milestones:

**Milestone 1 – ML-Based Appointment No-Show Prediction**
Predicts the likelihood of a patient missing an appointment using historical scheduling data.

**Milestone 2 – Agentic AI Care Coordination Assistant**
Extends the system using an agentic workflow to generate structured intervention recommendations based on predicted risks.

---

## Problem Statement

Healthcare institutions require efficient systems to:
-   Identify high-risk patients early.
-   Maximize resource utilization by reducing no-show rates.
-   Assist care coordinators with data-driven intervention strategies.
-   Maintain consistency in patient communication and follow-up.

This project builds a structured ML-based risk prediction system and extends it into an intelligent decision-support assistant.

---

## Key Features

### Milestone 1 – Machine Learning Risk Prediction

-   **Data Cleaning & Preprocessing**: Handling dates, categorical features, and calculating `LeadTime`.
-   **Feature Engineering**: Creating operational features like "days until appointment".
-   **Multiple ML Models**:
    -   Logistic Regression
    -   Decision Tree / Random Forest
    -   Gradient Boosting (XGBoost/LightGBM)
-   **Model Comparison & Evaluation**: Using Precision, Recall, F1-Score, and ROC-AUC.
-   **Risk Score Generation**: 0-1 probability estimation.
-   **Interactive Streamlit Interface**:
    -   Upload patient data (CSV).
    -   View risk predictions and feature importance.
-   **Public Deployment**: Hosted on Streamlit Community Cloud or Hugging Face Spaces.

### Milestone 2 – Agentic AI Extension

-   **Risk Analysis**: Agents analyze high-risk cases to determine contributing factors.
-   **Knowledge Retrieval**: Fetches best-practice care coordination guidelines (RAG).
-   **Actionable Recommendations**: Generates specific intervention steps (e.g., "Send SMS reminder 2 days before" or "Call to arrange transportation").
-   **Structured Reporting**: Outputs a clear plan for care coordinators.

---

## System Architecture

### Milestone 1 Workflow
1.  **User Input**: Upload appointment data.
2.  **Data Preprocessing**: Clean dates, encode features.
3.  **Feature Engineering**: Calculate `LeadTime`, `PreviousNoShows`.
4.  **ML Model**: Predict probability.
5.  **Risk Score**: Classify as Low/Medium/High Risk.
6.  **UI Display**: Show dashboard and risk table.

### Milestone 2 Workflow
1.  **Risk Prediction Output**: High-risk patients identified.
2.  **Agent Workflow**: LangGraph agent receives patient context.
3.  **Knowledge Retrieval**: Consults guideline database.
4.  **Reasoning**: Determines best intervention strategy.
5.  **Recommendation Generation**: Produces tailored advice.
6.  **UI Display**: Presents recommendations to the care coordinator.

---

## Project Structure

```
├── data/
│   ├── raw/             # Original dataset
│   ├── processed/       # Cleaned data
├── notebooks/           # Jupyter notebooks for EDA and prototyping
├── src/
│   ├── app/
│   │   └── main.py      # Streamlit application entry point
│   ├── data/
│   │   └── preprocessor.py # Data loading and cleaning logic
│   ├── features/        # Feature engineering scripts
│   ├── models/          # Model training scripts
│   ├── agent/           # Agentic AI logic
│   └── utils/           # Helper functions
├── tests/               # Unit tests
└── README.md            # Project documentation
```

---

## Installation & Setup

Follow these steps to set up the project on your local machine.

### Prerequisites

-   **Python 3.9+**: Ensure you have Python installed.
-   **pip**: Package installer for Python.

### Step 1: Clone the Repository

```bash
git clone https://github.com/sanjana2505006/Agentic-Care-Coordination-System.git
cd Agentic-Care-Coordination-System
```

### Step 2: Create a Virtual Environment

It is recommended to create a virtual environment to avoid dependency conflicts.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run src/app/main.py
```

The application will launch in your default web browser.
