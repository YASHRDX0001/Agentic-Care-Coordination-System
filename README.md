# Clinical Appointment No-Show Prediction and Agentic Care Coordination System

## Project Objective
Design and implement an AI-based healthcare operations system that:
1.  Predicts patient appointment no-shows using historical scheduling data.
2.  Extends into an agentic AI assistant that generates actionable care coordination and intervention recommendations.

## Requirements & Constraints
-   **Team Size:** 3-4 students
-   **APIs:** No paid APIs allowed. Only free-tier or open-source models.
-   **Interface:** Mandatory user interface (Streamlit/Gradio).
-   **Deployment:** Must be publicly hosted (e.g., Hugging Face Spaces, Streamlit Community Cloud, Render). Localhost demos are not accepted.

## Milestones

### Milestone 1: ML-Based Appointment No-Show Prediction (Mid-Sem)
-   **Goal:** Build a supervised learning model to predict no-show probability.
-   **Inputs:** Appointment lead time, time/day, patient history, department.
-   **Deliverables:**
    -   Problem understanding & operational use-case.
    -   System architecture diagram.
    -   Working local application with basic UI.
    -   Model evaluation and performance analysis.

### Milestone 2: Agentic AI Care Coordination Assistant (End-Sem)
-   **Goal:** Agentic AI assistant that reasons about risks and suggests interventions.
-   **Features:**
    -   Analyze predicted risks.
    -   Retrieve best-practice guidelines.
    -   Generate actionable intervention recommendations.
    -   Handle missing/noisy data.

## Tech Stack (Recommended)
-   **ML:** scikit-learn, pandas, NumPy
-   **LLMs:** Open-source models or free-tier APIs
-   **Agents:** LangGraph
-   **Vector Store:** Chroma, FAISS
-   **UI:** Streamlit or Gradio
-   **Hosting:** Hugging Face Spaces, Streamlit Cloud

## Setup
1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the application:
    ```bash
    streamlit run src/app/main.py
    ```
