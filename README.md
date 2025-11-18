# Student_Depression_Prediction_System

## Project Overview

This is a university group project focuses on developing, evaluating, and comparing multiple supervised machine learning models to predict student depression levels based on a comprehensive dataset of academic, lifestyle, and social factors. The primary goal was to identify the most accurate, reliable, and ethically sound model for deployment as a decision-support tool.

### Problem Statement
Student mental health is a critical concern in higher education. Early identification of students at risk of depression is essential for timely intervention and support. This system aims to provide a data-driven approach to flag high-risk individuals.

### Dataset
* **Source:** Kaggle - Student Depression Dataset
* **Records:** Approximately 20,000
* **Features:** 18 features covering areas such as academic pressure, financial concerns, sleep duration, social life, and suicidal thoughts.

***

##  Model Comparison and Selection

Six distinct machine learning models were developed, trained, and rigorously evaluated using a standardized preprocessing pipeline.

### Performance Summary

| Model | Developed By | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest Classifier** | Member 1 (Developed by me) | **86.25%** | **0.86** | **0.86** | **0.87** | **SELECTED** |
| Support Vector Machine (SVM) | Member 2 | 86.21% | 0.85 | 0.83 | 0.84 | Rejected |
| Artificial Neural Network (ANN) | Member 3 | 85.68% | 0.84 | 0.85 | 0.85 | Rejected |
| K-Nearest Neighbors (KNN) | Member 4 | 83.64% | 0.80 | 0.79 | 0.79 | Rejected |
| Logistic Regression | Member 5 | 83.44% | 0.83 | 0.83 | 0.83 | Rejected |
| Decision Tree | Member 6 | *Lower* | *Moderate* | *Moderate* | *Moderate* | Rejected |

### Final Model Selection Rationale

The **Random Forest Classifier** was selected as the optimal model for deployment based on the following criteria:

1.  **Highest Accuracy:** Achieved the peak accuracy of **86.25%**.
2.  **Balanced Performance:** Strong and consistent results across all metrics, crucial for minimizing **False Negatives** (missing a student who is depressed).
3.  **Robustness & Generalization:** Showed resistance to overfitting, providing stable and reliable results.
4.  **Interpretability:** Allows for clear feature importance analysis.

***

##  Deployment & Application (Streamlit Predictor)

The final selected **Random Forest Classifier** model was saved and integrated into a user-friendly **Streamlit web application** to provide a functional interface for real-time predictions.

* **Application Type:** **Streamlit App (`app.py`)**
* **Functionality:** The app allows users to input student data via a web form and instantly receive a prediction (Depression/No Depression) along with a confidence score.

***

##  How to Run Locally

### Prerequisites

* Python 3.x
* A stable internet connection (for package installation)
* The trained model file (e.g., `model.joblib`) and the Streamlit app script (`app.py`) must be in the `Depression_Predictor_App` directory.

### Step 1: Clone the Repository

```bash```
git clone <your-repo-link>
cd <repo-name>

### Step 2: Install Required Packages

The Streamlit app requires specific packages, including Streamlit itself, for deployment.
```bash```
pip install scikit-learn joblib pandas streamlit

### Step 3: Running the Streamlit App (Deployment)

Navigate to the directory containing your app.py file and run the following command:
```bash```
python -m streamlit run app.py

Upon successful execution, you will see output similar to this in your terminal:

 You can now view your Streamlit app in your browser.
 Local URL: http://localhost:8501
 Network URL: [http://192.168.1.59:8501](http://192.168.1.59:8501)
Open the Local URL in your web browser to interact with the Depression Predictor Application.

**Happy Coding!**
