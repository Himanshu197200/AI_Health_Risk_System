# Intelligent Patient Risk Assessment System  
**AI Health Risk System — End-to-End Healthcare Analytics Platform**

 **Live App:**  
https://aihealthrisksystem5duvbgvx8pvz9xarixkmhgi.streamlit.app/

**GitHub Repository:**  
https://github.com/Himanshu197200/AI_Health_Risk_System

---

##  Overview

The **Intelligent Patient Risk Assessment System** is a machine learning–based healthcare analytics platform designed to predict potential health risks using structured clinical and lifestyle data.

The system currently focuses on estimating risk scores for:

-  Heart Disease  
-  Diabetes  

By leveraging supervised learning techniques, the application processes patient health indicators and provides continuous risk score predictions to support early detection and preventive healthcare analysis.

The project demonstrates a **production-ready ML workflow** including data preprocessing, advanced EDA, feature engineering, model training, evaluation, and Streamlit deployment.

---

##  Objective

The primary objectives of this project are to:

- Design a robust data preprocessing pipeline  
- Perform advanced exploratory data analysis  
- Engineer meaningful predictive features  
- Train and evaluate supervised machine learning models  
- Compare multiple regression algorithms  
- Provide interpretable risk score predictions  
- Deploy an interactive web-based interface  
- Maintain modular and scalable architecture  

The system emphasizes **model interpretability, evaluation transparency, and clean software engineering practices**.

---

#  Milestone 1 — ML-Based Risk Prediction (Completed)

Milestone 1 focuses strictly on **traditional machine learning models (no GenAI)** to build a reliable healthcare risk prediction pipeline.

---

##  Technology Stack

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Streamlit  

---

##  Dataset

The dataset consists of large-scale structured patient health information including:

- Demographic attributes (e.g., Age, Gender)  
- Physiological indicators (e.g., Blood Pressure, BMI)  
- Lifestyle factors (e.g., Smoking, Physical Activity)  
- Clinical health indicators  
- Risk score targets for supervised learning  

The dataset is stored in the `data/` directory.

---

#  Machine Learning Approach

The problem is formulated primarily as a **regression task** to estimate continuous patient risk scores.

###  Models Used

- Linear Regression  
- Decision Tree Regressor  

Separate pipelines were developed for:

- Heart Disease Risk  
- Diabetes Risk  

---

##  Evaluation Metrics

Model performance is evaluated using regression metrics:

- MAE (Mean Absolute Error)  
- MSE (Mean Squared Error)  
- RMSE (Root Mean Squared Error)  
- R² Score  

These metrics provide a comprehensive understanding of prediction error and model generalization.

---

#  Model Performance (Test Set)

##  Heart Disease Risk

| Model | MAE | MSE | RMSE | R² |
|------|-----|-----|------|-----|
| Linear Regression | **4.21** | **27.76** | **5.27** | **0.8835** |
| Decision Tree | 4.94 | 39.49 | 6.28 | 0.8344 |

 **Selected Model:** Linear Regression (better generalization)

---

##  Diabetes Risk

| Model | MAE | MSE | RMSE | R² |
|------|-----|-----|------|-----|
| Linear Regression | **1.91** | **5.83** | **2.42** | **0.9518** |
| Decision Tree | 2.83 | 12.70 | 3.56 | 0.8951 |

 **Selected Model:** Linear Regression (lower prediction error)

---

##  System Pipeline

Raw Data  
↓  
Preprocessing (preprocess.py)  
↓  
Feature Engineering (features.py)  
↓  
Model Training (train.py)  
↓  
Evaluation (evaluate.py)  
↓  
Saved Models (models/)  
↓  
Prediction Pipeline (predict.py)  
↓  
Streamlit App (main.py)

---

##  Project Structure
```text
AI_Health_Risk_System/  
│  
├── app/                 # Streamlit UI components  

├── data/                # Raw and processed datasets  

├── models/              # Saved model artifacts (.pkl)  

├── notebooks/           # EDA and experimentation  

├── src/  
│   ├── preprocess.py    # Data cleaning pipeline  

│   ├── features.py      # Feature engineering logic  

│   ├── train.py         # Model training  

│   ├── evaluate.py      # Model evaluation  

│   └── predict.py       # Inference pipeline  
│  
├── main.py              # Streamlit entry point  

├── requirements.txt

├── README.md  

└── PROJECT_WORKFLOW.md
```
## Setup Instructions
```text
1. Clone the repository:

   git clone <repository_url>

2. Navigate to the project directory:

   cd <project_directory>

3. Install dependencies:

   pip install -r requirements.txt

4. Run the application:

   streamlit run app.py
```
---

## Design Principles

- Modular code organization  
- Clear separation between preprocessing and modeling  
- Reproducible training workflow  
- Interpretable healthcare predictions  
- Scalable for future extensions  

---

## License

This project is developed for academic purposes.
