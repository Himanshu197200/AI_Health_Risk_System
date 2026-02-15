# Intelligent Patient Risk Assessment System

## Overview

The Intelligent Patient Risk Assessment System is a machine learning–based healthcare analytics platform designed to predict potential health risks using structured clinical and lifestyle data.

The system currently focuses on estimating risk for:

- Heart Disease
- Diabetes

By leveraging supervised learning techniques, the application processes patient health indicators and provides probability-based risk predictions to support early detection and preventive healthcare analysis.

---

## Objective

The objective of this project is to:

- Design a structured data preprocessing pipeline  
- Train and evaluate supervised machine learning models  
- Compare multiple classification algorithms  
- Provide interpretable and probability-based risk predictions  
- Deploy an interactive web-based interface for user interaction  

The system emphasizes model interpretability, evaluation transparency, and clean software architecture.

---

## Technology Stack

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Streamlit  

---

## Dataset

The dataset consists of structured patient health information including:

- Demographic attributes (e.g., Age)
- Physiological indicators (e.g., Blood Pressure, BMI)
- Lifestyle factors (e.g., Smoking, Physical Activity)
- Clinical outcome labels for supervised learning

The dataset is stored in the `data/` directory.

---

## Machine Learning Approach

The system uses supervised classification models, including:

- Logistic Regression
- Decision Tree Classifier

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The system outputs probability-based risk predictions to enhance interpretability in healthcare contexts.

---

## Project Structure

project/
│
├── data/                # Dataset files  
├── notebooks/           # Exploratory Data Analysis  
├── model/               # Preprocessing and training scripts  
├── app.py               # Streamlit application  
├── requirements.txt     # Project dependencies  
└── README.md  

---

## Setup Instructions

1. Clone the repository:

   git clone <repository_url>

2. Navigate to the project directory:

   cd <project_directory>

3. Install dependencies:

   pip install -r requirements.txt

4. Run the application:

   streamlit run app.py

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
