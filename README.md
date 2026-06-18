# Fraudshield-With-XAI

A fraud payment detection tool operating before payment authorization. Suspicious transactions are blocked immediately. Features a responsive dashboard with a real-time graph, a pie chart, and SHAP explainability.

## Setup
```bash
pip install django djangorestframework scikit-learn pandas shap joblib

This installs Django and Django REST Framework to build the backend APIs, alongside Pandas, Scikit-learn, SHAP, and Joblib to handle data manipulation, train machine learning models, interpret them with Explainable AI (XAI), and save them for deployment.
