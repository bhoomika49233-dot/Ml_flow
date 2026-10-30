# Iris MLflow Project

## Setup

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Run

1. Train and log experiments (creates mlflow.db):

       python train.py

2. Browse runs in the MLflow UI, then open http://127.0.0.1:5000:

       mlflow ui --backend-store-uri sqlite:///mlflow.db

3. Load the best model and predict:

       python predict.py