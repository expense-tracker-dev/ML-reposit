# Expense Tracker — ML Module

## Overview
This module predicts daily expenses and estimates monthly spending using a trained Random Forest regression model.

## Files
- `main.py` — FastAPI application and prediction endpoints
- `train_model2.py` — model training and evaluation
- `train_model.py` — additional training script
- `check_data.py` — dataset checks
- `personal_finance_dataset.csv` — training dataset
- `expense_model.pkl` — saved trained model

## Run the API
Install the required dependencies if needed:

`python -m pip install pandas numpy scikit-learn joblib fastapi "uvicorn[standard]"`

Run the server from this project folder:

`python -m uvicorn main:app --reload`

Open the API documentation:

`http://127.0.0.1:8000/docs`

## Available Endpoints
- `GET /` — API welcome message
- `GET /health` — service health check
- `GET /predict/daily` — tomorrow's expense prediction
- `GET /predict/monthly` — current month-end estimate and next month's prediction

## Evaluation
- Model MAE: ₹320.32
- Model RMSE: ₹430.56
- Seven-day baseline MAE: ₹238.98

The baseline currently has lower error than the ML model. Model performance requires further improvement.

## Integration Status
The API has been tested locally. It currently uses saved training history rather than the logged-in user's actual expense history. Django REST Framework integration and user-specific predictions remain pending.

## Notes
The API is currently available only on the local machine at `127.0.0.1:8000`. Another device or service will need an appropriate network address or deployment configuration to access it.