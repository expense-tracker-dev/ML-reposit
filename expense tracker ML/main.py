
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "expense_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "expense_model.pkl not found. Run train_model2.py first."
    )

saved = joblib.load(MODEL_PATH)
model = saved["model"]
history = saved["daily_history"].copy()

app = FastAPI(title="Expense Tracker ML API")


def predict_daily(dates):
    past = history.copy()
    results = []

    for date in dates:
        if len(past) < 30:
            raise ValueError("At least 30 days of history are required.")

        row = pd.DataFrame([{
            "weekday": date.dayofweek,
            "day": date.day,
            "month": date.month,
            "day_year": date.dayofyear,
            "lag1": past.iloc[-1],
            "lag7": past.iloc[-7],
            "avg7": past.iloc[-7:].mean(),
            "avg30": past.iloc[-30:].mean()
        }])

        amount = max(float(model.predict(row)[0]), 0)
        results.append(amount)
        past.loc[date] = amount

    return results


@app.get("/")
def home():
    return {
        "message": "Expense Tracker ML API is running!"
    }
        
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Expense Tracker ML API"
    }


@app.get("/predict/daily")
def daily_prediction():
    today = pd.Timestamp.today().normalize()
    tomorrow = today + pd.Timedelta(days=1)

    amount = predict_daily([tomorrow])[0]

    return {
        "date": str(tomorrow.date()),
        "predicted_expense": round(amount, 2),
        "currency": "INR"
    }


@app.get("/predict/monthly")
def monthly_prediction():
    today = pd.Timestamp.today().normalize()
    month_start = today.replace(day=1)

    current_end = month_start + pd.offsets.MonthEnd(0)
    next_start = month_start + pd.offsets.MonthBegin(1)
    next_end = next_start + pd.offsets.MonthEnd(0)

    future_dates = pd.date_range(
        today + pd.Timedelta(days=1),
        next_end,
        freq="D"
    )

    amounts = predict_daily(list(future_dates))
    future = pd.Series(amounts, index=future_dates)

    current_actual = history[
        (history.index >= month_start) &
        (history.index <= today)
    ].sum()

    current_remaining = future[
        future.index <= current_end
    ].sum()

    next_total = future[
        (future.index >= next_start) &
        (future.index <= next_end)
    ].sum()

    return {
        "current_month": str(today.strftime("%B %Y")),
        "estimated_month_end_expense": round(
            float(current_actual + current_remaining), 2
        ),
        "next_month": str(next_start.strftime("%B %Y")),
        "predicted_next_month_expense": round(
            float(next_total), 2
        ),
        "currency": "INR"
    }
