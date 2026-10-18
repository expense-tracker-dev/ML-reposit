
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from calendar import monthrange
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load dataset
BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "personal_finance_dataset.csv")

df.columns = df.columns.str.strip().str.lower()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

df = df.dropna(subset=["date", "amount"])
df["income_or_expense"] = (
    df["income_or_expense"].astype(str).str.strip().str.lower()
)
df = df[df["income_or_expense"] == "expense"]
df = df[df["amount"] >= 0]

# Use only dates up to today
today = pd.Timestamp.today().normalize()
df = df[df["date"] <= today]

if df.empty:
    raise ValueError("No historical expense data found.")

# 2. Create daily expense history
daily = df.groupby(df["date"].dt.normalize())["amount"].sum()
daily = daily.sort_index().asfreq("D", fill_value=0)

# Fill missing calendar days through today with zero
full_dates = pd.date_range(daily.index.min(), today, freq="D")
daily = daily.reindex(full_dates, fill_value=0)

if len(daily) < 60:
    raise ValueError("At least 60 days of data are required.")

# 3. Build features for daily prediction
X = pd.DataFrame(index=daily.index)
X["weekday"] = daily.index.dayofweek
X["day"] = daily.index.day
X["month"] = daily.index.month
X["day_year"] = daily.index.dayofyear
X["lag1"] = daily.shift(1)
X["lag7"] = daily.shift(7)
X["avg7"] = daily.shift(1).rolling(7).mean()
X["avg30"] = daily.shift(1).rolling(30).mean()

data = X.copy()
data["target"] = daily
data = data.dropna()

X = data.drop(columns="target")
y = data["target"]

# 4. Chronological train-test split
split = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 5. Train and evaluate
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

pred = np.maximum(model.predict(X_test), 0)
baseline = np.maximum(X_test["avg7"].to_numpy(), 0)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
baseline_mae = mean_absolute_error(y_test, baseline)

print("\n--- MODEL EVALUATION ---")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Model MAE: Rs. {mae:.2f}")
print(f"Model RMSE: Rs. {rmse:.2f}")
print(f"7-day baseline MAE: Rs. {baseline_mae:.2f}")

# 6. Train final model on all available data
final_model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=3,
    random_state=42
)
final_model.fit(X, y)

# 7. Recursive future prediction
def forecast(history, trained_model, dates):
    history = history.copy()
    results = []

    for date in dates:
        row = pd.DataFrame([{
            "weekday": date.dayofweek,
            "day": date.day,
            "month": date.month,
            "day_year": date.dayofyear,
            "lag1": history.iloc[-1],
            "lag7": history.iloc[-7],
            "avg7": history.iloc[-7:].mean(),
            "avg30": history.iloc[-30:].mean()
        }])

        amount = max(float(trained_model.predict(row)[0]), 0)
        history.loc[date] = amount
        results.append(amount)

    return pd.Series(results, index=dates)

# 8. Predict tomorrow's expense
tomorrow = today + pd.Timedelta(days=1)
daily_prediction = forecast(
    daily, final_model, pd.DatetimeIndex([tomorrow])
).iloc[0]

# 9. Predict month-end and next-month expenses
month_start = today.replace(day=1)
month_end = today.replace(
    day=monthrange(today.year, today.month)[1]
)

next_start = month_start + pd.offsets.MonthBegin(1)
next_end = next_start.replace(
    day=monthrange(next_start.year, next_start.month)[1]
)

future_dates = pd.date_range(tomorrow, next_end, freq="D")
future = forecast(daily, final_model, future_dates)

actual_this_month = daily[
    (daily.index >= month_start) & (daily.index <= today)
].sum()

remaining_this_month = future[
    (future.index >= tomorrow) & (future.index <= month_end)
].sum()

next_month_total = future[
    (future.index >= next_start) & (future.index <= next_end)
].sum()

# 10. Save model and history for FastAPI
joblib.dump({
    "model": final_model,
    "daily_history": daily,
    "last_date": daily.index.max(),
    "features": list(X.columns),
    "mae": float(mae),
    "rmse": float(rmse),
    "baseline_mae": float(baseline_mae)
}, BASE / "expense_model.pkl")

print("\n--- EXPENSE PREDICTIONS ---")
print(f"Tomorrow's estimated expense: Rs. {daily_prediction:.2f}")
print(
    "Current month-end estimate: "
    f"Rs. {actual_this_month + remaining_this_month:.2f}"
)
print(f"Next month's estimated expense: Rs. {next_month_total:.2f}")
print("\nSaved trained model: expense_model.pkl")
