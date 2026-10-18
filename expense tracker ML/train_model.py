
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from calendar import monthrange
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load and clean dataset
BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "personal_finance_dataset.csv")
df.columns = df.columns.str.strip().str.lower()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df = df.dropna(subset=["date", "amount"])
df["income_or_expense"] = df["income_or_expense"].astype(str).str.strip().str.lower()
df = df[df["income_or_expense"] == "expense"]

# 2. Daily expense totals
daily = df.groupby(df["date"].dt.normalize())["amount"].sum().sort_index()
daily = daily.asfreq("D", fill_value=0).clip(lower=0)
today = pd.Timestamp.today().normalize()
daily = daily[daily.index <= today]
print("Today's date:", today.date())
print("Last historical date:", daily.index.max().date())
if len(daily) < 60:
    raise ValueError("At least 60 days of historical data are required.")

# 3. Create ML features
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
X, y = data.drop(columns="target"), data["target"]

# 4. Chronological train-test split
split = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 5. Train and evaluate model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)
pred = np.maximum(model.predict(X_test), 0)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred) if len(y_test) > 1 else float("nan")

print("\n--- MODEL EVALUATION ---")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print(f"MAE: Rs. {mae:.2f}")
print(f"RMSE: Rs. {rmse:.2f}")

# Compare ML model with a simple 7-day average baseline
baseline_pred = np.maximum(X_test["avg7"].to_numpy(), 0)

baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

print("\n--- BASELINE COMPARISON ---")
print(f"ML Model MAE: Rs. {mae:.2f}")
print(f"7-day Average MAE: Rs. {baseline_mae:.2f}")
print(f"ML Model RMSE: Rs. {rmse:.2f}")
print(f"7-day Average RMSE: Rs. {baseline_rmse:.2f}")


# 6. Train final model on all available data
final_model = RandomForestRegressor(
    n_estimators=300, random_state=42, min_samples_leaf=3
)
final_model.fit(X, y)

# Save model, history and evaluation metrics
joblib.dump({
    "model": final_model,
    "daily_history": daily,
    "last_date": daily.index.max(),
    "features": list(X.columns),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2)
}, BASE / "expense_model.pkl")

print("\nModel saved as expense_model.pkl")

# 7. Predict future daily expenses
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

# 8. Current month-end and next-month predictions
month_start = today.replace(day=1)
month_end = month_start.replace(
    day=monthrange(today.year, today.month)[1]
)
next_start = month_start + pd.offsets.MonthBegin(1)
next_end = next_start.replace(
    day=monthrange(next_start.year, next_start.month)[1]
)

# Start after the last historical date, not before it
start = max(today, daily.index.max()) + pd.Timedelta(days=1)
end = next_end

if start <= end:
    dates = pd.date_range(start, end, freq="D")
    future = forecast(daily, final_model, dates)

    actual = daily[
        (daily.index >= month_start) & (daily.index <= today)
    ].sum()

    current_pred = future[
        (future.index.month == today.month) &
        (future.index.year == today.year)
    ].sum()

    next_pred = future[
        (future.index.month == next_start.month) &
        (future.index.year == next_start.year)
    ].sum()

    print("\n--- EXPENSE PREDICTIONS ---")
    print(f"Current month-end estimate: Rs. {actual + current_pred:.2f}")
    print(f"Next month estimate: Rs. {next_pred:.2f}")
else:
    print("No future dates available for prediction.")

