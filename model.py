"""
model.py — Core Prediction Engine
==================================
This is the BRAIN of CovidSense.
It loads India's COVID data, smooths it, trains a model,
and predicts the next 7 days of cases.

CONCEPT: We use Linear Regression with Log Transformation.
- Log transformation converts exponential growth into linear growth
- Linear Regression then fits a straight line to that log-scale data
- We convert back (expm1) to get real case numbers
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# ==============================================================
# FUNCTION 1: load_india_data()
# PURPOSE: Read the CSV and extract India's cumulative case numbers
# ==============================================================
def load_india_data():
    """
    The JHU CSV looks like this:
    Country/Region | 1/22/20 | 1/23/20 | 1/24/20 | ...
    India          |    0    |    0    |    1    | ...

    We grab all columns from index 4 onwards (those are date columns)
    and flatten them into a single list of numbers.
    """
    data = pd.read_csv("data/covid_data.csv")

    # Filter only India rows
    india = data[data["Country/Region"] == "India"]

    # Columns 0-3 are: Province, Country, Lat, Long
    # Columns 4 onwards are dates → these are our case numbers
    date_cols = data.columns[4:]

    # .values gives numpy array, .flatten() makes it 1D
    india_cases = india[date_cols].values.flatten()

    return india_cases  # Example: [0, 0, 1, 1, 3, 5, 10, ...]


# ==============================================================
# FUNCTION 2: calculate_daily_new_cases()
# PURPOSE: Convert cumulative totals into daily new cases
# ==============================================================
def calculate_daily_new_cases(total_cases):
    """
    The CSV gives CUMULATIVE cases (total so far).
    We need DAILY new cases (how many new today).

    Example:
    Cumulative: [100, 150, 200, 180]
    Daily:      [50,  50,  -20]  → but negative is impossible
    After max(0): [50, 50, 0]   → we floor at zero

    Then we apply 7-DAY ROLLING AVERAGE (smoothing):
    WHY? Because hospitals report unevenly.
    Saturday might show 5000 cases, Sunday 95000 (catching up).
    The 7-day average removes these fake spikes.
    """
    daily = []
    for i in range(1, len(total_cases)):
        new_cases = total_cases[i] - total_cases[i - 1]
        daily.append(max(0, new_cases))  # Never allow negative cases

    # Rolling mean with window=7
    # min_periods=1 means even if we have less than 7 days, still compute
    smoothed = pd.Series(daily).rolling(window=7, min_periods=1).mean().tolist()

    return [float(x) for x in smoothed]  # Clean floats, no np.float64 mess


# ==============================================================
# FUNCTION 3: train_model()
# PURPOSE: Train a Linear Regression model on the smoothed daily cases
# ==============================================================
def train_model(daily_cases, window=14):
    """
    SLIDING WINDOW concept:
    We look at 14 days at a time to predict the 15th day.

    Example with window=3 (simplified):
    Days: [10, 20, 30, 40, 50, 60]
    X (input) → y (what we want to predict)
    [10,20,30] → 40
    [20,30,40] → 50
    [30,40,50] → 60

    LOG TRANSFORMATION:
    np.log1p(x) = log(x + 1)
    Why +1? Because log(0) is undefined. Adding 1 is safe.
    This converts exponential growth patterns into linear ones.
    """
    X, y = [], []

    for i in range(window, len(daily_cases)):
        # Take 14 days before index i → transform to log scale
        X.append(np.log1p(daily_cases[i - window:i]))
        # The target is day i → also in log scale
        y.append(np.log1p(daily_cases[i]))

    model = LinearRegression()
    model.fit(np.array(X), np.array(y))

    return model


# ==============================================================
# FUNCTION 4: predict_next_7_days()
# PURPOSE: Use trained model to predict the next 7 days
# ==============================================================
def predict_next_7_days(model, daily_cases, window=14):
    """
    ITERATIVE PREDICTION:
    We predict day 1, then use that prediction as input for day 2, etc.

    Step 1: Take last 14 real days (in log scale)
    Step 2: Predict next day (in log scale)
    Step 3: Convert back using np.expm1() (reverse of log1p)
    Step 4: Apply 35% growth cap (realism check)
    Step 5: Append prediction to our window, repeat

    WHY 35% CAP?
    A virus cannot realistically grow more than 35% per day
    in a stable population. Without this cap, small errors
    in the model compound into unrealistic explosions.
    """
    predictions = []
    current = list(np.log1p(daily_cases[-window:]))  # Last 14 days in log scale

    for _ in range(7):
        # Predict next day in log scale
        pred_log = model.predict([current[-window:]])[0]

        # Convert back from log to real number
        pred = np.expm1(pred_log)

        # Apply realism cap: max 35% growth over previous day
        last_val = np.expm1(current[-1])
        pred = min(pred, last_val * 1.35)

        pred = max(0, pred)  # Never negative
        predictions.append(int(pred))

        # Add this prediction to window for next iteration
        current.append(np.log1p(pred))

    return predictions  # List of 7 integers


# ==============================================================
# FUNCTION 5: get_trend()
# PURPOSE: Convert % change number into human-readable trend string
# ==============================================================
def get_trend(change):
    """
    change is a percentage value.
    > 10%  → Rising
    < -10% → Falling
    else   → Stable

    We use 10% as threshold because small fluctuations
    (under 10%) are considered normal noise, not a real trend.
    """
    if change > 10:
        return f"Rising by {abs(change):.1f}%"
    elif change < -10:
        return f"Falling by {abs(change):.1f}%"
    else:
        return "Stable"


# ==============================================================
# FUNCTION 6: get_risk_level()
# PURPOSE: Classify the situation into High / Medium / Low risk
# ==============================================================
def get_risk_level(change):
    """
    > 50% change = High Risk (Rapid Acceleration)
    > 10% change = Medium Risk (Steady Increase)
    else         = Low Risk (Stable or Declining)

    Why 50% for High? Because a 50%+ surge in a week
    indicates an active wave beginning, not just noise.
    """
    if change > 50:
        return "High"
    elif change > 10:
        return "Medium"
    else:
        return "Low"


# ==============================================================
# MAIN EXECUTION
# Run this file directly to test the model
# ==============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  CovidSense — Core Model Test")
    print("=" * 50)

    # Step 1: Load data
    total = load_india_data()
    print(f"\n[1] Total data points loaded: {len(total)}")

    # Step 2: Calculate daily new cases
    daily = calculate_daily_new_cases(total)
    print(f"[2] Daily cases computed: {len(daily)} days")
    print(f"    Last 7 days (smoothed): {[int(x) for x in daily[-7:]]}")

    # Step 3: Train model
    model = train_model(daily)
    print(f"\n[3] Model trained successfully")
    print(f"    Algorithm: Linear Regression (Log-transformed)")
    print(f"    Window size: 14 days")

    # Step 4: Predict
    predictions = predict_next_7_days(model, daily)
    print(f"\n[4] 7-Day Predictions: {predictions}")

    # Step 5: Analyze
    recent_avg = sum(daily[-7:]) / 7
    future_avg = sum(predictions) / 7
    change = ((future_avg - recent_avg) / max(recent_avg, 1)) * 100
    trend = get_trend(change)
    risk = get_risk_level(change)

    print(f"\n[5] Analysis:")
    print(f"    Recent Avg  : {recent_avg:.1f} cases/day")
    print(f"    Predicted   : {future_avg:.1f} cases/day")
    print(f"    Change      : {change:.1f}%")
    print(f"    Trend       : {trend}")
    print(f"    Risk Level  : {risk}")
    print("=" * 50)