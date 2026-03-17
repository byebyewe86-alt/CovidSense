"""
hotspot.py — State-wise Hotspot Detection Engine
=================================================
This file extends model.py to work on ALL Indian states.

CONCEPT:
Instead of predicting for India as a whole,
we loop through each state and compute:
1. 7-day prediction
2. Risk level
3. Severity Index (cases + deaths + recovery combined)

The Severity Index is our unique contribution.
It tells a richer story than cases alone.

SEVERITY INDEX FORMULA:
Severity = (Case Score × 0.5) + (Death Score × 0.3) + (Recovery Lag × 0.2)

WHY THESE WEIGHTS?
- Cases (50%): Primary driver of future spread
- Deaths (30%): Indicates how lethal the current wave is
- Recovery Lag (20%): Proxy for healthcare system stress
  If recoveries are slow, hospitals are overwhelmed
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from model import calculate_daily_new_cases, train_model, predict_next_7_days


# ==============================================================
# FUNCTION 1: load_state_data()
# PURPOSE: Load state-wise COVID data for India
# NOTE: This uses a different dataset than model.py
#       The JHU dataset is country-level only.
#       We use covid19india archived data for state-level.
# ==============================================================
def load_state_data(filepath="data/state_data.csv"):
    """
    Expected CSV format:
    Date       | State         | Confirmed | Recovered | Deceased
    2020-03-15 | Maharashtra   | 10        | 0         | 0
    2020-03-15 | Kerala        | 22        | 3         | 0
    ...

    We group by State and sort by Date to get time series per state.
    """
    try:
        df = pd.read_csv(filepath)

        # Ensure Date column is datetime type for proper sorting
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        return df
    except FileNotFoundError:
        print("ERROR: state_data.csv not found in data/ folder")
        print("Download from: https://api.covid19india.org/csv/latest/state_wise_daily.csv")
        return None


# ==============================================================
# FUNCTION 2: get_state_series()
# PURPOSE: Extract time series for one specific state
# ==============================================================
def get_state_series(df, state_name):
    """
    Filters dataframe for one state and returns
    arrays for confirmed, recovered, deceased cases.

    Example:
    state_name = "Maharashtra"
    Returns: (confirmed_array, recovered_array, deceased_array)
    Each array is a list of daily cumulative values.
    """
    state_df = df[df['State'] == state_name].copy()

    if state_df.empty:
        return None, None, None

    confirmed = state_df['Confirmed'].values.tolist()
    recovered = state_df['Recovered'].values.tolist()
    deceased = state_df['Deceased'].values.tolist()

    return confirmed, recovered, deceased


# ==============================================================
# FUNCTION 3: compute_severity_index()
# PURPOSE: Combine cases, deaths, recovery into one 0-100 score
# ==============================================================
def compute_severity_index(daily_cases, daily_deaths, daily_recovered, predictions):
    """
    This is our UNIQUE CONTRIBUTION.
    Instead of showing 3 separate numbers, we combine them.

    STEP 1: Case Score
    How much are cases predicted to grow?
    recent_avg vs future_avg → percentage change → normalized to 0-100

    STEP 2: Death Rate Score
    What percentage of cases are dying?
    death_rate = deaths / max(cases, 1)
    Normalized to 0-100

    STEP 3: Recovery Lag Score
    Are recoveries keeping up with cases?
    lag = cases - recovered (in recent period)
    Higher lag = healthcare is overwhelmed

    FINAL: Weighted average of all three
    """

    # --- Case Score (0-100) ---
    recent_case_avg = sum(daily_cases[-7:]) / 7 if len(daily_cases) >= 7 else 1
    future_case_avg = sum(predictions) / 7 if predictions else 1

    # % change in cases, clamped to 0-100 range
    case_change = ((future_case_avg - recent_case_avg) / max(recent_case_avg, 1)) * 100
    case_score = min(100, max(0, 50 + case_change))  # 50 = neutral baseline

    # --- Death Rate Score (0-100) ---
    if len(daily_deaths) >= 7 and sum(daily_cases[-7:]) > 0:
        recent_deaths = sum(daily_deaths[-7:])
        recent_cases = sum(daily_cases[-7:])
        death_rate = (recent_deaths / max(recent_cases, 1)) * 100  # as percentage
        death_score = min(100, death_rate * 10)  # scale up: 10% death rate = score of 100
    else:
        death_score = 0

    # --- Recovery Lag Score (0-100) ---
    if len(daily_recovered) >= 7:
        recent_recovered = sum(daily_recovered[-7:])
        recent_cases_count = sum(daily_cases[-7:])
        # If recoveries < cases, healthcare is stressed
        lag = recent_cases_count - recent_recovered
        lag_score = min(100, max(0, (lag / max(recent_cases_count, 1)) * 100))
    else:
        lag_score = 0

    # --- Weighted Severity Index ---
    severity = (case_score * 0.5) + (death_score * 0.3) + (lag_score * 0.2)
    severity = round(min(100, max(0, severity)), 1)

    return severity, case_score, death_score, lag_score


# ==============================================================
# FUNCTION 4: classify_severity()
# PURPOSE: Convert severity number to color/label
# ==============================================================
def classify_severity(severity_score):
    """
    0-40   → Low    (Green)  — Safe, monitor normally
    40-70  → Medium (Yellow) — Watch closely
    70-100 → High   (Red)    — Action needed

    These thresholds come from epidemiological practice:
    Below 40 = normal background fluctuation
    40-70 = emerging concern
    Above 70 = active outbreak
    """
    if severity_score >= 70:
        return "High", "#FF4444"    # Red
    elif severity_score >= 40:
        return "Medium", "#FFA500"  # Orange
    else:
        return "Low", "#44BB44"     # Green


# ==============================================================
# FUNCTION 5: analyze_all_states()
# PURPOSE: Run the full pipeline for every Indian state
# ==============================================================
def analyze_all_states(filepath="data/state_data.csv"):
    """
    This is the MAIN function of hotspot.py.
    It loops through all states and returns a summary dataframe.

    Returns a dataframe with columns:
    State | Predictions | Severity | Risk | Color | Change%
    """
    df = load_state_data(filepath)
    if df is None:
        return None

    # Get list of all unique states
    states = df['State'].unique()
    results = []

    for state in states:
        # Skip total/summary rows if present
        if state in ['Total', 'India', 'State Unassigned']:
            continue

        # Get time series for this state
        confirmed, recovered, deceased = get_state_series(df, state)

        if confirmed is None or len(confirmed) < 20:
            continue  # Skip states with insufficient data

        # Calculate daily new cases (same as model.py)
        daily_cases = calculate_daily_new_cases(confirmed)
        daily_deaths = calculate_daily_new_cases(deceased)
        daily_recovered = calculate_daily_new_cases(recovered)

        # Skip if all zeros (state had no cases)
        if sum(daily_cases[-7:]) == 0:
            continue

        try:
            # Train model and predict (same pipeline as model.py)
            model = train_model(daily_cases)
            predictions = predict_next_7_days(model, daily_cases)

            # Compute our unique Severity Index
            severity, cs, ds, ls = compute_severity_index(
                daily_cases, daily_deaths, daily_recovered, predictions
            )
            risk_label, color = classify_severity(severity)

            # Calculate % change for display
            recent_avg = sum(daily_cases[-7:]) / 7
            future_avg = sum(predictions) / 7
            change = ((future_avg - recent_avg) / max(recent_avg, 1)) * 100

            results.append({
                "State": state,
                "Recent_Avg": round(recent_avg, 1),
                "Predictions": predictions,
                "Future_Avg": round(future_avg, 1),
                "Change_Pct": round(change, 1),
                "Severity": severity,
                "Case_Score": round(cs, 1),
                "Death_Score": round(ds, 1),
                "Recovery_Lag": round(ls, 1),
                "Risk": risk_label,
                "Color": color
            })

        except Exception as e:
            print(f"  Skipping {state}: {e}")
            continue

    # Convert to DataFrame and sort by Severity (highest first)
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("Severity", ascending=False)

    return result_df


# ==============================================================
# MAIN EXECUTION — Test hotspot.py directly
# ==============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  CovidSense — State Hotspot Analysis")
    print("=" * 60)

    results = analyze_all_states()

    if results is not None and not results.empty:
        print(f"\nAnalyzed {len(results)} Indian states\n")
        print("TOP 10 HIGH RISK STATES:")
        print("-" * 60)

        top10 = results.head(10)
        for _, row in top10.iterrows():
            print(f"  {row['State']:<20} | Severity: {row['Severity']:>5} "
                  f"| Risk: {row['Risk']:<6} | Change: {row['Change_Pct']:>+6.1f}%")

        print("\nBOTTOM 5 SAFEST STATES:")
        print("-" * 60)
        bottom5 = results.tail(5)
        for _, row in bottom5.iterrows():
            print(f"  {row['State']:<20} | Severity: {row['Severity']:>5} "
                  f"| Risk: {row['Risk']:<6}")

        # Save results for dashboard to use
        results.to_csv("data/hotspot_results.csv", index=False)
        print("\nResults saved to data/hotspot_results.csv")
    else:
        print("No results. Check your state_data.csv file.")