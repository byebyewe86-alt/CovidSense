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

SEVERITY INDEX FORMULA:
Severity = (Case Score × 0.5) + (Death Score × 0.3) + (Recovery Lag × 0.2)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from model import calculate_daily_new_cases, train_model, predict_next_7_days


# ==============================================================
# FUNCTION 1: load_state_data()
# FIXED: Your CSV has state codes as columns not rows
# ==============================================================
def load_state_data(filepath="data/state_data.csv"):
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date_YMD'])
        df = df.sort_values('Date')
        return df
    except FileNotFoundError:
        print("ERROR: state_data.csv not found")
        return None


# ==============================================================
# FUNCTION 2: get_state_series()
# FIXED: Your CSV has state codes as columns (MH, DL, KA...)
# ==============================================================
def get_state_series(df, state_code):
    if state_code not in df.columns:
        return None, None, None

    confirmed = df[df['Status'] == 'Confirmed'][state_code].values.tolist()
    recovered = df[df['Status'] == 'Recovered'][state_code].values.tolist()
    deceased  = df[df['Status'] == 'Deceased'][state_code].values.tolist()

    return confirmed, recovered, deceased


# ==============================================================
# STATE CODE TO NAME MAPPING
# So we show "Maharashtra" instead of "MH"
# ==============================================================
STATE_NAMES = {
    'AN': 'Andaman & Nicobar',
    'AP': 'Andhra Pradesh',
    'AR': 'Arunachal Pradesh',
    'AS': 'Assam',
    'BR': 'Bihar',
    'CH': 'Chandigarh',
    'CT': 'Chhattisgarh',
    'DN': 'Dadra & Nagar Haveli',
    'DD': 'Daman & Diu',
    'DL': 'Delhi',
    'GA': 'Goa',
    'GJ': 'Gujarat',
    'HR': 'Haryana',
    'HP': 'Himachal Pradesh',
    'JK': 'Jammu & Kashmir',
    'JH': 'Jharkhand',
    'KA': 'Karnataka',
    'KL': 'Kerala',
    'LA': 'Ladakh',
    'LD': 'Lakshadweep',
    'MP': 'Madhya Pradesh',
    'MH': 'Maharashtra',
    'MN': 'Manipur',
    'ML': 'Meghalaya',
    'MZ': 'Mizoram',
    'NL': 'Nagaland',
    'OR': 'Odisha',
    'PY': 'Puducherry',
    'PB': 'Punjab',
    'RJ': 'Rajasthan',
    'SK': 'Sikkim',
    'TN': 'Tamil Nadu',
    'TG': 'Telangana',
    'TR': 'Tripura',
    'UP': 'Uttar Pradesh',
    'UT': 'Uttarakhand',
    'WB': 'West Bengal'
}


# ==============================================================
# FUNCTION 3: compute_severity_index()
# PURPOSE: Combine cases, deaths, recovery into one 0-100 score
# ==============================================================
def compute_severity_index(daily_cases, daily_deaths,
                            daily_recovered, predictions):
    # Case Score
    recent_case_avg = sum(daily_cases[-7:]) / 7 \
        if len(daily_cases) >= 7 else 1
    future_case_avg = sum(predictions) / 7 \
        if predictions else 1

    case_change = ((future_case_avg - recent_case_avg) /
                    max(recent_case_avg, 1)) * 100
    case_score = min(100, max(0, 50 + case_change))

    # Death Rate Score
    if len(daily_deaths) >= 7 and sum(daily_cases[-7:]) > 0:
        recent_deaths = sum(daily_deaths[-7:])
        recent_cases = sum(daily_cases[-7:])
        death_rate = (recent_deaths /
                       max(recent_cases, 1)) * 100
        death_score = min(100, death_rate * 10)
    else:
        death_score = 0

    # Recovery Lag Score
    if len(daily_recovered) >= 7:
        recent_recovered = sum(daily_recovered[-7:])
        recent_cases_count = sum(daily_cases[-7:])
        lag = recent_cases_count - recent_recovered
        lag_score = min(100, max(0,
            (lag / max(recent_cases_count, 1)) * 100))
    else:
        lag_score = 0

    # Weighted Severity Index
    severity = ((case_score * 0.5) +
                (death_score * 0.3) +
                (lag_score * 0.2))
    severity = round(min(100, max(0, severity)), 1)

    return severity, case_score, death_score, lag_score


# ==============================================================
# FUNCTION 4: classify_severity()
# ==============================================================
def classify_severity(severity_score):
    if severity_score >= 70:
        return "High", "#FF4444"
    elif severity_score >= 40:
        return "Medium", "#FFA500"
    else:
        return "Low", "#44BB44"


# ==============================================================
# FUNCTION 5: analyze_all_states()
# FIXED: Uses state codes as columns
# ==============================================================
def analyze_all_states(filepath="data/state_data.csv"):
    df = load_state_data(filepath)
    if df is None:
        return None

    # Skip non-state columns
    skip_cols = ['Date', 'Date_YMD', 'Status', 'TT', 'UN']
    state_codes = [
        col for col in df.columns
        if col not in skip_cols
    ]

    results = []

    for code in state_codes:
        confirmed, recovered, deceased = \
            get_state_series(df, code)

        if confirmed is None:
            continue
        if len(confirmed) < 20:
            continue

        daily_cases = calculate_daily_new_cases(confirmed)
        daily_deaths = calculate_daily_new_cases(deceased)
        daily_recovered = calculate_daily_new_cases(recovered)

        if sum(daily_cases[-7:]) == 0:
            continue

        try:
            model = train_model(daily_cases)
            predictions = predict_next_7_days(
                model, daily_cases
            )

            severity, cs, ds, ls = compute_severity_index(
                daily_cases, daily_deaths,
                daily_recovered, predictions
            )
            risk_label, color = classify_severity(severity)

            recent_avg = sum(daily_cases[-7:]) / 7
            future_avg = sum(predictions) / 7
            change = ((future_avg - recent_avg) /
                       max(recent_avg, 1)) * 100

            # Get full state name
            state_name = STATE_NAMES.get(code, code)

            results.append({
                "State": state_name,
                "Code": code,
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
            print(f"Skipping {code}: {e}")
            continue

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(
            "Severity", ascending=False
        )

    return result_df


# ==============================================================
# MAIN EXECUTION
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
            print(f"  {row['State']:<25} | "
                  f"Severity: {row['Severity']:>5} | "
                  f"Risk: {row['Risk']:<6} | "
                  f"Change: {row['Change_Pct']:>+6.1f}%")

        print("\nBOTTOM 5 SAFEST STATES:")
        print("-" * 60)
        bottom5 = results.tail(5)
        for _, row in bottom5.iterrows():
            print(f"  {row['State']:<25} | "
                  f"Severity: {row['Severity']:>5} | "
                  f"Risk: {row['Risk']:<6}")

        results.to_csv("data/hotspot_results.csv",
                       index=False)
        print("\nResults saved to data/hotspot_results.csv")

    else:
        print("No results. Check state_data.csv")




