# 🦠 CovidSense India
**Epidemic Spread Prediction Dashboard**
CodeCure Biohackathon — Track C

---

## 👋 About This Project

Hi! I am a 1st year student participating solo in the CodeCure Biohackathon.

This project predicts COVID-19 outbreak risk across India using real data from Johns Hopkins University. I built a machine learning model that looks at past case trends and predicts what the next 7 days might look like — for India as a whole and for individual states.

---

## 💡 The Problem I Am Solving

During COVID, it was hard to know:
- Is my state getting better or worse?
- How fast are cases rising?
- Which states need the most attention right now?

CovidSense answers all three questions in one dashboard.

---

## 🔍 What This Project Does

- Loads real COVID-19 data (1142 days of India data)
- Calculates daily new cases from cumulative totals
- Smooths out fake spikes caused by weekend reporting delays
- Predicts the next 7 days of cases using Machine Learning
- Scores every Indian state with a Severity Index (0-100)
- Shows everything on an interactive dashboard with graphs

---

## 🧠 Concepts I Learned and Used

### Log Transformation
Virus cases grow exponentially — they double very fast.
Normal math cannot predict exponential growth well.
So I convert the numbers using log before training the model,
then convert back after predicting.

```
Without log: 100 → 200 → 400 → 800  (hard to predict)
With log:    4.6 → 5.3 → 6.0 → 6.7  (easy to predict)
```

### 7-Day Smoothing
Hospitals do not report the same number of cases every day.
On weekends they report less, then catch up on Monday.
This creates fake spikes in the data.
I take a 7-day average to remove these fake spikes.

### Sliding Window of 14 Days
To predict tomorrow's cases, I look at the last 14 days.
This is called a sliding window.
14 days matches the COVID incubation period used by doctors.

### Severity Index
Instead of just showing case numbers, I combine 3 things:

```
Severity Score = (Cases x 50%) + (Deaths x 30%) + (Recovery Lag x 20%)
```

- Cases (50%) — How fast is it spreading?
- Deaths (30%) — How serious is this wave?
- Recovery Lag (20%) — Are hospitals keeping up?

This gives one number from 0 to 100 that tells the full story of a state.

---

## 🗂️ Project Files

```
CovidSense/
├── data/
│   ├── covid_data.csv       ← JHU COVID data (national)
│   ├── state_data.csv       ← India state-wise data
│   └── vaccine_data.csv     ← Vaccination data (OWID)
├── model.py                 ← Main prediction model
├── hotspot.py               ← State-wise risk scoring
├── app.py                   ← Interactive dashboard
├── requirements.txt         ← Libraries needed
└── README.md                ← This file
```

---

## 📊 Dashboard Graphs

1. India Risk Map — Every state colored Red / Orange / Green by risk
2. 7-Day Forecast — Past 30 days + next 7 days predicted
3. Severity Gauge — Speedometer showing 0-100 risk score
4. Wave Timeline — All 1142 days showing Wave 1, Wave 2, Wave 3
5. Vaccination vs Cases — How vaccines reduced case numbers
6. Top 5 Risk States — Quick view of highest risk states

---

## 📥 Data Sources

| Data | Source |
|---|---|
| COVID cases | Johns Hopkins University (JHU CSSE) |
| State-wise data | covid19india.org |
| Vaccination data | Our World in Data (OWID) |

---

## ⚙️ How to Run

Step 1 — Install libraries
```bash
pip install pandas numpy scikit-learn plotly dash
```

Step 2 — Test the prediction model
```bash
python model.py
```

Step 3 — Generate state risk scores
```bash
python hotspot.py
```

Step 4 — Launch the dashboard
```bash
python app.py
```
Open browser and go to: http://127.0.0.1:8050

---

## 🚀 Live Demo

Live link will be added after deployment on Render

---

## ⚠️ Honest Notes

- I am a 1st year student participating solo
- I used AI assistance to help write and debug parts of the code


---

## 👤 Participant

Solo | 1st Year Student
CodeCure Biohackathon | Track C — Epidemic Spread Prediction
