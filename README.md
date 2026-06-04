# PulseAI — Food Intelligence Dashboard

An AI-powered decision dashboard built on 15,000 food delivery orders.
Predicts customer reorder behaviour using Gradient Boosting (75% accuracy).

---

## Quick Start (PyCharm)

### Step 1 — Open the project
Open the `food_dashboard` folder in PyCharm.

### Step 2 — Create a virtual environment
In PyCharm terminal (bottom bar):
```
python -m venv venv
```
Then activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### Step 3 — Install dependencies
```
pip install -r requirements.txt
```

### Step 4 — Generate data + train model
```
python generate_data.py
python train_model.py
```

### Step 5 — Launch the dashboard
```
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## One-click scripts (no PyCharm needed)
- **Windows:** Double-click `run_windows.bat`
- **Mac/Linux:** Run `bash run_mac_linux.sh`

---

## Project structure
```
food_dashboard/
├── app.py                  # Main Streamlit dashboard
├── generate_data.py        # Synthetic dataset generator
├── train_model.py          # AI model training
├── requirements.txt        # Python dependencies
├── run_windows.bat         # Windows launcher
├── run_mac_linux.sh        # Mac/Linux launcher
├── data/
│   └── food_delivery_data.csv
└── models/
    ├── reorder_model.pkl
    ├── le_cuisine.pkl
    ├── le_time.pkl
    ├── le_payment.pkl
    ├── le_city.pkl
    └── model_meta.json
```

---

## Dashboard features
| Tab | What it shows |
|-----|---------------|
| Overview | Revenue trends, cuisine share, top restaurants, payment methods |
| City & Cuisine | Market heatmap, AOV by cuisine, reorder rates |
| Operations | Delivery time analysis, peak hours, promo impact |
| AI Predictor | Live customer reorder prediction with probability score |

---

## Tech stack
- **Streamlit** — dashboard framework
- **Plotly** — interactive charts
- **Scikit-learn** — Gradient Boosting classifier
- **Pandas / NumPy** — data processing
