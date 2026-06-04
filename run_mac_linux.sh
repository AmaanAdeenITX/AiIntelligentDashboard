#!/bin/bash
echo ""
echo " PulseAI Food Intelligence Dashboard"
echo " ====================================="
echo ""

# Create venv
if [ ! -d "venv" ]; then
    echo " Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo " Installing dependencies..."
pip install -r requirements.txt -q

if [ ! -f "data/food_delivery_data.csv" ]; then
    echo " Generating dataset..."
    python generate_data.py
fi

if [ ! -f "models/reorder_model.pkl" ]; then
    echo " Training AI model..."
    python train_model.py
fi

echo ""
echo " Launching at http://localhost:8501"
streamlit run app.py --server.port 8501
