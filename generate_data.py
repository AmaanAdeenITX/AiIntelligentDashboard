import pandas as pd
import numpy as np
import os

np.random.seed(42)

CITIES = ["Hyderabad", "Bangalore", "Mumbai", "Delhi", "Chennai", "Pune", "Kolkata", "Ahmedabad"]
CUISINES = ["Biryani", "Pizza", "Burger", "Chinese", "South Indian", "North Indian", "Sushi", "Desserts", "Healthy", "Street Food"]
RESTAURANTS = {
    "Biryani": ["Paradise Biryani", "Bawarchi", "Hyderabad House", "Dum Pukht", "Royal Biryani"],
    "Pizza": ["Domino's", "Pizza Hut", "La Pino'z", "Ovenstory", "Smoky Joe's"],
    "Burger": ["McDonald's", "Burger King", "Burger Singh", "The Burger Club", "Carl's Jr"],
    "Chinese": ["Mainland China", "Chowman", "Chinese Wok", "The Oriental", "Dragon House"],
    "South Indian": ["Saravana Bhavan", "Udupi Palace", "MTR", "Murugan Idli Shop", "Dakshin"],
    "North Indian": ["Moti Mahal", "Barbeque Nation", "Punjab Grill", "Pind Balluchi", "Saffron"],
    "Sushi": ["Edo Japan", "Sakura Sushi", "Wasabi", "Nobu", "Sushi Garden"],
    "Desserts": ["Baskin Robbins", "Keventers", "The Dessert Factory", "Naturals", "Wow! Momo"],
    "Healthy": ["Salad Days", "EatFit", "Freshly", "Green Bowl", "Nourish"],
    "Street Food": ["Haldiram's", "Bikaner", "Bikanervala", "Chaayos", "Oven Story"]
}

n = 15000

cuisines = np.random.choice(CUISINES, n)
cities = np.random.choice(CITIES, n, p=[0.22, 0.18, 0.17, 0.15, 0.1, 0.08, 0.06, 0.04])

restaurants = [np.random.choice(RESTAURANTS[c]) for c in cuisines]

base_order_value = {
    "Biryani": 280, "Pizza": 380, "Burger": 220, "Chinese": 320,
    "South Indian": 180, "North Indian": 350, "Sushi": 520,
    "Desserts": 150, "Healthy": 280, "Street Food": 120
}

order_value = np.array([base_order_value[c] for c in cuisines])
order_value = order_value * np.random.uniform(0.7, 1.8, n)
order_value = np.round(order_value, 0)

city_delivery = {"Hyderabad": 28, "Bangalore": 35, "Mumbai": 40, "Delhi": 38,
                 "Chennai": 30, "Pune": 32, "Kolkata": 33, "Ahmedabad": 29}

delivery_time = np.array([city_delivery[c] for c in cities]) + np.random.normal(0, 8, n)
delivery_time = np.clip(delivery_time, 10, 90).astype(int)

ratings = np.random.choice([3.0, 3.5, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0],
                            n, p=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.13, 0.13, 0.12, 0.09, 0.06, 0.03, 0.02])

is_weekend = np.random.choice([0, 1], n, p=[0.57, 0.43])
time_of_day = np.random.choice(["Morning", "Lunch", "Evening", "Late Night"], n, p=[0.10, 0.35, 0.40, 0.15])
promo_used = np.random.choice([0, 1], n, p=[0.55, 0.45])
payment_method = np.random.choice(["UPI", "Card", "Cash", "Wallet"], n, p=[0.48, 0.25, 0.15, 0.12])
num_items = np.random.randint(1, 8, n)

months = np.random.choice(range(1, 13), n)
days = np.random.randint(1, 28, n)
hours = np.array([{"Morning": np.random.randint(7, 11), "Lunch": np.random.randint(11, 15),
                   "Evening": np.random.randint(17, 22), "Late Night": np.random.randint(22, 24)}[t] for t in time_of_day])

order_date = pd.to_datetime({"year": 2024, "month": months, "day": days})

# Reorder probability (target variable) - based on multiple factors
reorder_score = (
    (ratings - 3.0) * 0.4 +
    (order_value / 500) * 0.15 +
    (1 - delivery_time / 90) * 0.25 +
    promo_used * 0.1 +
    is_weekend * 0.05 +
    np.random.normal(0, 0.15, n)
)
reorder = (reorder_score > np.median(reorder_score)).astype(int)

df = pd.DataFrame({
    "order_id": [f"ORD{str(i).zfill(6)}" for i in range(1, n+1)],
    "restaurant": restaurants,
    "cuisine": cuisines,
    "city": cities,
    "order_value": order_value,
    "delivery_time_mins": delivery_time,
    "rating": ratings,
    "num_items": num_items,
    "is_weekend": is_weekend,
    "time_of_day": time_of_day,
    "promo_used": promo_used,
    "payment_method": payment_method,
    "order_date": order_date,
    "month": months,
    "will_reorder": reorder
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/food_delivery_data.csv", index=False)
print(f"Dataset generated: {len(df)} rows")
print(df.head())
print(df.dtypes)
