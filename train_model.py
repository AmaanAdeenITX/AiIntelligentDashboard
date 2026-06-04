import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/food_delivery_data.csv")

le_cuisine = LabelEncoder()
le_time = LabelEncoder()
le_payment = LabelEncoder()
le_city = LabelEncoder()

df["cuisine_enc"] = le_cuisine.fit_transform(df["cuisine"])
df["time_enc"] = le_time.fit_transform(df["time_of_day"])
df["payment_enc"] = le_payment.fit_transform(df["payment_method"])
df["city_enc"] = le_city.fit_transform(df["city"])

features = ["order_value", "delivery_time_mins", "rating", "num_items",
            "is_weekend", "promo_used", "cuisine_enc", "time_enc",
            "payment_enc", "city_enc", "month"]

X = df[features]
y = df["will_reorder"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

importances = dict(zip(features, model.feature_importances_.tolist()))

joblib.dump(model, "models/reorder_model.pkl")
joblib.dump(le_cuisine, "models/le_cuisine.pkl")
joblib.dump(le_time, "models/le_time.pkl")
joblib.dump(le_payment, "models/le_payment.pkl")
joblib.dump(le_city, "models/le_city.pkl")

meta = {
    "accuracy": round(acc * 100, 2),
    "features": features,
    "cuisine_classes": le_cuisine.classes_.tolist(),
    "time_classes": le_time.classes_.tolist(),
    "payment_classes": le_payment.classes_.tolist(),
    "city_classes": le_city.classes_.tolist(),
    "feature_importances": importances
}
with open("models/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Model and encoders saved successfully.")
