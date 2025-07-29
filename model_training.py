# model_training.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Charger les données
df = pd.read_csv("winequality-red.csv")

# Caractéristiques (X) et cible (y)
X = df.drop("quality", axis=1)
y = df["quality"]

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sauvegarder le modèle
joblib.dump(model, "wine_quality_model5.pkl")
print("✅ Modèle entraîné et sauvegardé !")
