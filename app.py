# app.py
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle
model = joblib.load("wine_quality_model5.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            features = [
                float(request.form.get("fixed_acidity")),
                float(request.form.get("volatile_acidity")),
                float(request.form.get("citric_acid")),
                float(request.form.get("residual_sugar")),
                float(request.form.get("chlorides")),
                float(request.form.get("free_sulfur_dioxide")),
                float(request.form.get("total_sulfur_dioxide")),
                float(request.form.get("density")),
                float(request.form.get("pH")),
                float(request.form.get("sulphates")),
                float(request.form.get("alcohol")),
            ]
            prediction = model.predict([features])[0]
            return render_template("index.html", prediction=round(prediction, 2))
        except Exception as e:
            return render_template("index.html", prediction=f"Erreur: {e}")
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

