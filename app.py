import io
import base64
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__, static_url_path='', static_folder='.', template_folder='.')
CORS(app)

# ==== CARGA DIRECTA DEL DATASET NASA ====
def cargar_dataset():
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+top+2000+pl_name,disc_year,discoverymethod,pl_rade,pl_bmasse,default_flag,disposition+from+ps&format=csv"
    print("🌍 Descargando dataset NASA en vivo...")
    df = pd.read_csv(url, comment='#', low_memory=False)
    print(f"✅ Dataset cargado con {len(df)} registros reales.")
    return df

# ==== ENTRENAR MODELO UNA VEZ AL INICIAR ====
def train_model():
    df = cargar_dataset()

    cols_interes = ["pl_name", "disc_year", "discoverymethod", "pl_rade", "pl_bmasse", "default_flag", "disposition"]
    for col in cols_interes:
        if col not in df.columns:
            df[col] = None

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_cols = numeric_cols[:2]
    df = df.dropna(subset=numeric_cols)

    y = df["disposition"].fillna("UNKNOWN").astype(str)
    X = df[numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)

    print("✅ Modelo entrenado, precisión:", accuracy_score(y_test, model.predict(X_test)))
    return model, numeric_cols, df

# ==== VISUALIZACIONES ====
def grafico_histograma(df, cols, planeta):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[cols[0]], bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    if not pd.isna(planeta[cols[0]]):
        ax.axvline(planeta[cols[0]], color="red", linestyle="--", linewidth=2, label="Planeta detectado")
    ax.set_xlabel(cols[0])
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución de {cols[0]}")
    ax.legend()
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode("utf-8")

def grafico_dispersion(df, cols, planeta):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df[cols[0]], df[cols[1]], c="blue", alpha=0.4, label="Exoplanetas")
    if not pd.isna(planeta[cols[0]]) and not pd.isna(planeta[cols[1]]):
        ax.scatter(planeta[cols[0]], planeta[cols[1]], c="red", s=150, marker="*", edgecolors="yellow", label="Detectado")
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_title(f"Dispersión {cols[0]} vs {cols[1]}")
    ax.legend()
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode("utf-8")

# ==== ENTRENAMIENTO GLOBAL ====
model, features, dataset = train_model()

# ==== RUTA PRINCIPAL ====
@app.route('/')
def home():
    return app.send_static_file('index.html')

# ==== API ====
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        values = [[data.get('val1', 0), data.get('val2', 0)]]
        pred = model.predict(values)[0]

        random_planet = dataset.sample(1).iloc[0]

        histograma_b64 = grafico_histograma(dataset, features, random_planet)
        dispersion_b64 = grafico_dispersion(dataset, features, random_planet)

        return jsonify({
            "features": features,
            "input": values,
            "prediction": str(pred),
            "pred_label": str(pred),
            "planet_name": str(random_planet["pl_name"]),
            "disc_year": int(random_planet["disc_year"]) if not pd.isna(random_planet["disc_year"]) else None,
            "method": str(random_planet["discoverymethod"]),
            "radius": float(random_planet["pl_rade"]) if not pd.isna(random_planet["pl_rade"]) else None,
            "mass": float(random_planet["pl_bmasse"]) if not pd.isna(random_planet["pl_bmasse"]) else None,
            "disposition": str(random_planet["disposition"]) if "disposition" in random_planet else "UNKNOWN",
            "histograma": histograma_b64,
            "dispersion": dispersion_b64
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ==== EJECUCIÓN PARA RENDER ====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
