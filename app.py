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
    if not pd.isna(planeta
