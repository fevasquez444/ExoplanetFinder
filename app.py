import io
import base64
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ✅ Evita errores de entorno gráfico
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# === CONFIGURACIÓN FLASK ===
app = Flask(__name__, static_url_path='', static_folder='.', template_folder='.')
CORS(app)


# === CARGA DE DATASET (NASA LIVE + RESPALDO LOCAL) ===
def cargar_dataset():
    nasa_url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+top+2000+pl_name,disc_year,discoverymethod,"
        "pl_rade,pl_bmasse,default_flag,disposition+from+ps&format=csv"
    )
    try:
        print("🌍 Descargando dataset NASA en vivo...")
        df = pd.read_csv(nasa_url, comment="#", low_memory=False)
        print(f"✅ Dataset NASA cargado correctamente ({len(df)} registros).")
    except Exception as e:
        print(f"⚠️ Error al descargar dataset NASA: {e}")
        if os.path.exists("exoplanets_ps.csv"):
            print("📂 Cargando dataset de respaldo local...")
            df = pd.read_csv("exoplanets_ps.csv")
        else:
            raise RuntimeError("❌ No se pudo cargar ni el dataset NASA ni el respaldo local.")
    return df


# === ENTRENAMIENTO DEL MODELO ===
def train_model():
    df = cargar_dataset()

    # Seleccionar columnas clave
    cols = ["pl_name", "disc_year", "discoverymethod", "pl_rade", "pl_bmasse", "default_flag", "disposition"]
    for col in cols:
        if col not in df.columns:
            df[col] = None

    # Columnas numéricas simples
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    num_cols = num_cols[:2] if len(num_cols) >= 2 else ["pl_rade", "pl_bmasse"]

    df = df.dropna(subset=num_cols)

    # Etiquetas: disposición NASA
    y = df["disposition"].fillna("UNKNOWN").astype(str)
    X = df[num_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)

    print(f"✅ Modelo entrenado. Precisión: {accuracy_score(y_test, model.predict(X_test)):.2f}")

    return model, num_cols, df


# === VISUALIZACIONES ===
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


# === ENTRENAR MODELO GLOBAL ===
model, features, dataset = train_model()


# === RUTAS ===
@app.route('/')
def home():
    return app.send_static_file('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        val1 = data.get("val1", 0)
        val2 = data.get("val2", 0)
        values = [[val1, val2]]
        pred = model.predict(values)[0]

        # Planeta aleatorio
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
        return jsonify({"error": str(e)}), 500


# === MAIN ===
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
