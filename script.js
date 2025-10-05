document.getElementById("detect-btn").addEventListener("click", async () => {
  const planetContainer = document.getElementById("planet-container");
  const dataContainer = document.getElementById("data-container");
  const histogramaImg = document.getElementById("histograma");
  const dispersionImg = document.getElementById("dispersion");

  // Limpiar antes de generar otro
  planetContainer.innerHTML = "";
  dataContainer.innerHTML = "<p>🔄 Consultando IA en vivo...</p>";
  histogramaImg.style.display = "none";
  dispersionImg.style.display = "none";

  try {
    // 📡 Llamar al backend Flask (predicción IA + gráficos dinámicos)
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        val1: Math.random() * 10,
        val2: Math.random() * 5
      })
    });

    if (!response.ok) throw new Error("Error en la respuesta del servidor");
    const result = await response.json();

    // 🎨 Generar planeta visual
    const colors = ["#ff4a8d", "#4afcff", "#ffe14a", "#7d4aff", "#4aff72"];
    const color = colors[Math.floor(Math.random() * colors.length)];

    const planet = document.createElement("div");
    planet.style.width = "200px";
    planet.style.height = "200px";
    planet.style.borderRadius = "50%";
    planet.style.margin = "auto";
    planet.style.background = `radial-gradient(circle at 30% 30%, ${color}, black)`;
    planet.style.boxShadow = "0 0 30px " + color;
    planetContainer.appendChild(planet);

    // 🛰️ Texto + emoji + clase CSS de la predicción IA
    let predText = (result.pred_label || result.disposition || "UNKNOWN").toUpperCase();
    let predClass = "pred-unknown";
    let predEmoji = "🔍";

    if (predText.includes("CONFIRMED")) {
      predClass = "pred-confirmed";
      predEmoji = "✅";
    } else if (predText.includes("CANDIDATE")) {
      predClass = "pred-candidate";
      predEmoji = "🟡";
    } else if (predText.includes("FALSE")) {
      predClass = "pred-false";
      predEmoji = "❌";
    }

    // 📊 Mostrar datos reales del planeta
    dataContainer.innerHTML = `
      <h2>🌍 Nuevo Exoplaneta Detectado</h2>
      <p><b>Nombre:</b> ${result.planet_name}</p>
      <p><b>Año de descubrimiento:</b> ${result.disc_year ?? "Desconocido"}</p>
      <p><b>Método:</b> ${result.method}</p>
      <p><b>Radio:</b> ${result.radius ? result.radius + " radios terrestres" : "N/A"}</p>
      <p><b>Masa:</b> ${result.mass ? result.mass + " masas terrestres" : "N/A"}</p>
      <p><b>Disposición oficial NASA:</b> ${result.disposition ?? "No disponible"}</p>
      <p><b>Predicción IA:</b> <span class="${predClass}">${predEmoji} ${predText}</span></p>
      <hr>
      <p><b>Columnas usadas (IA):</b> ${result.features.join(", ")}</p>
      <p><b>Valores de entrada:</b> ${JSON.stringify(result.input)}</p>
    `;

    // 📈 Mostrar gráficos recibidos directamente desde la predicción
    if (result.histograma) {
      histogramaImg.src = "data:image/png;base64," + result.histograma;
      histogramaImg.style.display = "block";
    }

    if (result.dispersion) {
      dispersionImg.src = "data:image/png;base64," + result.dispersion;
      dispersionImg.style.display = "block";
    }

  } catch (error) {
    console.error("Error:", error);
    dataContainer.innerHTML = `
      <h2>⚠️ Error al conectar con la IA</h2>
      <p>Verifica que el servidor Flask o Render esté activo.</p>
    `;
  }
});
