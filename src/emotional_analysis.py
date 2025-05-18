import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import os

# Inicializar pipeline de análisis de sentimiento multilingüe
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Secciones a analizar
SECCIONES = ["economia", "el-pais", "sociedad", "el-mundo", "negrx"]
resultados = {}

for seccion in SECCIONES:
    path = f"data/raw/articulos_{seccion}.csv"
    if not os.path.exists(path):
        print(f"Archivo no encontrado para sección: {seccion}")
        continue

    df = pd.read_csv(path)
    df = df.dropna(subset=["contenido"])  # Asegurar que hay texto

    # Unir título + contenido (si hay título)
    textos = (df["titulo"].fillna("") + ". " + df["contenido"]).tolist()

    # Acotar a primeros 50 artículos para que sea rápido
    textos = textos[:50]

    # Aplicar análisis de sentimiento
    puntajes = []
    for t in textos:
        try:
            result = classifier(t[:512])[0]  # recortamos a 512 tokens
            estrellas = int(result["label"][0])  # "4 stars" -> 4
            puntajes.append(estrellas)
        except Exception as e:
            print(f"Error al analizar un texto: {e}")

    if puntajes:
        promedio = sum(puntajes) / len(puntajes)
        resultados[seccion] = promedio
    else:
        resultados[seccion] = 0

# Graficar resultados
plt.figure(figsize=(8, 6))
plt.bar(resultados.keys(), resultados.values(), color="mediumpurple")
plt.title("Emocionalidad promedio por sección (1 a 5 estrellas)")
plt.xlabel("Sección")
plt.ylabel("Puntaje promedio")
plt.ylim(1, 5)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("outputs/emocionalidad_por_seccion.png")
plt.show()
