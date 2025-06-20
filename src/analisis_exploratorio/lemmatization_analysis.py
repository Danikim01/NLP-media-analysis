import pandas as pd
import matplotlib.pyplot as plt
import sys
from preprocessing import limpiar_texto, lematizar
import re
sys.path.insert(0, "..")

def count_words(text):
    return len(re.findall(r"\b\w+\b", text))  # cuenta solo palabras reales

df = pd.read_csv("data/raw/articulos_el-pais.csv")
df["texto_completo"] = df["titulo"].fillna("") + " " + df["contenido"].fillna("")

# Limpieza previa
df["texto_limpio"] = df["texto_completo"].apply(limpiar_texto)

# Conteo original
original_counts = df["texto_limpio"].apply(count_words)

# Lematización
df["texto_lema"] = df["texto_limpio"].apply(lematizar)
lemmatized_counts = df["texto_lema"].apply(count_words)

# Diferencia
diferencias = original_counts - lemmatized_counts

# Graficar
plt.figure(figsize=(10, 6))
plt.hist(diferencias, bins=30, color="purple", edgecolor="black")
plt.title("Reducción de palabras tras lematización (el-pais)")
plt.xlabel("Palabras eliminadas (original - lematizado)")
plt.ylabel("Cantidad de artículos")
plt.tight_layout()
plt.savefig("outputs/diferencia_wordcount_lemmatizacion.png")
plt.show()