import pandas as pd
import matplotlib.pyplot as plt

SECCIONES = ["el-pais", "economia", "sociedad"]
wordcounts = {}

for seccion in SECCIONES:
    path = f"data/raw/articulos_{seccion}.csv"
    df = pd.read_csv(path)

    # Concatenar título + contenido y contar palabras
    df["texto_completo"] = df["titulo"].fillna("") + " " + df["contenido"].fillna("")
    df["wordcount"] = df["texto_completo"].apply(lambda x: len(x.split()))

    promedio = df["wordcount"].mean()
    wordcounts[seccion] = promedio

# Plot
plt.figure(figsize=(8, 6))
plt.bar(wordcounts.keys(), wordcounts.values(), color="steelblue")
plt.title("Cantidad promedio de palabras por sección")
plt.xlabel("Sección")
plt.ylabel("Promedio de palabras (título + contenido)")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("outputs/wordcount_por_seccion.png")
plt.show()