import pandas as pd
import matplotlib.pyplot as plt

SECCIONES = ["economia", "el-pais", "sociedad", "el-mundo", "negrx"]
total_con_autor = 0
total_sin_autor = 0

for seccion in SECCIONES:
    path = f"data/raw/articulos_{seccion}.csv"
    df = pd.read_csv(path)

    df["autor"] = df["autor"].fillna("Desconocido")
    total_con_autor += (df["autor"] != "Desconocido").sum()
    total_sin_autor += (df["autor"] == "Desconocido").sum()

# Preparar datos
labels = ["Con autor", "Desconocido"]
counts = [total_con_autor, total_sin_autor]
colors = ["#4CAF50", "#FF7043"]

# Pie chart
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
plt.title("Distribución de artículos con y sin autor (5 secciones)")
plt.axis("equal")  # círculo perfecto
plt.tight_layout()

# Guardar resultado
plt.savefig("outputs/distribucion_autoria_piechart.png")
plt.show()