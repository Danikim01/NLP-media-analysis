import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIGURACION
SECCIONES = ["economia", "el-mundo", "el-pais", "negrx", "sociedad"]
RESULTS_DIR = "outputs/experiment_1"
OUT_DIR = "outputs/experiment_1/finales"
os.makedirs(OUT_DIR, exist_ok=True)

# Juntar resultados
dfs = []
for seccion in SECCIONES:
    path = os.path.join(RESULTS_DIR, f"experimento1_{seccion}.csv")
    if not os.path.exists(path):
        print(f"No se encuentra el archivo para {seccion}")
        continue
    df = pd.read_csv(path)
    df["seccion"] = seccion
    dfs.append(df)

if not dfs:
    raise ValueError("No hay datos para generar los gráficos.")

full_df = pd.concat(dfs, ignore_index=True)
full_df.to_csv(os.path.join(OUT_DIR, "resultados_experimento1_todo.csv"), index=False)

# GRAFICOS
print("Generando gráficos...")

# 1. Barras promedio por sección
resumen = full_df.groupby("seccion")[
    ["emocionalidad_full", "subjetividad_full", "objetividad_full"]
].mean().reset_index()

for var in ["emocionalidad_full", "subjetividad_full", "objetividad_full"]:
    plt.figure(figsize=(10, 6))
    sns.barplot(x="seccion", y=var, data=resumen)
    plt.title(f"{var.capitalize()} promedio por sección")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{var}_promedios.png"))
    plt.close()

# 2. Dispersión emocionalidad vs subjetividad
plt.figure(figsize=(10, 6))
sns.scatterplot(data=full_df, x="emocionalidad_full", y="subjetividad_full", hue="seccion", alpha=0.6)
plt.title("Emocionalidad vs Subjetividad por artículo")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "emocionalidad_vs_subjetividad.png"))
plt.close()

# 3. Boxplot por autor conocido/desconocido
plt.figure(figsize=(8, 6))
full_df["autor_cat"] = full_df["autor"].apply(lambda x: "Desconocido" if x == "Desconocido" else "Identificado")
sns.boxplot(data=full_df, x="autor_cat", y="emocionalidad_full")
plt.title("Emocionalidad por tipo de autor")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "boxplot_autoria_emocionalidad.png"))
plt.close()

print("Gráficos guardados en 'outputs/experiment_1/finales'")