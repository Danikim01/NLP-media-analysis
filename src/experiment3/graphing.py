import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import numpy as np
from itertools import combinations

# CONFIG
SECCIONES = ["economia", "el-mundo", "el-pais", "negrx", "sociedad"]
MODELOS = [
    "bert_multilingual",
    "roberta_multilingual",
    "roberta_english",
    "vader"
]
INPUT_DIR = "outputs/experiment3"
OUT_DIR = "outputs/experiment3/analisis"
os.makedirs(OUT_DIR, exist_ok=True)

# Cargar resultados
dfs = []
for seccion in SECCIONES:
    path = os.path.join(INPUT_DIR, f"resultados_{seccion}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["seccion"] = seccion
        dfs.append(df)

if not dfs:
    raise ValueError("No hay datos disponibles para el análisis.")

full_df = pd.concat(dfs, ignore_index=True)
full_df.to_csv(os.path.join(OUT_DIR, "todos_resultados.csv"), index=False)

# 1. Correlaciones
print("\n--- Correlaciones entre modelos (Pearson) ---")
for m1, m2 in combinations(MODELOS, 2):
    r, _ = pearsonr(full_df[m1], full_df[m2])
    print(f"{m1} vs {m2}: Pearson r = {r:.3f}")

print("\n--- Correlaciones entre modelos (Spearman) ---")
for m1, m2 in combinations(MODELOS, 2):
    r, _ = spearmanr(full_df[m1], full_df[m2])
    print(f"{m1} vs {m2}: Spearman r = {r:.3f}")

# 2. Desviaciones y varianza
print("\n--- Estadísticas por modelo ---")
stats_df = full_df[MODELOS].agg(["mean", "std", "min", "max"])
print(stats_df.T)
stats_df.T.to_csv(os.path.join(OUT_DIR, "estadisticas_modelos.csv"))

# 3. Boxplots por modelo
plt.figure(figsize=(10, 6))
sns.boxplot(data=full_df[MODELOS])
plt.title("Distribución de puntajes por modelo")
plt.ylabel("Emocionalidad")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "boxplots_modelos.png"))
plt.close()

# 4. Discrepancias globales (desviación estándar por fila)
full_df["std_models"] = full_df[MODELOS].std(axis=1)
discrepantes = full_df.sort_values("std_models", ascending=False).head(10)
discrepantes.to_csv(os.path.join(OUT_DIR, "top_discrepancias_global.csv"), index=False)

# 5. Discrepancias por par de modelos
discrepancias_pares = []
for m1, m2 in combinations(MODELOS, 2):
    dif_col = f"diff_{m1}_{m2}"
    full_df[dif_col] = np.abs(full_df[m1] - full_df[m2])
    top_diffs = full_df.sort_values(dif_col, ascending=False).head(5)
    top_diffs[["titulo", "seccion", m1, m2, dif_col]].to_csv(os.path.join(
        OUT_DIR, f"discrepancias_{m1}_vs_{m2}.csv"), index=False)
    discrepancias_pares.append((m1, m2, full_df[dif_col].mean()))

# 6. Resumen de discrepancias promedio por par
print("\n--- Discrepancias promedio por par de modelos ---")
for m1, m2, avg_diff in discrepancias_pares:
    print(f"{m1} vs {m2}: diferencia promedio = {avg_diff:.3f}")

print("\nAnálisis y gráficos guardados en 'outputs/experiment3/analisis'")