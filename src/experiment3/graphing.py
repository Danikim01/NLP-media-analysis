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
    "emocionalidad_bert",
    "emocionalidad_roberta_multi",
    "emocionalidad_vader",
    "emocionalidad_bert_translated"
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
        df.rename(columns={"emocionalidad": "emocionalidad_bert_translated"}, inplace=True)
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

# 2. Estadísticas por modelo
print("\n--- Estadísticas por modelo ---")
stats_df = full_df[MODELOS].agg(["mean", "std", "min", "max"])
print(stats_df.T)
stats_df.T.to_csv(os.path.join(OUT_DIR, "estadisticas_modelos.csv"))

# 3. Boxplot original
plt.figure(figsize=(10, 6))
sns.boxplot(data=full_df[MODELOS])
plt.title("Distribución de puntajes por modelo (original)")
plt.ylabel("Emocionalidad")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "boxplots_modelos_original.png"))
plt.close()

# 3b. Normalización
normalized_cols = []
norm_df = full_df.copy()
for col in MODELOS:
    min_val = norm_df[col].min()
    max_val = norm_df[col].max()
    norm_col = f"{col}_norm"
    norm_df[norm_col] = 2 * ((norm_df[col] - min_val) / (max_val - min_val)) - 1  # [-1, 1]
    normalized_cols.append(norm_col)

# Boxplot normalizado
plt.figure(figsize=(10, 6))
sns.boxplot(data=norm_df[normalized_cols])
plt.title("Distribución de puntajes por modelo (normalizado)")
plt.ylabel("Emocionalidad normalizada")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "boxplots_modelos_normalizados.png"))
plt.close()

# 4. Discrepancia global con valores normalizados
norm_df["std_models"] = norm_df[normalized_cols].std(axis=1)
discrepantes = norm_df.sort_values("std_models", ascending=False).head(10)
discrepantes.to_csv(os.path.join(OUT_DIR, "top_discrepancias_global.csv"), index=False)

# 5. Discrepancias por par (normalizados)
discrepancias_pares = []
for m1, m2 in combinations(normalized_cols, 2):
    diff_col = f"diff_{m1}_{m2}"
    norm_df[diff_col] = np.abs(norm_df[m1] - norm_df[m2])
    top_diffs = norm_df.sort_values(diff_col, ascending=False).head(5)
    top_diffs[["titulo", "seccion", m1, m2, diff_col]].to_csv(
        os.path.join(OUT_DIR, f"discrepancias_{m1}_vs_{m2}.csv"), index=False
    )
    discrepancias_pares.append((m1, m2, norm_df[diff_col].mean()))

# 6. Resumen de discrepancias
print("\n--- Discrepancias promedio por par de modelos (normalizados) ---")
for m1, m2, avg in discrepancias_pares:
    print(f"{m1} vs {m2}: diferencia promedio = {avg:.3f}")

# 7. Diferencia BERT original vs traducido
full_df["bert_diff"] = full_df["emocionalidad_bert"] - full_df["emocionalidad_bert_translated"]
plt.figure(figsize=(10, 6))
sns.histplot(full_df["bert_diff"], bins=30, kde=True)
plt.title("Diferencia entre BERT original vs traducido")
plt.xlabel("emocionalidad_bert - emocionalidad_bert_translated")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bert_original_vs_traducido.png"))
plt.close()

# 8. Gráfico de discrepancias promedio por par de modelos
plt.figure(figsize=(10, 6))
pares_str = [f"{m1.replace('_norm', '')} vs {m2.replace('_norm', '')}" for m1, m2, _ in discrepancias_pares]
discrepancias_valores = [avg for _, _, avg in discrepancias_pares]
sns.barplot(x=discrepancias_valores, y=pares_str, palette="viridis")
plt.xlabel("Diferencia promedio (absoluta)")
plt.title("Discrepancia promedio por par de modelos (normalizados)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "discrepancias_promedio_por_par.png"))
plt.close()

# 9. Top 3 discrepancias entre BERT y RoBERTa (normalizados)
col_bert = "emocionalidad_bert_norm"
col_roberta = "emocionalidad_roberta_multi_norm"
col_diff = "diff_bert_roberta"

norm_df[col_diff] = np.abs(norm_df[col_bert] - norm_df[col_roberta])
top_3_discrepancias = norm_df.sort_values(col_diff, ascending=False).head(3)

top_3_discrepancias[[
    "titulo", "seccion", "emocionalidad_bert", "emocionalidad_roberta_multi", 
    col_bert, col_roberta, col_diff
]].to_csv(os.path.join(OUT_DIR, "top3_discrepancias_bert_vs_roberta.csv"), index=False)

print("\nTop 3 discrepancias entre BERT y RoBERTa (normalizados) guardadas.")

print("\nAnálisis y gráficos guardados en 'outputs/experiment3/analisis'")