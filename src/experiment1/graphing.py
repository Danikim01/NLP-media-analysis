import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# CONFIGURACION
SECCIONES = ["economia", "el-mundo", "el-pais", "negrx", "sociedad"]
RESULTS_DIR = "outputs/experiment1"
OUT_DIR = "outputs/experiment1/finales"
os.makedirs(OUT_DIR, exist_ok=True)

# Juntar resultados
dfs = []
for seccion in SECCIONES:
    path = os.path.join(RESULTS_DIR, f"resultados_{seccion}.csv")
    if not os.path.exists(path):
        print(f"No se encuentra el archivo para {seccion}: {path}")
        continue
    df = pd.read_csv(path)
    df["seccion"] = seccion
    dfs.append(df)

if not dfs:
    raise ValueError("No hay datos para generar los gráficos.")

full_df = pd.concat(dfs, ignore_index=True)
full_df.to_csv(os.path.join(OUT_DIR, "resultados_experimento1.csv"), index=False)

print("Generando gráficos y estadísticas...")

# 1. Promedios por sección
resumen = full_df.groupby("seccion")[
    ["emocionalidad", "subjetividad", "objetividad"]
].mean().reset_index()
print("\n--- Promedios por sección ---")
print(resumen.to_string(index=False))

for var in ["emocionalidad", "subjetividad", "objetividad"]:
    plt.figure(figsize=(10, 6))
    sns.barplot(x="seccion", y=var, data=resumen)
    plt.title(f"{var.capitalize()} promedio por sección")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{var}_promedios.png"))
    plt.close()

# 2. Comparación entre economia y negrx
eco_negrx = full_df[full_df["seccion"].isin(["economia", "negrx"])]
plt.figure(figsize=(10, 6))
sns.scatterplot(data=eco_negrx, x="emocionalidad", y="subjetividad", hue="seccion", alpha=0.6)
plt.title("Emocionalidad vs Subjetividad (economia vs negrx)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "emocionalidad_vs_subjetividad_economia_negrx.png"))
plt.close()

# 3. Dispersión emocionalidad vs subjetividad (general)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=full_df, x="emocionalidad", y="subjetividad", hue="seccion", alpha=0.6)
plt.title("Emocionalidad vs Subjetividad por artículo")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "emocionalidad_vs_subjetividad.png"))
plt.close()

# 4. KDE plot de emocionalidad
plt.figure(figsize=(10, 6))
sns.kdeplot(data=full_df, x="emocionalidad", hue="seccion", common_norm=False)
plt.title("Distribución de emocionalidad por sección (KDE)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "kde_emocionalidad.png"))
plt.close()

# 5. Cuartiles por sección
print("\n--- Cuartiles de emocionalidad por sección ---")
cuartiles = full_df.groupby("seccion")["emocionalidad"].quantile([0.25, 0.5, 0.75]).unstack()
print(cuartiles)
cuartiles.to_csv(os.path.join(OUT_DIR, "cuartiles_emocionalidad.csv"))

# 6. Boxplot por tipo de autor
full_df["autor_cat"] = full_df["autor"].apply(lambda x: "Desconocido" if x == "Desconocido" else "Identificado")
plt.figure(figsize=(8, 6))
sns.boxplot(data=full_df, x="autor_cat", y="emocionalidad")
plt.title("Emocionalidad por tipo de autor")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "boxplot_autoria_emocionalidad.png"))
plt.close()

# 7. Artículos más emocionales y más negativos (global y por sección)
top_pos = full_df.sort_values("emocionalidad", ascending=False).head(5)
top_pos[["titulo", "seccion", "emocionalidad"]].to_csv(os.path.join(OUT_DIR, "top5_positivos.csv"), index=False)

bot_neg = full_df.sort_values("emocionalidad", ascending=True).head(5)
bot_neg[["titulo", "seccion", "emocionalidad"]].to_csv(os.path.join(OUT_DIR, "top5_negativos.csv"), index=False)

for seccion in SECCIONES:
    sub_df = full_df[full_df["seccion"] == seccion]
    sub_df.sort_values("emocionalidad", ascending=False).head(3)[["titulo", "emocionalidad"]].to_csv(
        os.path.join(OUT_DIR, f"top3_positivos_{seccion}.csv"), index=False)
    sub_df.sort_values("emocionalidad", ascending=True).head(3)[["titulo", "emocionalidad"]].to_csv(
        os.path.join(OUT_DIR, f"top3_negativos_{seccion}.csv"), index=False)

# 8. Z-score emocionalidad por sección
full_df["z_emo"] = full_df.groupby("seccion")["emocionalidad"].transform(zscore)
full_df["z_emo_abs"] = full_df["z_emo"].abs()
outliers = full_df[full_df["z_emo_abs"] > 2][["titulo", "seccion", "emocionalidad", "z_emo"]]
outliers.to_csv(os.path.join(OUT_DIR, "outliers_emocionalidad.csv"), index=False)
print("\n--- Guardados artículos outliers con Z-score > 2 ---")
print(outliers.head())

print("\nGráficos y archivos de resumen guardados en 'outputs/experiment_1/finales'")