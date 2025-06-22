import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.stats import ttest_ind

# CONFIGURACION
SECCIONES = ["economia", "el-mundo", "el-pais", "negrx", "sociedad"]
RESULTS_DIR = "outputs/experiment1"
OUT_DIR = "outputs/experiment1/truncados"
os.makedirs(OUT_DIR, exist_ok=True)

# Juntar resultados
dfs = []
for seccion in SECCIONES:
    path = os.path.join(RESULTS_DIR, f"resultados_truncados_{seccion}.csv")
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
resumen["emocionalidad_desviada"] = resumen["emocionalidad"] - 3
print("\n--- Promedios por sección ---")
print(resumen.to_string(index=False))

color_map = {
    "emocionalidad": "#FF6F61",  # reddish
    "subjetividad": "#6A5ACD",   # purplish
    "objetividad": "#2E8B57",    # greenish
}

for var in ["emocionalidad", "subjetividad", "objetividad"]:
    plt.figure(figsize=(10, 6))

    if var == "emocionalidad":
        sns.barplot(x="seccion", y="emocionalidad_desviada", data=resumen, color=color_map[var])
        plt.title("Emocionalidad desviada (con respecto al neutro 3)")
        plt.ylabel("Emocionalidad - 3")
        plt.axhline(0, color="gray", linestyle="--")  # línea de referencia neutra
        plt.ylim(-2, 2)
    else:
        sns.barplot(x="seccion", y=var, data=resumen, color=color_map[var])
        plt.title(f"{var.capitalize()} promedio por sección")
        plt.ylim(0, 1)

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
print("\n--- Correlación emocionalidad vs subjetividad (Pearson) ---")
print(full_df[["emocionalidad", "subjetividad"]].corr(method="pearson"))
print("\n--- Estadísticas conjuntas por sección ---")
print(full_df.groupby("seccion")[["emocionalidad", "subjetividad"]].mean())
plt.figure(figsize=(10, 6))
sns.scatterplot(data=full_df, x="emocionalidad", y="subjetividad", hue="seccion", alpha=0.6)
plt.title("Emocionalidad vs Subjetividad por artículo")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "emocionalidad_vs_subjetividad.png"))
plt.close()

# 4. KDE plot de emocionalidad
plt.figure(figsize=(10, 6))
sns.kdeplot(data=full_df, x="emocionalidad", hue="seccion", common_norm=False)
plt.axvline(x=3.0, color="gray", linestyle="--", linewidth=1.5, label="Neutral (3.0)")
plt.title("Distribución de emocionalidad por sección (KDE)")
plt.xlabel("Emocionalidad")
plt.ylabel("Densidad")
plt.legend()
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
full_df["emocionalidad_desviada"] = full_df["emocionalidad"] - 3

print("\n--- Estadísticas emocionales por tipo de autor ---")
stats_autor = full_df.groupby("autor_cat")["emocionalidad"].describe()
print(stats_autor)
stats_autor.to_csv(os.path.join(OUT_DIR, "estadisticas_emocionalidad_autor.csv"))

plt.figure(figsize=(8, 6))
sns.boxplot(data=full_df, x="autor_cat", y="emocionalidad_desviada")
plt.axhline(0, color="gray", linestyle="--")
plt.title("Emocionalidad (desviada) por tipo de autor")
plt.ylabel("Emocionalidad - 3")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "boxplot_autoria_emocionalidad_desviada.png"))
plt.close()

emocionalidad_desconocido = full_df[full_df["autor_cat"] == "Desconocido"]["emocionalidad"]
emocionalidad_identificado = full_df[full_df["autor_cat"] == "Identificado"]["emocionalidad"]

t_stat, p_val = ttest_ind(emocionalidad_desconocido, emocionalidad_identificado, equal_var=False)
print(f"\nT-test: t = {t_stat:.3f}, p = {p_val:.4f}")

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

# 9. Top 5 más subjetivos y más objetivos
top_subjetivos = full_df.sort_values("subjetividad", ascending=False).head(5)
top_subjetivos[["titulo", "seccion", "subjetividad"]].to_csv(
    os.path.join(OUT_DIR, "top5_mas_subjetivos.csv"), index=False
)

top_objetivos = full_df.sort_values("subjetividad", ascending=True).head(5)
top_objetivos[["titulo", "seccion", "subjetividad"]].to_csv(
    os.path.join(OUT_DIR, "top5_mas_objetivos.csv"), index=False
)

print("\n--- Top 5 artículos más subjetivos ---")
print(top_subjetivos[["titulo", "seccion", "subjetividad"]])

print("\n--- Top 5 artículos más objetivos ---")
print(top_objetivos[["titulo", "seccion", "subjetividad"]])

print("\nGráficos y archivos de resumen guardados en 'outputs/experiment_1/finales'")