
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Rutas dinámicas según entorno
base_path = "/content/NLP-media-analysis" if os.path.exists("/content") else "NLP-media-analysis"
emo_path = os.path.join(base_path, "outputs/experiment1/finales")
output_path = os.path.join(base_path, "outputs/experimento2")
os.makedirs(output_path, exist_ok=True)

# Visualizaciones a partir de resultados existentes

def plot_topic_distribution(df, section_name):
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x="topic", order=sorted(df.topic.unique()))
    plt.title(f"Distribución de tópicos - {section_name.capitalize()}")
    plt.xlabel("Tópico dominante")
    plt.ylabel("Cantidad de artículos")
    plt.tight_layout()
    plt.savefig(f"{output_path}/plot_temas_{section_name}.png")
    plt.close()

def plot_emotion_subjectivity(df, section_name):
    if "emocionalidad_full" in df.columns:
        plt.figure(figsize=(10,6))
        sns.barplot(data=df, x="topic", y="emocionalidad_full", estimator=np.mean)
        plt.title(f"Emocionalidad promedio por tópico - {section_name}")
        plt.tight_layout()
        plt.savefig(f"{output_path}/emocionalidad_{section_name}.png")
        plt.close()
    if "subjetividad_full" in df.columns:
        plt.figure(figsize=(10,6))
        sns.barplot(data=df, x="topic", y="subjetividad_full", estimator=np.mean)
        plt.title(f"Subjetividad promedio por tópico - {section_name}")
        plt.tight_layout()
        plt.savefig(f"{output_path}/subjetividad_{section_name}.png")
        plt.close()

def plot_most_vs_least_emotional(df, section_name):
    if "emocionalidad_full" not in df.columns:
        return
    threshold = df["emocionalidad_full"].median()
    df["grupo_emo"] = np.where(df["emocionalidad_full"] >= threshold, "alta", "baja")
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x="topic", hue="grupo_emo")
    plt.title(f"Tópicos más frecuentes: alta vs baja emocionalidad - {section_name}")
    plt.tight_layout()
    plt.savefig(f"{output_path}/emocionalidad_vs_temas_{section_name}.png")
    plt.close()

def plot_heatmap_autores(df, section_name):
    if "autor" not in df.columns or "emocionalidad_full" not in df.columns:
        return
    top_autores = df["autor"].value_counts().head(10).index
    top_topics = df["topic"].value_counts().head(6).index
    pivot = df[df["autor"].isin(top_autores) & df["topic"].isin(top_topics)].pivot_table(
        index="autor", columns="topic", values="emocionalidad_full", aggfunc="mean"
    )
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, cmap="Reds")
    plt.title(f"Heatmap: emocionalidad por autor y tópico - {section_name}")
    plt.tight_layout()
    plt.savefig(f"{output_path}/heatmap_autor_tema_{section_name}.png")
    plt.close()

def plot_most_vs_least_emotional_by_section(df):
    if "emocionalidad_full" not in df.columns:
        return
    df = df.copy()
    df["grupo_emo"] = df.groupby("seccion")["emocionalidad_full"].transform(lambda x: np.where(x >= x.median(), "alta", "baja"))
    plt.figure(figsize=(12,6))
    sns.countplot(data=df, x="topic", hue="grupo_emo")
    plt.title("Comparativa de tópicos frecuentes en artículos más vs menos emocionales por sección")
    plt.tight_layout()
    plt.savefig(f"{output_path}/emocionalidad_vs_temas_todas_secciones.png")
    plt.close()

def generate_all_graphs():
    file_path = os.path.join(output_path, "resultados_experimento2.csv")
    if not os.path.exists(file_path):
        print("No se encontró resultados_experimento2.csv")
        return
    df = pd.read_csv(file_path)

    # Renombrar columnas reales para que coincidan con las que usan las funciones de graficado
    df.rename(columns={
        "emocionalidad": "emocionalidad_full",
        "subjetividad": "subjetividad_full"
    }, inplace=True)

    if "topic" not in df.columns:
        print("No se encontró la columna 'topic'. Asegurate de haber ejecutado LDA antes.")
        return

    secciones = df["seccion"].dropna().unique()
    for seccion in secciones:
        df_seccion = df[df["seccion"] == seccion].copy()
        plot_topic_distribution(df_seccion, seccion)
        plot_emotion_subjectivity(df_seccion, seccion)
        plot_most_vs_least_emotional(df_seccion, seccion)
        plot_heatmap_autores(df_seccion, seccion)

    plot_most_vs_least_emotional_by_section(df)



# Ejecutar visualizaciones solamente
generate_all_graphs()
