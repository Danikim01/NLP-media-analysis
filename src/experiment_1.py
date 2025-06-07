import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils import validar_seccion
from preprocessing import limpiar_texto
from tqdm import tqdm
from deep_translator import GoogleTranslator
from scipy.stats import zscore

# Configuraciones
SECCIONES = ["economia", "el-mundo", "el-pais", "negrx", "sociedad"]
BERT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
MAX_LEN = 512
NUM_WORDS_SNIPPET = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL).to(device)
translator = GoogleTranslator(source="auto", target="en")

def analizar_emocionalidad(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).squeeze()
    # Valor esperado (1 a 5 estrellas)
    estrellas = torch.arange(1, 6, dtype=torch.float).to(device)
    return torch.dot(scores, estrellas).item()

def analizar_subjetividad(texto):
    try:
        traducido = translator.translate(texto)
        tb = TextBlob(traducido)
        return tb.sentiment.subjectivity
    except:
        return None

def procesar_articulos(path):
    df = pd.read_csv(path).dropna(subset=["titulo", "contenido"])
    titulos = df["titulo"].astype(str)
    contenidos = df["contenido"].astype(str)

    full = titulos + ". " + contenidos
    # snippet = titulos + ". " + contenidos.str.split().str[:NUM_WORDS_SNIPPET].str.join(" ")
    # solo_titulo = titulos

    print(f"Analizando emocionalidad y subjetividad para {len(df)} artículos...")
    df["emocionalidad_full"] = [analizar_emocionalidad(t) for t in tqdm(full)]
    # df["emocionalidad_snippet"] = [analizar_emocionalidad(t) for t in tqdm(snippet)]
    # df["emocionalidad_titulo"] = [analizar_emocionalidad(t) for t in tqdm(solo_titulo)]

    df["subjetividad_full"] = [analizar_subjetividad(t) for t in tqdm(full)]
    # df["subjetividad_snippet"] = [analizar_subjetividad(t) for t in tqdm(snippet)]
    # df["subjetividad_titulo"] = [analizar_subjetividad(t) for t in tqdm(solo_titulo)]

    df["objetividad_full"] = 1 - df["subjetividad_full"]
    # df["objetividad_snippet"] = 1 - df["subjetividad_snippet"]
    # df["objetividad_titulo"] = 1 - df["subjetividad_titulo"]

    return df

def graficar_promedios_por_seccion(resultados, variable, output_file):
    plt.figure(figsize=(10,6))
    sns.barplot(x="seccion", y=variable, data=resultados)
    plt.title(f"{variable.capitalize()} promedio por sección")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Gráfico promedios por sección guardado en {output_file}")

def graficar_emocionalidad_vs_subjetividad(df, output_file):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="emocionalidad_full", y="subjetividad_full", hue="seccion", alpha=0.7)
    plt.title("Emocionalidad vs Subjetividad por Sección")
    plt.ylabel("Subjetividad")
    plt.xlabel("Emocionalidad (1 a 5 estrellas)")
    plt.legend(title="Sección")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Gráfico emocionalidad vs subjetividad guardado en {output_file}")

def graficar_autoria(df, output_file):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x="autor", y="emocionalidad_full")
    plt.title("Emocionalidad por tipo de autor")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Gráfico autoría guardado en {output_file}")

def guardar_graficar_top_autores(full_df):
    autores_top = (
        full_df[full_df["autor"] != "Desconocido"]
        .groupby("autor")[["emocionalidad_full", "objetividad_full"]]
        .mean()
        .sort_values("emocionalidad_full", ascending=False)
        .head(10)
    )
    autores_top.to_csv("outputs/top_autores_emocionales.csv")

    autores_obj = (
        full_df[full_df["autor"] != "Desconocido"]
        .groupby("autor")[["emocionalidad_full", "objetividad_full"]]
        .mean()
        .sort_values("objetividad_full", ascending=False)
        .head(10)
    )
    autores_obj.to_csv("outputs/top_autores_objetivos.csv")

    plt.figure(figsize=(10,6))
    sns.barplot(x=autores_top["emocionalidad_full"], y=autores_top.index, palette="Reds_r")
    plt.title("Autores más emocionales (promedio)")
    plt.xlabel("Emocionalidad promedio")
    plt.tight_layout()
    plt.savefig("outputs/barplot_top_autores_emocionales.png")

def guardar_autores_outliers(full_df):
    df_autores = full_df[full_df["autor"] != "Desconocido"].copy()
    df_autores["emo_zscore"] = df_autores.groupby("autor")["emocionalidad_full"].transform(zscore)

    outliers = df_autores[df_autores["emo_zscore"].abs() > 2]  
    outliers.to_csv("outputs/outliers_autores_emocionalidad.csv", index=False)

def guardar_extremos(df):
    top_emo = df.sort_values("emocionalidad_full", ascending=False).head(30)
    low_emo = df.sort_values("emocionalidad_full").head(30)
    top_subj = df.sort_values("subjetividad_full", ascending=False).head(30)
    low_subj = df.sort_values("subjetividad_full").head(30)

    top_emo.to_csv("outputs/top_emocionales.csv", index=False)
    low_emo.to_csv("outputs/menos_emocionales.csv", index=False)
    top_subj.to_csv("outputs/top_subjetivos.csv", index=False)
    low_subj.to_csv("outputs/menos_subjetivos.csv", index=False)

def experimento_1():
    todos = []
    for seccion in SECCIONES:
        archivo = f"data/new/articulos_{seccion}.csv"
        if not os.path.exists(archivo):
            print(f"Archivo no encontrado: {archivo}")
            continue
        df = procesar_articulos(archivo)
        df["seccion"] = seccion
        todos.append(df)

    full_df = pd.concat(todos, ignore_index=True)
    full_df.to_csv("outputs/experimento1_resultados.csv", index=False)

    guardar_extremos(full_df)
    graficar_autoria(full_df, "outputs/emocionalidad_autoria.png")
    guardar_autores_outliers(full_df)
    guardar_graficar_top_autores(full_df)

    # Gráficos
    resumen = full_df.groupby("seccion")[
        ["emocionalidad_full", "subjetividad_full", "objetividad_full"]
    ].mean().reset_index()

    graficar_emocionalidad_vs_subjetividad(full_df, "outputs/emocionalidad_vs_subjetividad.png")

    for var in ["emocionalidad_full", "subjetividad_full", "objetividad_full"]:
        graficar_promedios_por_seccion(resumen, var, f"outputs/{var}_por_seccion.png")

if __name__ == "__main__":
    experimento_1()