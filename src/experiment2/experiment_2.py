
# Experimento 2 - Análisis temático mediante LDA

## Preguntas a responder:
# 1. ¿Qué temáticas dominan cada sección?
# 2. ¿Es posible identificar patrones de subjetividad/emocionalidad asociados a ciertos tópicos?
# 3. ¿Autores específicos presentan patrones emocionales particulares por tema?

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import gensim
from gensim import corpora
from collections import defaultdict
from tqdm import tqdm

# Cargar modelo SpaCy en español
tqdm.pandas()
nlp = spacy.load("es_core_news_sm")

# Rutas dinámicas según entorno
base_path = "/content/NLP-media-analysis" if os.path.exists("/content") else "NLP-media-analysis"
data_path = os.path.join(base_path, "data/new")
emo_path = os.path.join(base_path, "outputs/experiment1")
output_path = os.path.join(base_path, "outputs/experiment2")
os.makedirs(output_path, exist_ok=True)

# Parte 1 - Cargar artículos y procesar con LDA

def load_articles():
    file_path = os.path.join(emo_path, "finales/resultados_experimento1.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["contenido"])
    return df

def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and len(token) > 2]

def apply_lda_to_section(df, section_name, num_topics=6):
    df = df[df["contenido"].notna() & df["contenido"].apply(lambda x: isinstance(x, str))].copy()
    texts = df["contenido"].progress_apply(preprocess_text)
    texts = [t for t in texts if len(t) > 0]
    if len(texts) == 0:
        print(f"No hay textos válidos para la sección {section_name}.")
        return None, None, None

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, random_state=42)

    dominant_topics = []
    for doc_bow in corpus:
        topic_probs = lda_model.get_document_topics(doc_bow)
        dominant = max(topic_probs, key=lambda x: x[1])[0]
        dominant_topics.append(dominant)

    df = df.iloc[:len(texts)].copy()
    df["tokens"] = texts
    df["topic"] = dominant_topics
    return df, lda_model, dictionary

def run_lda():
    all_articles = load_articles()
    secciones = all_articles["seccion"].unique()
    for seccion in secciones:
        print(f"
Procesando sección: {seccion}")
        df_seccion = all_articles[all_articles["seccion"] == seccion].copy()
        processed_df, lda_model, dictionary = apply_lda_to_section(df_seccion, seccion, num_topics=6)
        if processed_df is not None:
            processed_df.to_csv(f"{output_path}/lda_{seccion}.csv", index=False)
            save_topic_keywords(lda_model, seccion)

# Ejecutar procesamiento
run_lda()

# Unir resultados de experimento1 con tópicos
lda_files = glob.glob(os.path.join(output_path, "lda_*.csv"))
emo_df = pd.read_csv(os.path.join(emo_path, "resultados_experimento1.csv"))

dfs = []
for f in lda_files:
    lda_df = pd.read_csv(f)
    seccion = os.path.basename(f).replace("lda_", "").replace(".csv", "")
    lda_df["seccion"] = seccion
    dfs.append(lda_df)

lda_full = pd.concat(dfs, ignore_index=True)

# Unir por título
merged = pd.merge(emo_df, lda_full[["titulo", "topic"]], on="titulo", how="left")

# Guardar para graficar
merged.to_csv(os.path.join(output_path, "resultados_experimento2.csv"), index=False)
