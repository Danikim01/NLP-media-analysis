import os
import json
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import XLMRobertaTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# MODELOS A UTILIZAR
MODELOS = {
    "bert_multilingual": {
        "path": "nlptown/bert-base-multilingual-uncased-sentiment",
        "field": "emocionalidad_bert",
        "lang": "es"
    },
    "roberta_multilingual": {
        "path": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "field": "emocionalidad_roberta_multi",
        "lang": "es"
    },
    "roberta_english": {
        "path": "siebert/sentiment-roberta-large-english",
        "field": "emocionalidad_roberta_en",
        "lang": "en"
    },
    "vader": {
        "field": "emocionalidad_vader",
        "lang": "en"
    }
}

# CONFIG
MAX_ARTICLES = 300
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Funciones de procesamiento de modelos

def load_model(model_name):
    model_info = MODELOS[model_name]
    path = model_info["path"]

    # Caso especial: modelo que rompe si usamos AutoTokenizer
    if model_name == "roberta_multilingual":
        tokenizer = XLMRobertaTokenizer.from_pretrained(path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)

    model = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True).to(DEVICE)
    return tokenizer, model

def predict_emotion(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).squeeze()

    if len(scores) == 5:
        stars = torch.arange(1, 6, dtype=torch.float).to(DEVICE)  # BERT
        return torch.dot(scores, stars).item()
    elif len(scores) == 3:
        values = torch.tensor([-1, 0, 1], dtype=torch.float).to(DEVICE)  # RoBERTa
        return torch.dot(scores, values).item()
    else:
        return float("nan")

def predict_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)["compound"]

def procesar_seccion(seccion):
    path = f"outputs/experiment3/resultados_{seccion}.csv"
    df = pd.read_csv(path)
    df = df.iloc[:MAX_ARTICLES].copy()

    for modelo in MODELOS:
        field = MODELOS[modelo]["field"]
        lang = MODELOS[modelo]["lang"]

        if field in df.columns:
            print(f"Saltando modelo {modelo}, ya existe columna {field}")
            continue

        print(f"Procesando modelo: {modelo}")

        if modelo == "vader":
            textos = (df["titulo_en"] + ". " + df["contenido_en"]).fillna("").tolist()
            df[field] = [predict_vader(t) for t in tqdm(textos)]
        else:
            tokenizer, model = load_model(modelo)
            if lang == "en":
                textos = (df["titulo_en"] + ". " + df["contenido_en"]).fillna("").tolist()
            else:
                textos = (df["titulo"] + ". " + df["contenido"]).fillna("").tolist()
            df[field] = [predict_emotion(t, tokenizer, model) for t in tqdm(textos)]

    df.to_csv(path, index=False)
    print(f"Actualización completada: {path}")

if __name__ == "__main__":
    with open("configs/process_config.json") as f:
        config = json.load(f)

    seccion = config["section"]
    procesar_seccion(seccion)