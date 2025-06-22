import os
import json
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# MODELOS A UTILIZAR
MODELOS = {
    "bert_multilingual": {
        "path": "nlptown/bert-base-multilingual-uncased-sentiment",
        "field": "emocionalidad_bert"
    },
    "roberta_multilingual": {
        "path": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "field": "emocionalidad_roberta_multi"
    },
    "roberta_english": {
        "path": "siebert/sentiment-roberta-large-english",
        "field": "emocionalidad_roberta_en"
    }
}

# CONFIG
MAX_ARTICLES = 300  
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Funciones de procesamiento de modelos

def load_model(model_name):
    model_info = MODELOS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_info["path"])
    model = AutoModelForSequenceClassification.from_pretrained(model_info["path"]).to(DEVICE)
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

def procesar_emocionalidad(seccion, modelo):
    tokenizer, model = load_model(modelo)
    model_info = MODELOS[modelo]
    field = model_info["field"]

    df = pd.read_csv(f"outputs/experiment3/resultados_{seccion}.csv")
    if modelo == "roberta_english":
        textos = (df["titulo_en"] + ". " + df["contenido_en"]).fillna("").tolist()
    else:
        textos = (df["titulo"] + ". " + df["contenido"]).fillna("").tolist()

    textos = textos[:MAX_ARTICLES]
    print(f"Procesando {len(textos)} articulos con modelo {modelo}...")
    df = df.iloc[:MAX_ARTICLES].copy()
    df[field] = [predict_emotion(t, tokenizer, model) for t in tqdm(textos)]
    df.to_csv(f"outputs/experiment3/resultados_{seccion}_{modelo}.csv", index=False)
    print(f"Guardado en outputs/experiment3/resultados_{seccion}_{modelo}.csv")

def procesar_vader(seccion):
    df = pd.read_csv(f"outputs/experiment3/resultados_{seccion}.csv")
    textos = (df["titulo_en"] + ". " + df["contenido_en"]).fillna("").tolist()
    textos = textos[:MAX_ARTICLES]

    print(f"Procesando {len(textos)} articulos con VADER...")
    df = df.iloc[:MAX_ARTICLES].copy()
    df["emocionalidad_vader"] = [predict_vader(t) for t in tqdm(textos)]
    df.to_csv(f"outputs/experiment3/resultados_{seccion}_vader.csv", index=False)
    print(f"Guardado en outputs/experiment3/resultados_{seccion}_vader.csv")

if __name__ == "__main__":
    with open("configs/process_config.json") as f:
        config = json.load(f)

    seccion = config["section"]
    modelo = config.get("modelo", "bert_multilingual")

    if modelo == "vader":
        procesar_vader(seccion)
    else:
        procesar_emocionalidad(seccion, modelo)