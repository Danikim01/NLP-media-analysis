import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
from tqdm import tqdm
import torch

# Configuracion
BERT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
MAX_LEN = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL).to(device)

def analizar_emocionalidad(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).squeeze()
    estrellas = torch.arange(1, 6, dtype=torch.float).to(device)
    return torch.dot(scores, estrellas).item()

def analizar_subjetividad(texto):
    try:
        tb = TextBlob(texto)
        return tb.sentiment.subjectivity
    except:
        return None
    
def get_snippet(row, num_words=50):
    title = row["titulo_en"]
    content_words = row["contenido_en"].split()
    snippet = " ".join(content_words[:num_words])
    return f"{title}. {snippet}"

def procesar_seccion(seccion):
    input_path = f"data/translated/traducido_articulos_{seccion}.csv"
    output_path = f"outputs/experiment1/resultados_{seccion}.csv"

    if not os.path.exists(input_path):
        print(f"No se encontró el archivo traducido para la sección {seccion}")
        return

    df = pd.read_csv(input_path).dropna(subset=["titulo_en", "contenido_en"])
    # textos = df.apply(get_snippet, axis=1)
    # textos = df["titulo"] + ". " + df["contenido"]
    textos_bert = df["titulo"] + ". " + df["contenido"]
    textos_textblob = df["titulo_en"] + ". " + df["contenido_en"]


    print(f"Procesando sección {seccion} ({len(df)} artículos)...")
    df["emocionalidad"] = [analizar_emocionalidad(t) for t in tqdm(textos_bert)]
    df["subjetividad"] = [analizar_subjetividad(t) for t in tqdm(textos_textblob)]
    df["objetividad"] = 1 - df["subjetividad"]

    df.to_csv(output_path, index=False)
    print(f"Resultados guardados en {output_path}")

if __name__ == "__main__":
    import json
    with open("configs/process_config.json") as f:
        config = json.load(f)

    seccion = config["section"]
    procesar_seccion(seccion)