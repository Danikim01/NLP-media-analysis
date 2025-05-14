# exploratory_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocessing import preprocesar
from utils import validar_seccion
import sys
import json
sys.path.insert(0, "..")

CONFIG_PATH = "configs/analysis_config.json"

with open(CONFIG_PATH) as f:
    config = json.load(f)

SECCION = config["section"]
if not validar_seccion(SECCION):
    raise ValueError(f"'{SECCION}' no es una sección válida.")

REMOVE_STOPWORDS = config.get("remove_stopwords", True)
LEMMATIZE = config.get("lemmatize", True)

file_path = f"data/raw/articulos_{SECCION}.csv"

# Cargar dataset
df = pd.read_csv(file_path)

# Elegir secciones a analizar (o toda la base)
articulos = df["contenido"].dropna().tolist()

# Preprocesar (probá cambiar estos flags)
preprocesados = [preprocesar(a, stopwords=REMOVE_STOPWORDS, lematizar_texto=LEMMATIZE) for a in articulos]

# Concatenar todo el corpus
texto_completo = " ".join(preprocesados)

# Wordcloud
wordcloud = WordCloud(width=1000, height=600, background_color="white", colormap="inferno").generate(texto_completo)

# Guardar wordcloud como archivo PNG
output_path = f"outputs/ff_wordcloud_{SECCION}.png"
wordcloud.to_file(output_path)

print(f"Wordcloud guardada en {output_path}")