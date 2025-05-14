import spacy
import nltk
from nltk.corpus import stopwords
import re

# Cargar recursos una vez
nlp = spacy.load("es_core_news_sm")
try:
    stopwords_es = set(stopwords.words("spanish"))
except LookupError:
    nltk.download("stopwords")
    stopwords_es = set(stopwords.words("spanish"))

def limpiar_texto(texto):
    """Elimina caracteres raros, múltiple espacios y baja a minúsculas"""
    texto = texto.lower()
    texto = re.sub(r"\s+", " ", texto)  # múltiples espacios → uno solo
    texto = re.sub(r"[^\wáéíóúñü\s]", "", texto)  # eliminar puntuación
    return texto.strip()

def remover_stopwords(texto):
    """Elimina stopwords en español"""
    palabras = texto.split()
    filtradas = [p for p in palabras if p not in stopwords_es]
    return " ".join(filtradas)

def lematizar(texto):
    """Lematiza un texto en español usando spaCy"""
    doc = nlp(texto)
    lemas = [token.lemma_ for token in doc if not token.is_punct]
    return " ".join(lemas)

def preprocesar(texto, stopwords=True, lematizar_texto=True):
    """Pipeline general para preprocesamiento"""
    texto = limpiar_texto(texto)
    if stopwords:
        texto = remover_stopwords(texto)
    if lematizar_texto:
        texto = lematizar(texto)
    return texto