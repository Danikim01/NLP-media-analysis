"""
Analizador de sentimientos minimalista basado en el proyecto NLP-media-analysis
Extrae solo las funciones esenciales para análisis de emocionalidad y subjetividad
"""

import spacy
import nltk
from nltk.corpus import stopwords
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
from deep_translator import GoogleTranslator
import warnings
warnings.filterwarnings("ignore")

class SentimentAnalyzer:
    def __init__(self):
        """Inicializa el analizador con los modelos necesarios"""
        print("Inicializando analizador de sentimientos...")
        
        # Cargar modelo BERT para emocionalidad
        self.bert_model = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.bert_model).to(self.device)
        
        # Cargar spaCy para preprocesamiento
        self.nlp = spacy.load("es_core_news_sm")
        
        # Cargar stopwords en español
        try:
            self.stopwords_es = set(stopwords.words("spanish"))
        except LookupError:
            nltk.download("stopwords")
            self.stopwords_es = set(stopwords.words("spanish"))
        
        # Inicializar traductor para subjetividad
        self.translator = GoogleTranslator(source="auto", target="en")
        
        print("Analizador inicializado correctamente!")
    
    def limpiar_texto(self, texto):
        """Elimina caracteres raros, múltiple espacios y baja a minúsculas"""
        texto = texto.lower()
        texto = re.sub(r"\s+", " ", texto)  # múltiples espacios → uno solo
        texto = re.sub(r"[^\wáéíóúñü\s]", "", texto)  # eliminar puntuación
        return texto.strip()
    
    def remover_stopwords(self, texto):
        """Elimina stopwords en español"""
        palabras = texto.split()
        filtradas = [p for p in palabras if p not in self.stopwords_es]
        return " ".join(filtradas)
    
    def lematizar(self, texto):
        """Lematiza un texto en español usando spaCy"""
        doc = self.nlp(texto)
        lemas = [token.lemma_ for token in doc if not token.is_punct]
        return " ".join(lemas)
    
    def preprocesar(self, texto, remove_stopwords=True, lemmatize=True):
        """Pipeline de preprocesamiento"""
        texto = self.limpiar_texto(texto)
        if remove_stopwords:
            texto = self.remover_stopwords(texto)
        if lemmatize:
            texto = self.lematizar(texto)
        return texto
    
    def analizar_emocionalidad(self, texto, max_length=512):
        """
        Analiza la emocionalidad usando BERT multilingüe
        Retorna un score de 1-5 (1=muy negativo, 5=muy positivo)
        """
        try:
            inputs = self.tokenizer(
                texto, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            scores = torch.softmax(outputs.logits, dim=1).squeeze()
            # Valor esperado (1 a 5 estrellas)
            estrellas = torch.arange(1, 6, dtype=torch.float).to(self.device)
            score = torch.dot(scores, estrellas).item()
            
            return round(score, 2)
        except Exception as e:
            print(f"Error en análisis de emocionalidad: {e}")
            return None
    
    def analizar_subjetividad(self, texto):
        """
        Analiza la subjetividad usando TextBlob
        Retorna un score de 0-1 (0=objetivo, 1=subjetivo)
        """
        try:
            # Traducir al inglés para mejor precisión de TextBlob
            traducido = self.translator.translate(texto)
            tb = TextBlob(traducido)
            return round(tb.sentiment.subjectivity, 3)
        except Exception as e:
            print(f"Error en análisis de subjetividad: {e}")
            return None
    
    def get_sentiment_label(self, score):
        """Convierte el score numérico a etiqueta descriptiva"""
        if score is None:
            return "error"
        elif score <= 1.5:
            return "muy_negativo"
        elif score <= 2.5:
            return "negativo"
        elif score <= 3.5:
            return "neutral"
        elif score <= 4.5:
            return "positivo"
        else:
            return "muy_positivo"
    
    def analyze(self, texto, preprocess=True):
        """
        Función principal que analiza un texto completo
        """
        if not texto or not texto.strip():
            return {
                "error": "Texto vacío o inválido",
                "sentiment_score": None,
                "subjectivity_score": None,
                "sentiment_label": "error",
                "processed_text": ""
            }
        
        # Preprocesar si se solicita
        processed_text = self.preprocesar(texto) if preprocess else texto
        
        # Analizar emocionalidad y subjetividad
        sentiment_score = self.analizar_emocionalidad(processed_text)
        subjectivity_score = self.analizar_subjetividad(processed_text)
        
        # Obtener etiqueta de sentimiento
        sentiment_label = self.get_sentiment_label(sentiment_score)
        
        return {
            "sentiment_score": sentiment_score,
            "subjectivity_score": subjectivity_score,
            "sentiment_label": sentiment_label,
            "processed_text": processed_text,
            "original_length": len(texto),
            "processed_length": len(processed_text)
        }

# Función de conveniencia para uso directo
def analyze_sentiment(texto, preprocess=True):
    """
    Función de conveniencia para análisis rápido
    """
    analyzer = SentimentAnalyzer()
    return analyzer.analyze(texto, preprocess)

if __name__ == "__main__":
    # Ejemplo de uso
    analyzer = SentimentAnalyzer()
    
    # Textos de prueba
    textos_prueba = [
        "Me encanta este producto, es fantástico!",
        "Este servicio es terrible, muy malo",
        "El clima está normal hoy",
        "Estoy muy preocupado por la situación económica del país"
    ]
    
    print("\n=== ANÁLISIS DE SENTIMIENTOS ===\n")
    
    for i, texto in enumerate(textos_prueba, 1):
        print(f"Texto {i}: {texto}")
        resultado = analyzer.analyze(texto)
        print(f"  Sentimiento: {resultado['sentiment_score']} ({resultado['sentiment_label']})")
        print(f"  Subjetividad: {resultado['subjectivity_score']}")
        print(f"  Texto procesado: {resultado['processed_text'][:100]}...")
        print()
