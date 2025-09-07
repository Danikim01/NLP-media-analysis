# API de Análisis de Sentimientos

API REST para analizar emocionalidad y subjetividad de textos en español, basada en modelos BERT multilingües y TextBlob.

## Características

- **Análisis de Emocionalidad**: Score de 1-5 (1=muy negativo, 5=muy positivo) usando BERT multilingüe
- **Análisis de Subjetividad**: Score de 0-1 (0=objetivo, 1=subjetivo) usando TextBlob
- **Preprocesamiento**: Limpieza, lematización y remoción de stopwords en español
- **API REST**: Endpoints simples con documentación automática
- **Detección de Negatividad**: Endpoint específico para detectar sentimientos negativos

## Instalación

1. **Crear entorno virtual:**

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate   # Linux/Mac
```

2. **Instalar dependencias:**

```bash
pip install -r requirements_api.txt
```

3. **Descargar modelo de spaCy:**

```bash
python -m spacy download es_core_news_sm
```

## Uso

### Iniciar el servidor:

```bash
python api_server.py
```

El servidor estará disponible en: `http://localhost:8000`

### Documentación automática:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints Principales

### 1. Análisis Completo

```bash
POST /analyze
{
    "text": "Me encanta este producto!",
    "preprocess": true,
    "analysis_type": "both"
}
```

**Respuesta:**

```json
{
  "sentiment_score": 4.84,
  "subjectivity_score": 0.9,
  "sentiment_label": "muy_positivo",
  "processed_text": "encantar producto",
  "original_length": 25,
  "processed_length": 15,
  "analysis_type": "both"
}
```

### 2. Detección de Negatividad

```bash
POST /detect/negative
{
    "text": "Este servicio es terrible"
}
```

**Respuesta:**

```json
{
  "is_negative": true,
  "sentiment_score": 1.21,
  "sentiment_label": "muy_negativo",
  "confidence": 0.895
}
```

### 3. Análisis Rápido (sin preprocesamiento)

```bash
POST /analyze/quick
{
    "text": "Texto a analizar"
}
```

## Uso Programático

```python
from sentiment_analyzer import SentimentAnalyzer

# Inicializar analizador
analyzer = SentimentAnalyzer()

# Analizar texto
resultado = analyzer.analyze("Me encanta este producto!")
print(f"Sentimiento: {resultado['sentiment_score']} ({resultado['sentiment_label']})")
print(f"Subjetividad: {resultado['subjectivity_score']}")
```

## Pruebas

Ejecutar el script de pruebas:

```bash
python test_api.py
```

## Estructura del Proyecto

```
sentiment-api/
├── api_server.py          # Servidor FastAPI
├── sentiment_analyzer.py  # Clase principal de análisis
├── test_api.py           # Script de pruebas
├── requirements_api.txt  # Dependencias mínimas
└── README.md            # Este archivo
```

## Modelos Utilizados

- **BERT**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **spaCy**: `es_core_news_sm` (modelo en español)
- **TextBlob**: Para análisis de subjetividad (con traducción automática)

## Limitaciones

- Textos muy largos se truncan a 512 tokens para BERT
- La traducción para subjetividad puede fallar ocasionalmente
- Requiere conexión a internet para descargar modelos inicialmente

## Licencia

Basado en el proyecto NLP-media-analysis original.
