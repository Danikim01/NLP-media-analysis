# NLP-media-analysis
Este proyecto aplica técnicas de procesamiento de lenguaje natural (NLP) para analizar artículos periodísticos publicados en el diario **Página 12**, con el objetivo de estudiar su **carga emocional**, **subjetividad** y **objetividad**. A partir de estas métricas, se exploran posibles patrones relacionados con el contenido, autoría, temática o contexto político.

## Objetivo principal

Evaluar y visualizar cómo varía el tono emocional y el grado de objetividad de las noticias según:

- El tema tratado (e.g. política, economía)
- El autor o la sección
- Eventos específicos (e.g. cambios de gobierno)
- Comparación entre herramientas de análisis de sentimiento

## Tecnologías y librerías utilizadas

- **Python 3**
- **Web scraping**:
  - `requests`
  - `BeautifulSoup`
  - `pandas`

- **Preprocesamiento de texto**:
  - `nltk`
  - `spaCy` (`es_core_news_sm`)

- **Análisis de sentimiento y subjetividad**:
  - `TextBlob` (versión en español)
  - `VADER` (traducción opcional de textos)
  - `transformers` (`BERT` y variantes en español como BETO)

- **Análisis exploratorio y visualización**:
  - `matplotlib`
  - `seaborn`
  - `wordcloud`

- **Modelado temático**:
  - `gensim` (LDA para topic modeling)
