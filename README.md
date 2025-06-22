# NLP-media-analysis
Este proyecto aplica técnicas de procesamiento de lenguaje natural (NLP) para analizar artículos periodísticos publicados en el diario **Página 12**, con el objetivo de estudiar su **carga emocional**, **subjetividad** y **objetividad**. A partir de estas métricas, se exploran posibles patrones relacionados con el contenido, autoría, temática o contexto político.

## Objetivo principal

Evaluar y visualizar cómo varía el tono emocional y el grado de objetividad de las noticias según:

- El tema tratado (e.g. política, economía)
- El autor o la sección
- Comparación entre herramientas de análisis de sentimiento

## Tecnologías y librerías utilizadas

- **Python 3**
- **Web scraping**:
  - `requests`
  - `BeautifulSoup`
  - `pandas`

- **Preprocesamiento de texto**:
  - `nltk`
  - `spaCy` 
  - `Deep Translator`

- **Análisis de sentimiento y subjetividad**:
  - `TextBlob` (versión en español)
  - `VADER` (traducción opcional de textos)
  - `transformers`
    - [BERT-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment#accuracy)
    - [twitter-XLM-roBERTa-base](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)

- **Análisis exploratorio y visualización**:
  - `matplotlib`
  - `seaborn`
  - `wordcloud`
  - `scipy`

- **Modelado temático**:
  - `gensim` (LDA para topic modeling)

## Instalación de Dependencias
Se deben instalar las dependencias necesarias
```bash
pip install -r requirements.txt
```
## Scraping de Artículos
Primero se realizó el scraping de ~200 páginas de cada sección (*Economia*, *El Mundo*, *El País*, *Sociedad*, *Negrx*) de la página web de [Página 12](https://www.pagina12.com.ar/). Se hizo ejecutando el siguiente archivo:
```bash
python src/scraper.py
```
Que automáticamente toma las configuraciones especificadas en el archivo 'configs/scraper_config.json'. Un ejemplo de posibles valores para esta configuración:
```json
{
  "section": "el-mundo",
  "max_pages": 200,
  "starting_page": 0
}
```
Los valores válidos para 'section' son: {*economia*, *el-pais*, *el-mundo*, *sociedad*, *negrx*}
## Análisis Exploratorio
1. WordClouds y Frecuencias
    Ejecutar el siguiente comando
    ```bash
    python src/wordcloud_analysis.py
    ```
    Se tomarán las configuraciones del archivo 'configs/analysis_config.json' donde se especifica la sección y preferencias de preprocesamiento:
    ```json
    {
      "section": "economia",
      "remove_stopwords": true,
      "lemmatize": false
    }
    ``` 
    Se generará la WordCloud y gráfico de frecuencias con las 20 palabras más frecuentes de la sección especificada.

2. WordCount
    Ejecutar el siguiente comando
    ```bash
    python src/wordcount_analysis.py
    ```
    Genera un gráfico de barras con el promedio de cantidad de palabras por artículo en las secciones: *Economía*, *Sociedad*, *El País*.
3. Autoría
    Ejecutar el siguiente comando
    ```bash
    python src/authorship_analysis.py
    ```
    Genera gráfico comparando porcentaje de artículos con y sin autor en las 5 secciones.
4. Emocionalidad
    Ejecutar el siguiente comando
    ```bash
    python src/emotional_analysis.py
    ```
    Utiliza [*BERT base multilingual uncased sentiment*](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment#accuracy) para medir la valencia emocional de 50 artículos de cada sección. Toma como máximo 512 tokens, los artículos que se exceden de ese límite serán truncados. Se genera un gráfico de barras con el promedio de puntaje de valencia emocional de cada sección, donde un puntaje muy bajo (1) representa sentimiento negativo, mientras que uno alto (5) representa sentimiento positivo.

## Experimentos
### Experimento 1
Evaluar emocionalidad y subjetividad de los artículos periodísticos

Se realizan las traducciones de los artículos de cada sección (su título y contenido):
```bash
python src/experiment1/translate_articles.py
```
Las traducciones quedan guardadas en '*data/translated*'.
Luego se debe realizar el procesamiento de los artículos de cada sección para analizar emocionalidad y subjetividad:
```bash
python src/experiment1/process_section.py
```
Donde se debe aclarar en la configuración (*configs/process_config.json*) la sección que se desea procesar:
```json
{
  "section": "economia"
}
```
Una vez procesadas todas las secciones, se deben generar los gráficos comparativos:
```bash
python src/experiment1/graphing.py
```
Los resultados se guardarán en *outputs/experiment1/finales*

### Experimento 2
Análisis temático mediante LDA

Este experimento utiliza técnicas de modelado temático (Latent Dirichlet Allocation - LDA) para descubrir las principales temáticas presentes en los artículos, así como su relación con la emocionalidad, subjetividad y autoría.

Paso 1: Procesamiento y modelado LDA
Los artículos se preprocesan (lematización y remoción de stopwords) y se aplica LDA sobre cada sección por separado para identificar temas dominantes.

Ejecutar el siguiente comando:
```bash
python src/experiment2/process_lda.py
```
Esto realiza:

Preprocesamiento de los artículos por sección

Modelado LDA con 6 tópicos por sección

Asociación de cada artículo a su tópico dominante

Guardado de resultados y palabras clave por tópico en:
```bash
outputs/experiment2/
```
Paso 2: Generación de visualizaciones
Una vez asignado el tópico dominante a cada artículo, se generan visualizaciones para responder las siguientes preguntas:

¿Qué temáticas dominan cada sección?

¿Es posible identificar patrones de subjetividad/emocionalidad asociados a ciertos tópicos?

¿Autores específicos presentan patrones emocionales particulares por tema?

Ejecutar:
```bash
python src/experiment2/graphing.py
```
Se generarán automáticamente los siguientes gráficos:

Gráfico de barras: temas más frecuentes por sección

Gráfico de barras: emocionalidad promedio por tópico

Gráfico de barras: subjetividad promedio por tópico

Gráfico de barras: tópicos más frecuentes en artículos con alta vs baja emocionalidad

Heatmap: promedio de emocionalidad por autor y tópico

Gráfico global: comparación de tópicos entre artículos más emocionales vs menos emocionales (todas las secciones)

Los resultados se guardan automáticamente en:
```bash
outputs/experiment2/
```
### Experimento 3
Evaluar la consistencia y diferencias entre herramientas que analizan emocionalidad

Se utilizarán las traducciones de experimentos anteriores para herramientas que lo requieran

Se debe realizar el procesamiento de los artículos de cada sección para analizar emocionalidad según cada modelo:
```bash
python src/experiment3/processing.py
```
Donde se debe aclarar en la configuración (*configs/process_config.json*) la sección que se desea procesar:
```json
{
  "section": "sociedad"
}
```
Una vez procesadas todas las secciones, se deben generar los gráficos y métricas comparativas:
```bash
python src/experiment3/graphing.py
```
Los resultados se guardarán en *outputs/experiment3/analisis*
