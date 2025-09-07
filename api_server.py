"""
API REST para análisis de sentimientos
Basada en el proyecto NLP-media-analysis pero simplificada para uso como servicio
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import uvicorn
from sentiment_analyzer import SentimentAnalyzer
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="API de Análisis de Sentimientos",
    description="API para analizar emocionalidad y subjetividad de textos en español",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Inicializar el analizador (se carga una sola vez al iniciar el servidor)
analyzer = None

@app.on_event("startup")
async def startup_event():
    """Inicializa el analizador al arrancar el servidor"""
    global analyzer
    logger.info("Inicializando analizador de sentimientos...")
    analyzer = SentimentAnalyzer()
    logger.info("Servidor listo para recibir requests!")

# Modelos Pydantic para validación
class TextAnalysisRequest(BaseModel):
    text: str = Field(..., description="Texto a analizar", min_length=1, max_length=10000)
    preprocess: bool = Field(True, description="Aplicar preprocesamiento (limpieza, lematización)")
    analysis_type: Literal["sentiment", "subjectivity", "both"] = Field("both", description="Tipo de análisis a realizar")

class TextAnalysisResponse(BaseModel):
    sentiment_score: Optional[float] = Field(None, description="Score de emocionalidad (1-5)")
    subjectivity_score: Optional[float] = Field(None, description="Score de subjetividad (0-1)")
    sentiment_label: str = Field(..., description="Etiqueta del sentimiento")
    processed_text: str = Field(..., description="Texto después del preprocesamiento")
    original_length: int = Field(..., description="Longitud del texto original")
    processed_length: int = Field(..., description="Longitud del texto procesado")
    analysis_type: str = Field(..., description="Tipo de análisis realizado")

class HealthResponse(BaseModel):
    status: str
    message: str
    analyzer_ready: bool

# Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Endpoint de salud básico"""
    return HealthResponse(
        status="ok",
        message="API de Análisis de Sentimientos funcionando correctamente",
        analyzer_ready=analyzer is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de salud detallado"""
    return HealthResponse(
        status="ok",
        message="Servicio de análisis de sentimientos operativo",
        analyzer_ready=analyzer is not None
    )

@app.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analiza el sentimiento y/o subjetividad de un texto
    
    - **text**: Texto a analizar (máximo 10,000 caracteres)
    - **preprocess**: Si aplicar preprocesamiento (limpieza, lematización, stopwords)
    - **analysis_type**: Tipo de análisis ("sentiment", "subjectivity", "both")
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analizador no está listo")
    
    try:
        # Realizar análisis completo
        result = analyzer.analyze(request.text, request.preprocess)
        
        # Filtrar resultados según el tipo de análisis solicitado
        if request.analysis_type == "sentiment":
            result["subjectivity_score"] = None
        elif request.analysis_type == "subjectivity":
            result["sentiment_score"] = None
            result["sentiment_label"] = "n/a"
        
        result["analysis_type"] = request.analysis_type
        
        return TextAnalysisResponse(**result)
        
    except Exception as e:
        logger.error(f"Error procesando texto: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/analyze/sentiment")
async def analyze_sentiment_only(request: TextAnalysisRequest):
    """
    Analiza solo la emocionalidad del texto (endpoint específico)
    """
    request.analysis_type = "sentiment"
    return await analyze_text(request)

@app.post("/analyze/subjectivity")
async def analyze_subjectivity_only(request: TextAnalysisRequest):
    """
    Analiza solo la subjetividad del texto (endpoint específico)
    """
    request.analysis_type = "subjectivity"
    return await analyze_text(request)

# Endpoint para análisis rápido sin preprocesamiento
@app.post("/analyze/quick", response_model=TextAnalysisResponse)
async def analyze_quick(request: TextAnalysisRequest):
    """
    Análisis rápido sin preprocesamiento (más rápido pero menos preciso)
    """
    request.preprocess = False
    return await analyze_text(request)

# Endpoint para detectar sentimientos negativos específicamente
@app.post("/detect/negative")
async def detect_negative_sentiment(request: TextAnalysisRequest):
    """
    Detecta específicamente si el texto tiene sentimientos negativos
    Retorna True si el sentimiento es negativo o muy negativo
    """
    result = await analyze_text(request)
    
    is_negative = result.sentiment_label in ["negativo", "muy_negativo"]
    
    return {
        "is_negative": is_negative,
        "sentiment_score": result.sentiment_score,
        "sentiment_label": result.sentiment_label,
        "confidence": abs(result.sentiment_score - 3.0) / 2.0 if result.sentiment_score else 0
    }

if __name__ == "__main__":
    # Configuración del servidor
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
