"""
Script de prueba para la API de análisis de sentimientos
"""

import requests
import json
import time

# URL base de la API (ajustar si cambias el puerto)
BASE_URL = "http://localhost:8000"

def test_health():
    """Prueba el endpoint de salud"""
    print("🔍 Probando endpoint de salud...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_analyze_sentiment():
    """Prueba el análisis de sentimientos"""
    print("😊 Probando análisis de sentimientos...")
    
    test_texts = [
        "Me encanta este producto, es fantástico!",
        "Este servicio es terrible, muy malo",
        "El clima está normal hoy",
        "Estoy muy preocupado por la situación económica del país",
        "¡Qué día tan hermoso! Todo va perfecto",
        "No me gusta nada esta situación, es horrible"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Texto {i} ---")
        print(f"Original: {text}")
        
        payload = {
            "text": text,
            "preprocess": True,
            "analysis_type": "both"
        }
        
        response = requests.post(f"{BASE_URL}/analyze", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Sentimiento: {result['sentiment_score']} ({result['sentiment_label']})")
            print(f"✅ Subjetividad: {result['subjectivity_score']}")
            print(f"✅ Texto procesado: {result['processed_text'][:100]}...")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
        
        time.sleep(0.5)  # Pausa entre requests

def test_detect_negative():
    """Prueba la detección específica de sentimientos negativos"""
    print("\n🚨 Probando detección de sentimientos negativos...")
    
    negative_texts = [
        "Esto es terrible",
        "No me gusta nada",
        "Es horrible",
        "Muy malo"
    ]
    
    positive_texts = [
        "Me encanta",
        "Es fantástico",
        "Muy bueno",
        "Excelente"
    ]
    
    print("\n--- Textos Negativos ---")
    for text in negative_texts:
        payload = {"text": text}
        response = requests.post(f"{BASE_URL}/detect/negative", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"'{text}' -> Negativo: {result['is_negative']} (Score: {result['sentiment_score']})")
    
    print("\n--- Textos Positivos ---")
    for text in positive_texts:
        payload = {"text": text}
        response = requests.post(f"{BASE_URL}/detect/negative", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"'{text}' -> Negativo: {result['is_negative']} (Score: {result['sentiment_score']})")

def test_quick_analysis():
    """Prueba el análisis rápido sin preprocesamiento"""
    print("\n⚡ Probando análisis rápido...")
    
    text = "Este es un texto de prueba para análisis rápido"
    payload = {
        "text": text,
        "preprocess": False,
        "analysis_type": "sentiment"
    }
    
    response = requests.post(f"{BASE_URL}/analyze/quick", json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Análisis rápido completado")
        print(f"   Sentimiento: {result['sentiment_score']} ({result['sentiment_label']})")
        print(f"   Sin preprocesamiento: {result['processed_text'] == text}")
    else:
        print(f"❌ Error: {response.status_code}")

if __name__ == "__main__":
    print("🚀 Iniciando pruebas de la API de Análisis de Sentimientos")
    print("=" * 60)
    
    try:
        # Probar que el servidor esté funcionando
        test_health()
        
        # Ejecutar todas las pruebas
        test_analyze_sentiment()
        test_detect_negative()
        test_quick_analysis()
        
        print("\n" + "=" * 60)
        print("✅ Todas las pruebas completadas!")
        print("\n📖 Documentación disponible en:")
        print(f"   - Swagger UI: {BASE_URL}/docs")
        print(f"   - ReDoc: {BASE_URL}/redoc")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se puede conectar al servidor")
        print("   Asegúrate de que el servidor esté ejecutándose en http://localhost:8000")
        print("   Ejecuta: python api_server.py")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
