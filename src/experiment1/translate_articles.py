import os
import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator

SECCIONES = [
    "economia",
    "el-mundo",
    "el-pais",
    "negrx",
    "sociedad"
]

TRADUCIR_COLS = ["titulo", "contenido"]
INPUT_DIR = "data/new"
OUTPUT_DIR = "data/translated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

translator = GoogleTranslator(source="auto", target="en")

def traducir_texto(texto):
    try:
        return translator.translate(texto)
    except:
        return None

def traducir_csv(seccion):
    archivo_entrada = os.path.join(INPUT_DIR, f"articulos_{seccion}.csv")
    archivo_salida = os.path.join(OUTPUT_DIR, f"traducido_articulos_{seccion}.csv")

    if not os.path.exists(archivo_entrada):
        print(f"Archivo no encontrado: {archivo_entrada}")
        return

    df = pd.read_csv(archivo_entrada)

    for col in TRADUCIR_COLS:
        if col in df.columns:
            print(f"Traduciendo columna '{col}' de la sección {seccion}...")
            df[f"{col}_en"] = [traducir_texto(t) for t in tqdm(df[col].fillna(""))]

    df.to_csv(archivo_salida, index=False, encoding="utf-8")
    print(f"Archivo traducido guardado en {archivo_salida}")

if __name__ == "__main__":
    for seccion in SECCIONES:
        traducir_csv(seccion)