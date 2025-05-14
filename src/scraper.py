import requests
from bs4 import BeautifulSoup
from utils import validar_seccion
import pandas as pd
import time
import os
import json

BASE_URL = "https://www.pagina12.com.ar"
ARTICULOS = []
CONFIG_PATH = "configs/scraper_config.json"

with open(CONFIG_PATH) as f:
    config = json.load(f)

SECCION = config["section"]
MAX_PAGINAS = config["max_pages"]
STARTING_PAGE = config.get("starting_page", 0)

if not validar_seccion(SECCION):
    raise ValueError(f"'{SECCION}' no es una sección válida.")

def get_links_de_seccion(seccion, max_paginas, starting_page=0):
    links = []
    for i in range(starting_page, starting_page + max_paginas - 1):
        url = f"{BASE_URL}/secciones/{seccion}?page={i}"
        print(f"Scraping: {url}")
        try:
            print(f"Scraping página {i}... before request")
            resp = requests.get(url, timeout=10)
            print(f"Scraping página {i}... after request")
            soup = BeautifulSoup(resp.text, "html.parser")
            articulos = soup.find_all("article")
        except requests.exceptions.RequestException as e:
            print(f"Error al cargar página {url}: {e}")
            continue

        for articulo in articulos:
            # Saltear exclusivos para socios
            if articulo.find("div", class_="exclusive-teaser"):
                continue

            a_tag = articulo.find("a")
            if a_tag and "href" in a_tag.attrs:
                link = BASE_URL + a_tag["href"]
                links.append(link)
        time.sleep(1)

        # Pausa para evitar bloqueos
        if (i - starting_page + 1) % 15 == 0:
            print("Pausa larga para evitar bloqueos...")
            time.sleep(5)
    return links

def scrapear_articulo(url):
    try:
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        titulo = soup.find("h1").get_text(strip=True)

        # Cuerpo: concatenar párrafos
        cuerpo = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])
        cuerpo = cuerpo.replace("\n", " ").replace("\r", " ")
        # cuerpo = cuerpo.replace('"', "'")  # reemplaza comillas dobles internas por comillas simples
        if len(cuerpo) < 50:  # umbral para evitar notas vacías o mal parseadas
            return None

        # Autor
        autor_tag = soup.find("div", class_="author-name")
        if autor_tag:
            autor = autor_tag.get_text(strip=True)
            autor = autor.replace("Por", "").strip()
        else:
            autor = "Desconocido"

        # Fecha
        time_tag = soup.find("time")
        fecha = time_tag["datetime"] if time_tag and "datetime" in time_tag.attrs else "Sin fecha"

        return {
            "titulo": titulo,
            "contenido": cuerpo,
            "autor": autor,
            "fecha": fecha,
            "url": url
        }

    except Exception as e:
        print(f"Error al scrapear {url}: {e}")
        return None

if __name__ == "__main__":
    STARTING_PAGE = config.get("starting_page", 0)
    links = []

    try:
        links = get_links_de_seccion(SECCION, MAX_PAGINAS, STARTING_PAGE)
        for link in links:
            articulo = scrapear_articulo(link)
            if articulo:
                ARTICULOS.append(articulo)
    except KeyboardInterrupt:
        print("Scraping interrumpido manualmente.")
    finally:
        if ARTICULOS:
            df = pd.DataFrame(ARTICULOS)
            os.makedirs("data/raw", exist_ok=True)
            nombre_archivo = f"data/raw/articulos_{SECCION}.csv"
            df.to_csv(nombre_archivo, index=False, encoding="utf-8", quotechar='"')
            print(f"Guardados {len(df)} artículos en '{nombre_archivo}'")
        else:
            print("No se guardaron artículos.")