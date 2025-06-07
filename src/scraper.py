import requests
from bs4 import BeautifulSoup
from utils import validar_seccion
import pandas as pd
import time
import os
import json
import signal

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

# -------- Manejador de timeout con señal --------
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout en BeautifulSoup")

signal.signal(signal.SIGALRM, timeout_handler)

# ------------------------------------------------
def get_links_de_seccion(seccion, max_paginas, starting_page=0):
    links = []
    last_page = starting_page + max_paginas - 1
    for i in range(starting_page, last_page):
        if(i == last_page-2):
            break
        url = f"{BASE_URL}/secciones/{seccion}?page={i}"
        print(f"Scraping: {url}")
        try:
            resp = requests.get(url, timeout=10)
            signal.alarm(3)  # máximo 5 segundos para procesar el HTML
            soup = BeautifulSoup(resp.text, "html.parser")
            articulos = soup.find_all("article")
            signal.alarm(0)  # cancelar alarma si fue todo bien
        except TimeoutException:
            print(f"Timeout al procesar HTML de {url}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"Error al cargar página {url}: {e}")
            continue

        for articulo in articulos:
            if articulo.find("div", class_="exclusive-teaser"):
                continue
            a_tag = articulo.find("a")
            if a_tag and "href" in a_tag.attrs:
                link = BASE_URL + a_tag["href"]
                links.append(link)
        time.sleep(1)

    return links

def scrapear_articulo(url):
    try:
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")

        titulo = soup.find("h1").get_text(strip=True)
        cuerpo = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])
        cuerpo = cuerpo.replace("\n", " ").replace("\r", " ")

        if len(cuerpo) < 50:
            return None

        autor_tag = soup.find("div", class_="author-name")
        autor = autor_tag.get_text(strip=True).replace("Por", "").strip() if autor_tag else "Desconocido"

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
            os.makedirs("data/new", exist_ok=True)
            nombre_archivo = f"data/new/articulos_{SECCION}.csv"
            df.to_csv(
                nombre_archivo,
                mode='a',
                header=not os.path.exists(nombre_archivo),
                index=False,
                encoding="utf-8",
                quotechar='"'
            )
            print(f"Guardados {len(df)} artículos en '{nombre_archivo}'")
        else:
            print("No se guardaron artículos.")