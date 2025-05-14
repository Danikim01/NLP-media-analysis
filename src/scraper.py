import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

BASE_URL = "https://www.pagina12.com.ar"
SECCION = "el-pais"
MAX_PAGINAS = 2
ARTICULOS = []

def get_links_de_seccion(seccion, max_paginas):
    links = []
    for i in range(max_paginas):
        url = f"{BASE_URL}/secciones/{seccion}?page={i}"
        print(f"Scrapeando: {url}")
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        for articulo in soup.find_all("article"):
            # Saltear exclusivos para socios
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
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        titulo = soup.find("h1").get_text(strip=True)

        # Cuerpo: concatenar párrafos
        cuerpo = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])
        cuerpo = cuerpo.replace("\n", " ").replace("\r", " ")
        # cuerpo = cuerpo.replace('"', "'")  # reemplaza comillas dobles internas por comillas simples
        if len(cuerpo) < 100:  # umbral para evitar notas vacías o mal parseadas
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
    links = get_links_de_seccion(SECCION, MAX_PAGINAS)

    for link in links:
        articulo = scrapear_articulo(link)
        if articulo:
            ARTICULOS.append(articulo)

    df = pd.DataFrame(ARTICULOS)

    # Asegurarse de que el directorio existe
    os.makedirs("data/raw", exist_ok=True)

    # Guardar a CSV con codificación segura
    df.to_csv("data/raw/pagina12_articulos.csv", index=False, encoding="utf-8", quotechar='"')
    print(f"Guardados {len(df)} artículos en 'data/raw/pagina12_articulos.csv'")