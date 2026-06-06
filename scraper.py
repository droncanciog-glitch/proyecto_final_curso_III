"""
scraper.py
----------
Adquisición de artículos nuevos de la revista Q1 'Machine Learning and
Knowledge Extraction' (MDPI, ISSN 2504-4990) para actualizar la base SQLite.

ESTRATEGIA (Selenium con respaldo de APIs):
    1) Selenium (navegador real): recorre las páginas de los issues del
       Volumen 8 (2026) en MDPI, igual que el Taller 1 hizo en R con
       read_html_live. Extrae título, autores, fecha, DOI, resumen y, sobre
       todo, el número de VISTAS ("Viewed by"), que solo está en MDPI.
       Funciona en local (donde hay navegador Chrome instalado).

    2) APIs de respaldo (Crossref + OpenAlex): si Selenium no está disponible
       (p. ej. en Streamlit Cloud, que no tiene navegador) o falla, se usan
       estas APIs públicas. No traen vistas, pero sí citas y metadatos.

    Así la app funciona en los dos entornos: completa en local, parcial online.
"""

import re
import time
import sqlite3
from datetime import datetime

import requests

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
ISSN = "2504-4990"  # Machine Learning and Knowledge Extraction
JOURNAL_NAME = "Machine Learning and Knowledge Extraction"

# Crossref recomienda identificarse con un correo en el User-Agent ("polite pool")
CONTACT_EMAIL = "estudiante@unal.edu.co"  # <-- cámbialo por tu correo real
HEADERS = {"User-Agent": f"MineriaDatos-Taller2/1.0 (mailto:{CONTACT_EMAIL})"}
TIMEOUT = 30

# Palabras clave para clasificar el tema (igual criterio que el Taller 1)
TOPIC_KEYWORDS = {
    "IA Generativa": [
        "generative", "gpt", "llm", "large language model", "diffusion",
        "transformer", "text-to-image", "gan", "chatbot", "prompt",
    ],
    "Estadística": [
        "statistic", "bayesian", "regression", "inference", "probabil",
        "hypothesis", "variance", "distribution", "sampling",
    ],
    "Machine Learning": [
        "learning", "neural", "classification", "clustering", "model",
        "prediction", "deep", "network", "feature",
    ],
}


# ---------------------------------------------------------------------------
# Utilidades de base de datos
# ---------------------------------------------------------------------------
def classify_topic(text: str) -> str:
    t = (text or "").lower()
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return topic
    return "Machine Learning"


def get_existing_dois(db_path: str) -> set:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute("SELECT doi FROM papers WHERE doi IS NOT NULL").fetchall()
    finally:
        con.close()
    return {r[0].strip().lower() for r in rows if r[0]}


def get_max_paper_id(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    try:
        r = con.execute("SELECT MAX(paper_id) FROM papers").fetchone()
    finally:
        con.close()
    return r[0] or 0


def get_last_n_papers(db_path: str, n: int = 5):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT paper_id, doi, url, title, citations, views "
            "FROM papers ORDER BY paper_id DESC LIMIT ?",
            (n,),
        ).fetchall()
    finally:
        con.close()
    return [dict(r) for r in rows]


def insert_paper(db_path: str, paper: dict, paper_id: int):
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO papers
            (paper_id, journal_name, title, publication_date, year, doi, url,
             abstract, authors_raw, n_authors, citations, downloads, views,
             n_references, topic_label)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                paper_id, paper["journal_name"], paper["title"],
                paper["publication_date"], paper["year"], paper["doi"],
                paper["url"], paper["abstract"], paper["authors_raw"],
                paper["n_authors"], paper["citations"], paper["downloads"],
                paper["views"], paper["n_references"], paper["topic_label"],
            ),
        )
        con.commit()
    finally:
        con.close()


def update_metrics(db_path: str, doi: str, citations):
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "UPDATE papers SET citations = COALESCE(?, citations) WHERE doi = ?",
            (citations, doi),
        )
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Fuente principal: Crossref
# ---------------------------------------------------------------------------
def _crossref_item_to_paper(item: dict) -> dict | None:
    """Convierte un item de Crossref al formato de la tabla papers."""
    doi = item.get("DOI")
    titles = item.get("title") or []
    title = titles[0] if titles else None
    if not doi and not title:
        return None

    # Autores
    authors = item.get("author") or []
    nombres = []
    for a in authors:
        given = a.get("given", "")
        family = a.get("family", "")
        full = f"{given} {family}".strip()
        if full:
            nombres.append(full)
    authors_raw = "; ".join(nombres) if nombres else None
    n_authors = float(len(nombres)) if nombres else None

    # Fecha de publicación: published -> date-parts [[YYYY, MM, DD]]
    pub_date, year = None, None
    dp = (item.get("published") or item.get("published-online")
          or item.get("published-print") or {}).get("date-parts")
    if dp and dp[0]:
        parts = dp[0]
        year = parts[0] if len(parts) >= 1 else None
        try:
            if len(parts) >= 3:
                pub_date = datetime(parts[0], parts[1], parts[2]).strftime("%d %b %Y")
            elif len(parts) == 2:
                pub_date = datetime(parts[0], parts[1], 1).strftime("%b %Y")
            else:
                pub_date = str(parts[0])
        except (ValueError, TypeError):
            pub_date = str(parts[0])

    abstract = item.get("abstract")
    if abstract:
        # Crossref a veces devuelve el abstract con etiquetas JATS; limpieza simple
        import re as _re
        abstract = _re.sub(r"<[^>]+>", "", abstract).strip()

    n_references = item.get("reference-count")
    url = f"https://doi.org/{doi}" if doi else None
    topic = classify_topic(f"{title or ''} {abstract or ''}")

    return {
        "journal_name": JOURNAL_NAME,
        "title": title,
        "publication_date": pub_date,
        "year": year,
        "doi": doi,
        "url": url,
        "abstract": abstract,
        "authors_raw": authors_raw,
        "n_authors": n_authors,
        "citations": item.get("is-referenced-by-count", 0) or 0,
        "downloads": None,
        "views": None,
        "n_references": n_references,
        "topic_label": topic,
    }


# ---------------------------------------------------------------------------
# Fuente principal: Selenium (navegador real, igual que read_html_live en R)
# ---------------------------------------------------------------------------
# Volumen 8 = 2026. Issues 1 a 6 según la portada de la revista.
VOLUMEN_2026 = 8
ISSUES_2026 = [1, 2, 3, 4, 5, 6]


def selenium_disponible() -> bool:
    """Comprueba si Selenium y un navegador están instalados."""
    try:
        import selenium  # noqa: F401
        return True
    except ImportError:
        return False


def _crear_driver():
    """Crea un navegador Chrome headless. Lanza excepción si no se puede."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    return webdriver.Chrome(options=opts)


def _parsear_nodo(nodo) -> dict | None:
    """
    Extrae los datos de un nodo de artículo (div.generic-item.article-item),
    replicando la función extraer_articulo() del Taller 1 en R.
    """
    from selenium.webdriver.common.by import By

    def _texto(sel):
        try:
            return nodo.find_element(By.CSS_SELECTOR, sel).text.strip()
        except Exception:
            return None

    def _attr(sel, attr):
        try:
            return nodo.find_element(By.CSS_SELECTOR, sel).get_attribute(attr)
        except Exception:
            return None

    titulo = _texto("a.title-link")
    if not titulo:
        return None

    # Autores
    try:
        autores_els = nodo.find_elements(By.CSS_SELECTOR, "div.authors strong")
        autores = "; ".join(a.text.strip() for a in autores_els if a.text.strip())
    except Exception:
        autores = None
    n_authors = float(len(autores.split(";"))) if autores else None

    # DOI y URL
    href_doi = _attr("a[href*='doi.org']", "href")
    doi = href_doi.replace("https://doi.org/", "") if href_doi else None
    href_art = _attr("a.title-link", "href")
    art_url = ("https://www.mdpi.com" + href_art) if href_art and href_art.startswith("/") else href_art

    # Texto completo del nodo, para extraer vistas, citas y fecha por regex
    texto_completo = nodo.text or ""

    m_views = re.search(r"Viewed by ([\d,]+)", texto_completo)
    views = int(m_views.group(1).replace(",", "")) if m_views else None

    m_cited = re.search(r"Cited by (\d+)", texto_completo)
    citations = int(m_cited.group(1)) if m_cited else 0

    m_fecha = re.search(r"(\d{1,2} \w+ \d{4})", texto_completo)
    pub_date = m_fecha.group(1) if m_fecha else None
    year = None
    if pub_date:
        my = re.search(r"(\d{4})", pub_date)
        if my:
            year = int(my.group(1))

    abstract = _texto(".abstract-full")
    if abstract:
        abstract = re.sub(r"Full article$", "", abstract).strip()

    topic = classify_topic(f"{titulo} {abstract or ''}")

    return {
        "journal_name": JOURNAL_NAME,
        "title": titulo,
        "publication_date": pub_date,
        "year": year,
        "doi": doi,
        "url": art_url,
        "abstract": abstract,
        "authors_raw": autores,
        "n_authors": n_authors,
        "citations": citations,
        "downloads": None,
        "views": views,
        "n_references": None,
        "topic_label": topic,
    }


def fetch_selenium(issues=None, progress_callback=None) -> list[dict]:
    """
    Recorre los issues del Volumen 8 (2026) con un navegador real y extrae los
    artículos, incluyendo VISTAS y CITAS que solo están en la página de MDPI.

    Lanza excepción si Selenium/Chrome no están disponibles (la maneja la
    función principal para caer al respaldo de APIs).
    """
    from selenium.webdriver.common.by import By

    if issues is None:
        issues = ISSUES_2026

    driver = _crear_driver()
    papers = []
    try:
        for idx, issue in enumerate(issues):
            url = f"https://www.mdpi.com/{ISSN}/{VOLUMEN_2026}/{issue}"
            if progress_callback:
                progress_callback(idx + 1, len(issues), url)
            driver.get(url)
            time.sleep(4)  # esperar a que cargue el JS (como Sys.sleep en R)
            nodos = driver.find_elements(
                By.CSS_SELECTOR, "div.generic-item.article-item"
            )
            for nodo in nodos:
                p = _parsear_nodo(nodo)
                if p and p.get("doi"):
                    papers.append(p)
    finally:
        driver.quit()
    return papers


def fetch_crossref(from_date: str = "2026-01-01", rows: int = 200) -> list[dict]:
    """
    Trae artículos de la revista (por ISSN) publicados desde from_date.
    Devuelve lista de papers en el formato de la tabla. Usa paginación por cursor.
    """
    base = f"https://api.crossref.org/journals/{ISSN}/works"
    papers = []
    cursor = "*"
    while True:
        params = {
            "filter": f"from-pub-date:{from_date}",
            "rows": min(rows, 100),
            "cursor": cursor,
            "select": ("DOI,title,published,published-online,published-print,"
                       "author,reference-count,abstract,is-referenced-by-count"),
        }
        resp = requests.get(base, params=params, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        msg = resp.json()["message"]
        items = msg.get("items", [])
        for it in items:
            p = _crossref_item_to_paper(it)
            if p:
                papers.append(p)
        next_cursor = msg.get("next-cursor")
        if not items or not next_cursor or len(papers) >= rows:
            break
        cursor = next_cursor
        time.sleep(0.3)
    return papers


# ---------------------------------------------------------------------------
# Respaldo: OpenAlex
# ---------------------------------------------------------------------------
def fetch_openalex(from_date: str = "2026-01-01", per_page: int = 200) -> list[dict]:
    """Respaldo si Crossref falla. OpenAlex también es pública y sin key."""
    base = "https://api.openalex.org/works"
    flt = f"primary_location.source.issn:{ISSN},from_publication_date:{from_date}"
    params = {"filter": flt, "per-page": min(per_page, 200),
              "mailto": CONTACT_EMAIL}
    resp = requests.get(base, params=params, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json().get("results", [])
    papers = []
    for w in data:
        doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
        title = w.get("title")
        if not doi and not title:
            continue
        authorships = w.get("authorships", [])
        nombres = [a.get("author", {}).get("display_name", "") for a in authorships]
        nombres = [n for n in nombres if n]
        pub_date = w.get("publication_date")
        pretty_date = None
        year = w.get("publication_year")
        if pub_date:
            try:
                pretty_date = datetime.strptime(pub_date, "%Y-%m-%d").strftime("%d %b %Y")
            except ValueError:
                pretty_date = pub_date
        topic = classify_topic(title or "")
        papers.append({
            "journal_name": JOURNAL_NAME,
            "title": title,
            "publication_date": pretty_date,
            "year": year,
            "doi": doi,
            "url": f"https://doi.org/{doi}" if doi else None,
            "abstract": None,
            "authors_raw": "; ".join(nombres) if nombres else None,
            "n_authors": float(len(nombres)) if nombres else None,
            "citations": w.get("cited_by_count", 0) or 0,
            "downloads": None,
            "views": None,
            "n_references": w.get("referenced_works_count"),
            "topic_label": topic,
        })
    return papers


# ---------------------------------------------------------------------------
# Función principal usada por el dashboard
# ---------------------------------------------------------------------------
def buscar_nuevos_articulos(
    db_path: str,
    from_date: str = "2026-01-01",
    progress_callback=None,
) -> dict:
    """
    Busca artículos nuevos vía Crossref (con respaldo OpenAlex) e inserta los
    que no estén en la BD.

    Devuelve dict con: nuevos, n_nuevos, revisados, fuente, error, log.
    """
    log = []
    existing = get_existing_dois(db_path)
    next_id = get_max_paper_id(db_path) + 1
    nuevos = []
    fuente = None
    error = None

    # 1) Intentar Selenium (trae vistas y citas reales de MDPI).
    #    Si no hay navegador o falla, caer a las APIs (Crossref -> OpenAlex).
    candidatos = []
    if selenium_disponible():
        try:
            log.append("Intentando scraping con navegador (Selenium)...")
            candidatos = fetch_selenium(progress_callback=progress_callback)
            fuente = "Selenium (MDPI)"
            log.append(f"  Selenium extrajo {len(candidatos)} artículo(s) de MDPI.")
        except Exception as e:
            log.append(f"  Selenium no disponible o falló ({e}). Usando APIs...")
            candidatos = []
    else:
        log.append("Selenium no instalado. Usando APIs de metadatos...")

    # Respaldo: si Selenium no trajo nada, usar Crossref y luego OpenAlex
    if not candidatos:
        try:
            log.append("Consultando Crossref API...")
            candidatos = fetch_crossref(from_date=from_date)
            fuente = "Crossref"
            log.append(f"  Crossref devolvió {len(candidatos)} artículo(s) desde {from_date}.")
        except Exception as e:
            log.append(f"  Crossref falló ({e}). Probando OpenAlex...")
            try:
                candidatos = fetch_openalex(from_date=from_date)
                fuente = "OpenAlex"
                log.append(f"  OpenAlex devolvió {len(candidatos)} artículo(s).")
            except Exception as e2:
                error = f"Ambas APIs fallaron. Crossref: {e}. OpenAlex: {e2}."
                log.append(f"  ⚠️ {error}")

    # 2) Insertar los que no existan
    total = len(candidatos)
    for i, paper in enumerate(candidatos):
        if progress_callback:
            progress_callback(i + 1, total, paper.get("doi") or "")
        doi = paper.get("doi")
        if not doi:
            continue
        if doi.strip().lower() in existing:
            continue
        insert_paper(db_path, paper, next_id)
        existing.add(doi.strip().lower())
        nuevos.append(paper)
        log.append(f"Nuevo: {(paper['title'] or '')[:60]}... ({doi})")
        next_id += 1

    resultado = {
        "nuevos": nuevos,
        "n_nuevos": len(nuevos),
        "revisados": [],
        "fuente": fuente,
        "error": error,
        "log": log,
    }

    # 3) Si no hubo nuevos (y no hubo error), re-verificar los últimos 5 por DOI
    if not nuevos and not error:
        log.append("No se encontraron artículos nuevos. Verificando los últimos 5 vía Crossref...")
        for p in get_last_n_papers(db_path, 5):
            doi = p.get("doi")
            if not doi:
                continue
            try:
                r = requests.get(f"https://api.crossref.org/works/{doi}",
                                 headers=HEADERS, timeout=TIMEOUT)
                if r.ok:
                    cites = r.json()["message"].get("is-referenced-by-count")
                    if cites is not None:
                        update_metrics(db_path, doi, cites)
                    resultado["revisados"].append(
                        {"title": p["title"], "doi": doi, "citas_actualizadas": cites}
                    )
                    log.append(f"Verificado: {(p['title'] or '')[:55]}... (citas: {cites})")
            except Exception:
                pass
            time.sleep(0.3)

    resultado["log"] = log
    return resultado


if __name__ == "__main__":
    r = buscar_nuevos_articulos("make_q1_2025.sqlite")
    print("Fuente:", r["fuente"], "| Nuevos:", r["n_nuevos"])
    for line in r["log"][:12]:
        print(" -", line)
