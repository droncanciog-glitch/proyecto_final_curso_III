"""
app.py
------
Aplicación única en Streamlit que integra un dashboard sobre la misma base de
datos SQLite de revistas Q1 (revista 'Machine Learning and Knowledge Extraction',
MDPI) construida en el Taller 1. La aplicación tiene tres pestañas:

  Pestaña 1 — Información general de la revista y del web scraping:
      descripción de la revista MAKE (Q1, MDPI, ISSN 2504-4990), explicación
      de cómo se organizan los volúmenes e issues, y conteo de artículos
      almacenados por volumen e issue.

  Pestaña 2 — Dashboard:
      conexión SQLite, sidebar con filtros, widgets, indicadores,
      visualizaciones Plotly, tabla interactiva y scraping de nuevos artículos.

  Pestaña 3 — Clustering no supervisado:
      segmentación de artículos con K-means vs jerárquico, codo + silhouette,
      perfilado de clusters, proyección PCA y conclusiones.
"""

import re
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

import scraper  # módulo de scraping (botón de actualización)

DB_PATH = "make_q1_2025.sqlite"

st.set_page_config(
    page_title="Revista MAKE (Machine Learning and Knowledge Extraction) — Análisis y Clustering",
    page_icon="📊",
    layout="wide",
)

import os


# ===========================================================================
# CAPA DE DATOS
# ===========================================================================
@st.cache_data(show_spinner=False)
def load_papers(_db_mtime: float) -> pd.DataFrame:
    """
    Lee la tabla papers. La llave del caché es la fecha de modificación del
    archivo .sqlite (_db_mtime): si el archivo cambia (p. ej. tras un scraping),
    el caché se invalida automáticamente y se releen los datos.
    """
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM papers", con)
    finally:
        con.close()

    df["fecha"] = pd.to_datetime(
        df["publication_date"], format="%d %b %Y", errors="coerce"
    )
    # downloads viene vacío en la BD -> usamos views como métrica de popularidad
    df["popularidad"] = df["downloads"].fillna(df["views"])
    df["citations"] = pd.to_numeric(df["citations"], errors="coerce").fillna(0)
    df["n_authors"] = pd.to_numeric(df["n_authors"], errors="coerce")
    df["n_references"] = pd.to_numeric(df["n_references"], errors="coerce")
    return df


def db_mtime() -> float:
    """Fecha de última modificación del archivo .sqlite (llave del caché)."""
    try:
        return os.path.getmtime(DB_PATH)
    except OSError:
        return 0.0


def contar_papers_en_bd() -> int:
    """Cuenta filas leyendo el archivo directamente (sin caché)."""
    con = sqlite3.connect(DB_PATH)
    try:
        return con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    finally:
        con.close()


def first_author(authors_raw: str) -> str:
    if not isinstance(authors_raw, str) or not authors_raw.strip():
        return ""
    return authors_raw.split(";")[0].strip()


df = load_papers(db_mtime())


# ===========================================================================
# SIDEBAR — FILTROS (compartido, afecta al Dashboard)
# ===========================================================================
st.sidebar.title("🔎 Filtros")
st.sidebar.caption("Proceso KDD: selección y filtrado de datos")

fechas_validas = df["fecha"].dropna()
if not fechas_validas.empty:
    fmin = fechas_validas.min().date()
    fmax = fechas_validas.max().date()
    rango = st.sidebar.date_input(
        "📅 Rango de fechas",
        value=(fmin, fmax),
        min_value=fmin,
        max_value=fmax,
    )
else:
    rango = None

temas = sorted(df["topic_label"].dropna().unique().tolist())
tema_sel = st.sidebar.multiselect("🏷️ Tema / categoría", temas, default=temas)
autor_query = st.sidebar.text_input("👤 Autor (contiene)", "")
doi_query = st.sidebar.text_input("🔗 DOI (contiene)", "")
kw_query = st.sidebar.text_input("🔍 Palabra clave en título / abstract", "")

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Actualización de datos")
st.sidebar.caption(f"Artículos en la base: **{contar_papers_en_bd()}**")
buscar = st.sidebar.button("Buscar artículos nuevos (2026)", type="primary")
if st.sidebar.button("↻ Recargar datos de la base"):
    load_papers.clear()
    st.rerun()


def aplicar_filtros(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    if rango and isinstance(rango, (tuple, list)) and len(rango) == 2:
        ini, fin = rango
        out = out[
            (out["fecha"].isna())
            | ((out["fecha"].dt.date >= ini) & (out["fecha"].dt.date <= fin))
        ]
    if tema_sel:
        out = out[out["topic_label"].isin(tema_sel)]
    if autor_query.strip():
        out = out[out["authors_raw"].str.contains(autor_query, case=False, na=False)]
    if doi_query.strip():
        out = out[out["doi"].str.contains(doi_query, case=False, na=False)]
    if kw_query.strip():
        mask = (
            out["title"].str.contains(kw_query, case=False, na=False)
            | out["abstract"].str.contains(kw_query, case=False, na=False)
        )
        out = out[mask]
    return out


df_f = aplicar_filtros(df)


# ===========================================================================
# ENCABEZADO + PESTAÑAS
# ===========================================================================
st.title("📊 Revista MAKE (Machine Learning and Knowledge Extraction) — Análisis y Clustering")
st.markdown(
    "Artículos de **Machine Learning**, **IA Generativa** y **Estadística** "
    "almacenados en SQLite."
)
st.markdown(
"By: Daniela Roncancio Gomez"
)

tab_info, tab_dash, tab_cluster = st.tabs([
    "📖 Información",
    "📈 Dashboard",
    "🧩 Clustering",
])


# ===========================================================================
# PESTAÑA 0 — INFORMACIÓN DE LA REVISTA
# ===========================================================================
with tab_info:
    st.subheader("📖 Sobre la revista y los datos")
 
    st.markdown(
        "**Machine Learning and Knowledge Extraction (MAKE)** es una revista "
        "científica de acceso abierto publicada por **MDPI**, clasificada en el "
        "**cuartil Q1**. Se especializa en aprendizaje automático, extracción de "
        "conocimiento, inteligencia artificial y temas afines. Su ISSN es "
        "**2504-4990** y publica de forma continua a lo largo del año.\n\n"
        "En **MDPI**, cada año corresponde a un **volumen**, y cada volumen se "
        "divide en varios **issues** (números), normalmente uno por mes. Cada "
        "issue agrupa los artículos publicados en ese periodo."
    )
 
    st.markdown("#### ¿Cómo se hizo la recolección de los datos?") 

    st.markdown(
        "En esta etapa se obtiene información de los artículos publicados en el "
        "año 2025 en la revista científica **Machine Learning and Knowledge "
        "Extraction (MAKE)** (https://www.mdpi.com/2504-4990/7/4). Esta es una "
        "revista de acceso abierto, que cubre investigaciones en áreas como el "
        "aprendizaje automático, la extracción de conocimiento y otras "
        "relacionadas con la inteligencia artificial basada en datos.\n\n"
        "Para el desarrollo de esta fase, se seleccionó el **Volumen 7**, el cual "
        "contiene un total de **61 artículos**. A partir de estos artículos, se "
        "recolectó la siguiente información:"
    )

    st.markdown(
        "- Título de la revista\n"
        "- Título del artículo\n"
        "- Fecha de publicación\n"
        "- Año\n"
        "- DOI\n"
        "- URL del artículo\n"
        "- Autores\n"
        "- Resumen\n"
        "- Número de visualizaciones\n"
        "- Número de páginas\n"
        "- Número de citas\n"
        "- Número de referencias\n"
        "- Referencias"
    )
    
    st.info(
        "**Nota:** Como no se cuenta con la variable *Descargas*, se hace uso del "
        "*Número de visualizaciones* en su defecto."
    )
    st.markdown(
        "La recolección se realizó mediante **web scraping** con **R**, usando "
        "principalmente la librería **rvest** (con `read_html_live` para cargar "
        "páginas dinámicas) junto a **dplyr**, **stringr** y **purrr** para "
        "procesar la información. El proceso siguió estos pasos:"
    )
    st.markdown(
        "1. Se estableció conexión con la página del **Volumen 7, número 4 "
        "(diciembre 2025)** de la revista en MDPI y se verificó que contenía "
        "**61 artículos**.\n"
        "2. Se recorrió cada artículo extrayendo sus metadatos: título, autores, "
        "fecha, DOI, URL, resumen y número de visualizaciones.\n"
        "3. Para complementar la información bibliométrica se consultaron dos APIs "
        "externas: **Semantic Scholar** (conteo de citas y referencias) y "
        "**CrossRef** (texto completo de las referencias).\n"
        "4. Toda la información se consolidó en *dataframes* y se almacenó en una "
        "base de datos **SQLite** mediante los paquetes **DBI** y **RSQLite**."
    )


    st.markdown("#### Actualización: artículos de 2026")
    st.markdown(
        "Además de los datos del Taller 1 (Volumen 7, año 2025), el dashboard "
        "permite **actualizar la base con artículos nuevos** del **Volumen 8 (2026)** "
        "mediante el botón *Buscar artículos nuevos* en el panel lateral. La "
        "actualización combina dos estrategias:"
    )
    st.markdown(
        "- **Selenium (método principal):** abre un navegador real que recorre los "
        "issues del Volumen 8 en el sitio de MDPI y extrae los metadatos de cada "
        "artículo, incluyendo el número de **visualizaciones** y **citas**, datos "
        "que solo están disponibles en la página de la revista.\n"
        "- **APIs de respaldo (Crossref y OpenAlex):** si el navegador no está "
        "disponible (por ejemplo, en un despliegue en la nube) o el sitio bloquea "
        "el acceso, se consultan estas APIs públicas de metadatos académicos. No "
        "aportan visualizaciones, pero sí los metadatos básicos y las citas.\n\n"
        "El sistema compara los **DOI** obtenidos contra los ya almacenados e "
        "inserta únicamente los artículos nuevos en la base **SQLite**. Si no hay "
        "artículos nuevos, verifica los últimos cinco almacenados para actualizar "
        "sus métricas."
    )

    st.markdown(
        "La siguiente tabla resume los **213 artículos** almacenados en la base de "
        "datos, distribuidos por **volumen** (año) e **issue** (número) de la "
        "revista. Los **61 artículos** del **Volumen 7 (2025)** corresponden a la "
        "recolección inicial del Taller 1, mientras que los **139 artículos** del "
        "**Volumen 8 (2026)** (https://www.mdpi.com/2504-4990/8) se incorporaron mediante la actualización automática "
        "con las APIs de Crossref y OpenAlex. El gráfico de barras acompaña la tabla "
        "mostrando visualmente cuántos artículos hay en cada issue."
    )

    # --- Conteo real de artículos por volumen e issue (desde la BD) ---
    def extraer_vol_issue(doi):
        if not isinstance(doi, str):
            return (None, None)
        m = re.search(r"make(\d)(\d{2})\d+", doi)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (None, None)

    info_df = df.copy()
    info_df[["volumen", "issue"]] = info_df["doi"].apply(
        lambda d: pd.Series(extraer_vol_issue(d))
    )

    # Mapa de año por volumen (vol 7 = 2025, vol 8 = 2026, etc.)
    def vol_a_anio(v):
        return 2018 + int(v) if pd.notna(v) else None

    st.markdown("#### Artículos por volumen e issue")

    resumen = (
        info_df.dropna(subset=["volumen", "issue"])
        .groupby(["volumen", "issue"])
        .size()
        .reset_index(name="N° de artículos")
    )
    resumen["volumen"] = resumen["volumen"].astype(int)
    resumen["issue"] = resumen["issue"].astype(int)
    resumen["Año"] = resumen["volumen"].apply(vol_a_anio)
    resumen = resumen.rename(columns={"volumen": "Volumen", "issue": "Issue"})
    resumen = resumen[["Volumen", "Año", "Issue", "N° de artículos"]].sort_values(
        ["Volumen", "Issue"]
    )

    col_t, col_g = st.columns([1, 1])
    with col_t:
        st.dataframe(resumen, use_container_width=True, hide_index=True)
    with col_g:
        fig_info = px.bar(
            resumen,
            x="Issue",
            y="N° de artículos",
            color="Volumen",
            barmode="group",
            title="Artículos por issue y volumen",
        )
        st.plotly_chart(fig_info, use_container_width=True)

    # Totales por volumen
    tot_vol = (
        resumen.groupby(["Volumen", "Año"])["N° de artículos"].sum().reset_index()
    )
    partes = [
        f"Vol. {int(row['Volumen'])} ({int(row['Año'])}): "
        f"{int(row['N° de artículos'])} artículos"
        for _, row in tot_vol.iterrows()
    ]
    resumen_txt = " · ".join(partes)
    st.info(
        f"**Total en la base: {contar_papers_en_bd()} artículos.**  \n{resumen_txt}"
    )


# ===========================================================================
# PESTAÑA 1 — DASHBOARD 
# ===========================================================================
with tab_dash:

    st.markdown(
        "Em esta pertaña se va a explorar, filtrar y visualizar los artículos "
        "almacenados en la base **SQLite**, siguiendo el proceso **KDD** "
        "(Knowledge Discovery in Databases). Incluye filtros por fechas, tema, "
        "autor, DOI y palabras clave. Los indicadores descriptivos muestran el "
        "total de artículos, promedios de autores, citas y referencias, y los "
        "artículos más citados y vistos. Las visualizaciones interactivas "
        "(**Plotly**) cubren la evolución temporal de publicaciones, artículos "
        "por categoría, distribución de citas y top de autores. También incluye "
        "una tabla interactiva descargable en **CSV**."
    )

    # ---- Acción: scraping ----
    if buscar:
        progress = st.progress(0.0, text="Iniciando scraping...")

        def cb(i, total, url):
            progress.progress(i / total, text=f"Revisando {i}/{total}: {url[-30:]}")

        with st.spinner("Buscando artículos nuevos (MDPI / Crossref / OpenAlex)..."):
            try:
                res = scraper.buscar_nuevos_articulos(DB_PATH, progress_callback=cb)
            except Exception as e:
                res = {"n_nuevos": 0, "nuevos": [], "revisados": [],
                       "fuente": None, "error": str(e), "log": [f"Error: {e}"]}
        progress.empty()

        if res.get("error"):
            st.error(
                "⚠️ No se pudieron consultar las APIs de metadatos. "
                "Revisa tu conexión a internet e inténtalo de nuevo."
            )
            with st.expander("Ver log del proceso"):
                st.code("\n".join(res["log"]) or "Sin mensajes.")
        else:
            n = res["n_nuevos"]
            if n > 0:
                st.success(
                    f"✅ Se encontraron y almacenaron {n} artículo(s) nuevo(s) "
                    f"(fuente: {res.get('fuente')}). "
                    f"Total en la base ahora: {contar_papers_en_bd()} artículos."
                )
                st.dataframe(
                    pd.DataFrame(res["nuevos"])[
                        ["title", "doi", "publication_date", "topic_label"]
                    ],
                    use_container_width=True,
                )
                # Invalidar caché para que el dashboard muestre los nuevos datos
                load_papers.clear()
                st.rerun()
            else:
                st.info(
                    f"ℹ️ No se encontraron artículos nuevos (fuente: {res.get('fuente')}). "
                    "Se verificaron los últimos 5 artículos almacenados y se "
                    "actualizaron sus citas si cambiaron."
                )
                if res["revisados"]:
                    st.dataframe(pd.DataFrame(res["revisados"]), use_container_width=True)
            with st.expander("Ver log del proceso"):
                st.code("\n".join(res["log"]) or "Sin mensajes.")

    # ---- Indicadores ----
    st.subheader("📈 Indicadores")
    total_art = len(df_f)
    prom_autores = df_f["n_authors"].mean()
    prom_citas = df_f["citations"].mean()
    prom_refs = df_f["n_references"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total artículos", f"{total_art}")
    c2.metric("Prom. autores", f"{prom_autores:.1f}" if total_art else "—")
    c3.metric("Prom. citas", f"{prom_citas:.1f}" if total_art else "—")
    c4.metric("Prom. referencias", f"{prom_refs:.0f}" if total_art else "—")
    c5.metric("Temáticas", f"{df_f['topic_label'].nunique()}")

    if total_art:
        mas_citado = df_f.loc[df_f["citations"].idxmax()]
        mas_popular = (
            df_f.loc[df_f["popularidad"].idxmax()]
            if df_f["popularidad"].notna().any() else None
        )
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**🏆 Artículo más citado**")
            st.write(f"{mas_citado['title']}")
            st.caption(f"{int(mas_citado['citations'])} citas · {mas_citado['topic_label']}")
        with cc2:
            if mas_popular is not None:
                st.markdown("**👁️ Artículo más visto / descargado**")
                st.write(f"{mas_popular['title']}")
                st.caption(f"{int(mas_popular['popularidad'])} vistas · {mas_popular['topic_label']}")

    st.markdown("---")

    # ---- Visualizaciones ----
    st.subheader("📊 Visualizaciones")
    g1, g2 = st.columns(2)

    with g1:
        serie = (
            df_f.dropna(subset=["fecha"])
            .assign(mes=lambda d: d["fecha"].dt.to_period("M").dt.to_timestamp())
            .groupby("mes").size().reset_index(name="publicaciones")
        )
        if not serie.empty:
            fig1 = px.line(serie, x="mes", y="publicaciones", markers=True,
                           title="Evolución temporal de publicaciones")
            fig1.update_layout(xaxis_title="Mes", yaxis_title="N° publicaciones")
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Sin datos de fecha para graficar.")

    with g2:
        por_tema = df_f["topic_label"].value_counts().reset_index()
        por_tema.columns = ["topic_label", "conteo"]
        if not por_tema.empty:
            fig2 = px.bar(por_tema, x="topic_label", y="conteo", color="topic_label",
                          title="Artículos por categoría")
            fig2.update_layout(xaxis_title="Tema", yaxis_title="N° artículos", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        if total_art:
            fig3 = px.histogram(df_f, x="citations", nbins=10, color="topic_label",
                                title="Distribución de citas")
            fig3.update_layout(xaxis_title="Citas", yaxis_title="Frecuencia")
            st.plotly_chart(fig3, use_container_width=True)
    with g4:
        autores = df_f["authors_raw"].dropna().apply(first_author)
        autores = autores[autores != ""]
        top_aut = autores.value_counts().head(10).reset_index()
        top_aut.columns = ["autor", "conteo"]
        if not top_aut.empty:
            fig4 = px.bar(top_aut, x="conteo", y="autor", orientation="h",
                          title="Top 10 autores (primer autor)")
            fig4.update_layout(yaxis={"categoryorder": "total ascending"},
                               xaxis_title="N° artículos", yaxis_title="")
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ---- Tabla interactiva ----
    st.subheader("📋 Tabla de artículos filtrados")
    st.caption(f"{total_art} artículo(s) coinciden con los filtros actuales.")
    cols_tabla = ["title", "authors_raw", "publication_date", "topic_label",
                  "doi", "citations", "popularidad", "n_references"]
    tabla = df_f[cols_tabla].rename(columns={
        "title": "Título", "authors_raw": "Autores", "publication_date": "Fecha",
        "topic_label": "Tema", "doi": "DOI", "citations": "Citas",
        "popularidad": "Vistas/Descargas", "n_references": "Referencias",
    })
    st.dataframe(tabla, use_container_width=True, height=420)
    st.download_button(
        "⬇️ Descargar tabla filtrada (CSV)",
        data=tabla.to_csv(index=False).encode("utf-8"),
        file_name="articulos_filtrados.csv",
        mime="text/csv",
    )


# ===========================================================================
# PESTAÑA 2 — CLUSTERING
# ===========================================================================
with tab_cluster:
    st.subheader("🧩 Segmentación de artículos (clustering no supervisado)")

    st.markdown("#### ¿Qué se analiza y por qué es interesante?")
    st.markdown(
        "Se analiza el conjunto de artículos científicos de la revista **MAKE** "
        "(Machine Learning and Knowledge Extraction) almacenados en la base de "
        "datos SQLite, con el objetivo de identificar **segmentos naturales** "
        "entre los artículos según su impacto y características estructurales "
        "(citas, visualizaciones, número de autores y número de referencias).\n\n"
        "Esto es interesante porque la revista publica artículos muy heterogéneos: "
        "algunos acumulan muchas citas, otros muchas vistas, otros son trabajos "
        "extensos con muchas referencias. Identificar estos perfiles **sin imponer "
        "categorías a priori** permite descubrir patrones reales de publicación, "
        "como por ejemplo grupos de artículos de *alto impacto*, de *alta "
        "visibilidad* o de *producción estándar*."
    )
    st.markdown("#### Enfoque elegido: Clustering (aprendizaje no supervisado)")
    st.markdown(
        "Se eligió el enfoque de **clustering** porque no se dispone de una "
        "etiqueta objetivo predefinida (no hay una columna que diga si un artículo "
        "es 'bueno' o 'malo'), y el interés está en **descubrir estructura latente** "
        "en los datos, no en predecir un valor conocido.\n\n"
        "Se comparan dos algoritmos:\n"
        "- **K-means:** agrupa los artículos minimizando la varianza dentro de cada "
        "cluster. Es eficiente y ampliamente usado como línea base.\n"
        "- **Clustering jerárquico (Ward):** construye una jerarquía de grupos de "
        "abajo hacia arriba. No asume forma esférica de los clusters y es útil para "
        "confirmar que la estructura encontrada por K-means es robusta.\n\n"
        "El modelo final se selecciona según el **coeficiente de silhouette** "
        "(mayor = mejor separación entre clusters), y la concordancia entre ambos "
        "algoritmos se mide con el **Adjusted Rand Index (ARI)**."
    )

    features = ["citations", "popularidad", "n_authors", "n_references"]

    # Datos para clustering (sobre TODA la base, no filtrada)
    dfc = df.copy()
    for c in features:
        dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
    dfc[features] = dfc[features].fillna(dfc[features].median())

    # Control interactivo del número de clusters
    col_a, col_b = st.columns([1, 2])
    with col_a:
        modo_k = st.radio(
            "Número de clusters (k)",
            ["Automático (mejor silhouette)", "Manual"],
            index=0,
        )
        k_manual = st.slider("k manual", 2, 8, 3, disabled=(modo_k.startswith("Auto")))

    X = dfc[features].values
    X_scaled = StandardScaler().fit_transform(X)

    # Curvas de codo y silhouette
    K_range = list(range(2, 9))
    inertias, sils = [], []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X_scaled, lab))

    best_k = K_range[int(np.argmax(sils))]
    K = best_k if modo_k.startswith("Auto") else k_manual
 
    with col_b:
        df_codo = pd.DataFrame({"k": K_range, "Inercia": inertias, "Silhouette": sils})
        fig_codo = px.line(df_codo, x="k", y="Inercia", markers=True,
                           title="Método del codo (K-means)")
        st.plotly_chart(fig_codo, use_container_width=True)
        fig_sil = px.line(df_codo, x="k", y="Silhouette", markers=True,
                          title="Coeficiente de silhouette")
        st.plotly_chart(fig_sil, use_container_width=True)

    st.info(f"k automático sugerido por silhouette: **{best_k}** · usando k = **{K}**")

    # Modelos comparados
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    lab_km = kmeans.fit_predict(X_scaled)
    sil_km = silhouette_score(X_scaled, lab_km)

    agg = AgglomerativeClustering(n_clusters=K, linkage="ward")
    lab_agg = agg.fit_predict(X_scaled)
    sil_agg = silhouette_score(X_scaled, lab_agg)

    ari = adjusted_rand_score(lab_km, lab_agg)

    m1, m2, m3 = st.columns(3)
    m1.metric("Silhouette K-means", f"{sil_km:.3f}")
    m2.metric("Silhouette Jerárquico", f"{sil_agg:.3f}")
    m3.metric("Concordancia (ARI)", f"{ari:.3f}")

    if sil_km >= sil_agg:
        dfc["cluster"] = lab_km
        modelo_final = "K-means"
    else:
        dfc["cluster"] = lab_agg
        modelo_final = "Jerárquico"
    dfc["cluster"] = dfc["cluster"].astype(str)
    st.success(f"Modelo final seleccionado (mejor silhouette): **{modelo_final}**")

    st.markdown(
    "**Interpretación del codo:** la inercia baja de forma marcada de k=2 a "
    "k=4, y luego la curva se aplana. Esto sugiere que agregar más de 2 o 3 "
    "clusters no aporta una reducción significativa de la varianza interna, "
    "por lo que el codo visual apunta a **k=2** como valor óptimo."
    )

    st.markdown(
        "**Interpretación del silhouette:** el coeficiente es máximo en **k=2** "
        "(≈0.59), lo que confirma que con dos grupos los artículos están mejor "
        "separados entre sí. A partir de k=3 el coeficiente cae abruptamente, "
        "indicando que dividir en más clusters produce grupos menos cohesivos."
    )

    st.markdown(
        "Ambos algoritmos obtienen un silhouette similar (~0.59), lo que indica "
        "que la estructura de dos grupos es robusta e independiente del método. "
        "El **ARI de 0.828** confirma una alta concordancia entre K-means y el "
        "clustering jerárquico: los dos están encontrando prácticamente la misma "
        "partición. Se selecciona el **Jerárquico** como modelo final por tener "
        "un silhouette ligeramente superior."
    )
    st.markdown("---")

    # Perfil de clusters
    st.markdown("### Perfil de cada cluster (medias por variable)")
    perfil = dfc.groupby("cluster")[features].mean().round(2)
    perfil["n_articulos"] = dfc.groupby("cluster").size()
    st.dataframe(perfil, use_container_width=True)

    st.markdown(
        "**Interpretación de los clusters:**\n\n"
        "- **Cluster 0 — Artículos estándar (200 artículos):** representa la gran "
        "mayoría de la producción de la revista. Tiene pocas citas (0.3 en "
        "promedio), visibilidad moderada (1164 vistas) y más autores por artículo "
        "(4.08). Son artículos recientes o de nicho que aún no han acumulado "
        "impacto bibliométrico.\n\n"
        "- **Cluster 1 — Artículos de alto impacto (17 artículos):** un grupo "
        "pequeño pero destacado. Triplica las vistas del cluster 0 (3368 vistas), "
        "tiene muchas más citas (3.88), más referencias (64 vs 43) y menos autores "
        "(3.29). Son los artículos más influyentes y consolidados de la revista."
    )

    # PCA 2D
    st.markdown("### Proyección PCA en 2D")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    dfc["pc1"], dfc["pc2"] = coords[:, 0], coords[:, 1]
    var_exp = pca.explained_variance_ratio_[:2].sum() * 100

    fig_pca = px.scatter(
        dfc, x="pc1", y="pc2", color="cluster",
        hover_data={"title": True, "citations": True, "popularidad": True,
                    "pc1": False, "pc2": False},
        title=f"Clusters proyectados con PCA ({modelo_final}) — "
              f"{var_exp:.1f}% de varianza explicada",
    )
    fig_pca.update_traces(marker=dict(size=11))
    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown(
        "El gráfico captura el **67.4% de la varianza** original, lo que es "
        "suficiente para una interpretación visual confiable. Se observa que el "
        "**Cluster 0** (artículos estándar) se concentra en valores bajos de PC1, "
        "mientras que el **Cluster 1** (alto impacto) se dispersa hacia valores "
        "más altos, confirmando que la separación entre grupos es real."
    )
    # Composición temática por cluster
    st.markdown("### Composición temática por cluster")
    ct = (pd.crosstab(dfc["cluster"], dfc["topic_label"], normalize="index") * 100).round(1)
    ct_long = ct.reset_index().melt(id_vars="cluster", var_name="Tema", value_name="Porcentaje")
    fig_ct = px.bar(ct_long, x="cluster", y="Porcentaje", color="Tema",
                    title="Composición temática por cluster (%)")
    st.plotly_chart(fig_ct, use_container_width=True)


    st.markdown(
        "Ambos clusters están dominados por **Machine Learning** como temática "
        "principal, lo cual es esperable dado el enfoque de la revista. Sin embargo, "
        "la diferencia está en los temas secundarios: el **Cluster 0** (artículos "
        "estándar) tiene mayor presencia de **IA Generativa** (16.3%), mientras que "
        "el **Cluster 1** (alto impacto) concentra más artículos de **Estadística** "
        "(23.5%). Esto sugiere que los artículos más influyentes de la revista tienden "
        "a tener un enfoque más metodológico y estadístico, mientras que los de IA "
        "Generativa, siendo un área más reciente, aún no han acumulado el mismo "
        "nivel de citas y vistas."
    )
    # Conclusiones
    st.markdown("---")
    st.markdown("### Conclusiones")
    st.markdown(
        f"- Los artículos de la revista **MAKE** se agrupan en **{K} segmentos** "
        "naturales según su impacto bibliométrico y características estructurales "
        "(citas, visualizaciones, número de autores y referencias)."
    )
    st.markdown(
        f"- El modelo final seleccionado fue **{modelo_final}** con un silhouette de "
        f"**{max(sil_km, sil_agg):.3f}**, lo que indica una separación buena entre "
        f"los grupos. La alta concordancia entre K-means y el clustering jerárquico "
        f"(ARI = {ari:.3f}) confirma que la estructura encontrada es robusta e "
        "independiente del algoritmo utilizado."
    )
    st.markdown(
        "- **Cluster 0 — Artículos estándar (196 artículos):** representa la mayoría "
        "de la producción de la revista. Con pocas citas y visibilidad moderada, "
        "agrupa artículos recientes o de nicho que aún no han acumulado impacto "
        "bibliométrico. Predominan los temas de Machine Learning e IA Generativa."
    )
    st.markdown(
        "- **Cluster 1 — Artículos de alto impacto (17 artículos):** grupo reducido "
        "pero destacado, con más del triple de vistas y citas que el Cluster 0, "
        "mayor número de referencias y mayor presencia de artículos de Estadística. "
        "Corresponde a los trabajos más influyentes y consolidados de la revista."
    )
    st.markdown(
        "- **Limitaciones:** la base cubre principalmente una revista y dos años "
        "(2025-2026); la columna `downloads` estaba vacía y se usó `views` como "
        "proxy de popularidad; las citas de artículos de 2026 son aún muy bajas "
        "por ser recientes, lo que puede sesgar la segmentación."
    )
    st.markdown(
        "- **Recomendación:** ampliar la recolección a más revistas Q1 y más años "
        "para validar si los perfiles se mantienen. El botón de actualización ya "
        "permite incorporar artículos nuevos de 2026 de forma automática, lo que "
        "enriquecerá el análisis con el tiempo."
    )

    with st.expander("📝 Uso de IA"):
        st.markdown("### Documentación del uso de IA (Claude - Anthropic)")

        st.markdown("**¿Qué pedí?**")
        st.markdown(
            "Se utilizó IA como asistente de desarrollo a lo largo de todo el proyecto. "
            "Las solicitudes principales fueron:\n"
            "- Resolver el bloqueo HTTP 403 de MDPI al hacer scraping directo, "
            "buscando alternativas para traer artículos de 2026.\n"
            "- Implementar scraping con Selenium (navegador real) como método "
            "principal para obtener vistas y citas de MDPI, con Crossref y OpenAlex "
            "como respaldo cuando no hay navegador disponible.\n"
            "- Integrar el análisis de clustering como una segunda "
            "pestaña dentro del mismo dashboard de Streamlit.\n"
         )

        st.markdown("**¿Qué me dio?**")
        st.markdown(
            "- Solución al problema del caché de Streamlit que mostraba datos "
            "desactualizados: se implementó invalidación por fecha de modificación "
            "del archivo SQLite.\n"
            "- Estrategia de scraping en tres capas: Selenium → Crossref → OpenAlex, "
            "con degradación automática según el entorno.\n"
            "- Pipeline de clustering: estandarización con StandardScaler, selección "
            "de k con codo y silhouette, comparación K-means vs jerárquico, "
            "proyección PCA 2D y perfilado de clusters.\n"
        )

        st.markdown("**¿Qué ajusté?**")
        st.markdown(

            "- Se adaptó todo el código a la estructura real de la base SQLite "
            "(columnas, formatos de fecha, DOIs) del Taller 1 hecho en R.\n"
            "- Se usó `views` como métrica de popularidad dado que `downloads` "
            "estaba vacío en la base.\n"
            "- Se interpretaron los clusters con los datos reales obtenidos: "
            "k=2, silhouette=0.595, modelo jerárquico, Cluster 0 (196 artículos "
            "estándar) y Cluster 1 (17 artículos de alto impacto).\n"
            "- Se tomaron decisiones propias sobre la estructura del proyecto: "
            "integrar ambas entregas en un solo Streamlit, elegir clustering "
            "como enfoque del Proyecto Final, y usar APIs en lugar de scraping "
            "directo cuando Selenium no está disponible."
        )


