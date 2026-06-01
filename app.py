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
    
    st.markdown(
        "La recolección se realizó mediante **web scraping** con Python, usando las "
        "librerías **requests** (para descargar las páginas) y **BeautifulSoup** "
        "(para extraer la información del HTML). El proceso siguió estos pasos:"
    )
    st.markdown(
        "1. Se accedió al índice del **Volumen 7** de la revista en el sitio de MDPI.\n"
        "2. Se recorrió cada artículo listado, extrayendo sus metadatos "
        "(título, autores, fecha, DOI, resumen, citas, referencias y visualizaciones).\n"
        "3. La información se limpió y estructuró en un **DataFrame** de pandas.\n"
        "4. Finalmente, los datos se almacenaron en una base de datos **SQLite** "
        "para su posterior consulta y análisis."
    )

    st.info(
        "**Nota:** Como no se cuenta con la variable *Descargas*, se hace uso del "
        "*Número de visualizaciones* en su defecto."
    )

    st.markdown("#### Actualización: artículos de 2026")
    st.markdown(
        "Además de los datos del Taller 1 (Volumen 7, año 2025), el dashboard "
        "permite **actualizar la base con artículos nuevos** del **Volumen 8 (2026)** "
        "mediante el botón *Buscar artículos nuevos* en el panel lateral.\n\n"
        "Para esta actualización **no se hace scraping directo al sitio de MDPI**, "
        "ya que su servidor bloquea las peticiones automáticas (error HTTP 403). "
        "En su lugar se consultan **APIs públicas de metadatos académicos**:"
    )
    st.markdown(
        "- **Crossref** (fuente principal): API abierta y gratuita que se consulta "
        "por el ISSN de la revista y permite filtrar por fecha, recuperando todos "
        "los artículos nuevos de 2026 (de todos los issues del volumen).\n"
        "- **OpenAlex** (respaldo): si Crossref falla, se consulta esta segunda API, "
        "que además aporta el número de citas.\n\n"
        "El sistema compara los **DOI** obtenidos contra los ya almacenados e "
        "inserta únicamente los artículos nuevos en la base **SQLite**. Si no hay "
        "artículos nuevos, verifica los últimos cinco almacenados para actualizar "
        "sus métricas."
    )

    st.markdown(
        "La siguiente tabla resume los **200 artículos** almacenados en la base de "
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

    # ---- Acción: scraping ----
    if buscar:
        progress = st.progress(0.0, text="Iniciando scraping...")

        def cb(i, total, url):
            progress.progress(i / total, text=f"Revisando {i}/{total}: {url[-30:]}")

        with st.spinner("Consultando Crossref / OpenAlex..."):
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
# PESTAÑA 2 — PROYECTO FINAL
# ===========================================================================
with tab_cluster:
    st.subheader("🧩 Segmentación de artículos (clustering no supervisado)")
    st.markdown("##### ¿Qué segmentos naturales existen entre los artículos?")
    st.markdown(
        "Análisis según su impacto y características (citas, vistas, autores, "
        "referencias). Se comparan **K-means** y **clustering jerárquico**, "
        "eligiendo el número de clusters con el método del codo y el "
        "coeficiente de *silhouette*."
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

    st.markdown("---")

    # Perfil de clusters
    st.markdown("### Perfil de cada cluster (medias por variable)")
    perfil = dfc.groupby("cluster")[features].mean().round(2)
    perfil["n_articulos"] = dfc.groupby("cluster").size()
    st.dataframe(perfil, use_container_width=True)

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

    # Composición temática por cluster
    st.markdown("### Composición temática por cluster")
    ct = (pd.crosstab(dfc["cluster"], dfc["topic_label"], normalize="index") * 100).round(1)
    ct_long = ct.reset_index().melt(id_vars="cluster", var_name="Tema", value_name="Porcentaje")
    fig_ct = px.bar(ct_long, x="cluster", y="Porcentaje", color="Tema",
                    title="Composición temática por cluster (%)")
    st.plotly_chart(fig_ct, use_container_width=True)

    # Conclusiones
    st.markdown("---")
    st.markdown(
        "### Conclusiones\n"
        f"- Los artículos se agrupan en **{K} segmentos** diferenciados según impacto "
        "(citas), visibilidad (vistas) y características estructurales (autores, referencias).\n"
        f"- El modelo final fue **{modelo_final}** (silhouette "
        f"{max(sil_km, sil_agg):.3f}); la alta concordancia entre algoritmos "
        f"(ARI {ari:.3f}) sugiere que la estructura es estable.\n"
        "- **Limitaciones:** dataset pequeño (una revista, un año); `downloads` vacío, "
        "se usó `views` como proxy; las citas de artículos recientes aún son bajas.\n"
        "- **Recomendación:** ampliar la recolección a más años/revistas (el botón de "
        "scraping ya permite incorporar artículos de 2026) para validar los perfiles."
    )

    with st.expander("📝 Uso de IA (edita con tu caso real)"):
        st.markdown(
            "- **Qué pedí:** estructurar el pipeline de clustering y elegir métricas.\n"
            "- **Qué me dio:** escalado + comparación K-means vs jerárquico, codo, "
            "silhouette y visualizaciones.\n"
            "- **Qué ajusté:** adapté las columnas a mi base real, interpreté los "
            "clusters con mis datos y reescribí las conclusiones."
        )

