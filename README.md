# Minería de Datos — Dashboard Q1 + Clustering (Streamlit)

Aplicación única en **Streamlit** que integra dos entregas sobre la misma base de
datos SQLite de revistas Q1 (revista *Machine Learning and Knowledge Extraction*,
MDPI), construida por web scraping en el Taller 1.

La app tiene **dos pestañas**:

1. **Dashboard (Taller 2)** — conexión SQLite, sidebar con filtros, widgets,
   indicadores, visualizaciones Plotly, tabla interactiva y botón de scraping para
   buscar artículos nuevos (p. ej. de 2026).
2. **Proyecto Final — Clustering** — segmentación de artículos con K-means vs
   jerárquico, método del codo + silhouette, perfilado de clusters, proyección PCA
   2D y conclusiones.

## Archivos

```
.
├── app.py                 # Toda la aplicación (las dos pestañas)
├── scraper.py             # Lógica de scraping (la usa app.py)
├── make_q1_2025.sqlite    # Base de datos del Taller 1
├── requirements.txt
└── README.md
```

## Cómo correr

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre en http://localhost:8501

> Nota: la columna `downloads` venía vacía en la base; se usa `views` como métrica de
> popularidad/descargas.

## Despliegue (Streamlit Community Cloud)

1. Subir este repositorio a GitHub.
2. Entrar a https://streamlit.io/cloud e iniciar sesión con GitHub.
3. **New App** → elegir el repo, la rama y `app.py` como archivo principal.
4. **Deploy**.

**App desplegada:** _(pega aquí tu link de Streamlit Cloud)_
