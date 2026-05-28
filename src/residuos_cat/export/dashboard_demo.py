"""Dashboard PÚBLICO en modo demo (sin contactos sensibles).

Diferencias respecto a ``dashboard.py``:
- Lee desde ``data/demo/`` (anonimizado).
- Banner permanente arriba avisando que es DEMO.
- CTAs de contacto en sidebar y footer.
- Sin tabs de "competencia con NIF/contacto"; muestra todo agregado.
- Tooltip explicando que el dataset completo se entrega con contacto+NIF tras contratar.

Pensado para deploy en Streamlit Cloud o HuggingFace Spaces.

Lanzar localmente para test:
    uv run streamlit run src/residuos_cat/export/dashboard_demo.py
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = REPO_ROOT / "data" / "demo"

LEADS_PATH = DEMO_DIR / "leads_scored_demo.parquet"
GESTORES_PATH = DEMO_DIR / "gestores_demo.parquet"

CONTACT_EMAIL = "yoelcp1988@gmail.com"


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia haversine en km entre dos puntos."""
    radius = 6371.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Streamlit no instalado. Ejecuta: uv sync --extra dashboard") from exc

    st.set_page_config(
        page_title="Residuos Cataluña — Demo pública",
        page_icon="♻️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": f"mailto:{CONTACT_EMAIL}?subject=Producto%20leads%20residuos",
            "Report a bug": f"mailto:{CONTACT_EMAIL}?subject=Bug%20demo%20residuos",
            "About": "Demo pública del producto de leads. "
            f"Para el dataset completo: {CONTACT_EMAIL}",
        },
    )

    # ── Banner demo permanente ──
    st.markdown(
        """
        <div style="background:#fff3cd; border:1px solid #ffeaa7; padding:10px 16px;
                    border-radius:6px; margin-bottom:16px; font-size:14px;">
        🧪 <b>DEMO PÚBLICA</b> · Dataset real de empresas catalanas productoras de
        residuos, con contactos (email/teléfono/NIF web) <b>ocultos</b>. El dataset
        completo se entrega tras contratar el encargo.
        👉 <a href="mailto:"""
        + CONTACT_EMAIL
        + """?subject=Producto%20leads%20residuos">
        Solicitar dataset completo</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("Leads de empresas productoras de residuos — Catalunya")
    st.caption(
        "Pipeline construido sobre 8 fuentes públicas: PRTR-ES (MITECO), "
        "Dades Obertes Catalunya (ARC), RGPGRC, OSM Nominatim, ICGC."
    )

    if not LEADS_PATH.exists():
        st.error(
            f"Dataset demo no encontrado en {LEADS_PATH}. "
            "Ejecuta: `uv run python scripts/anonymize_for_demo.py`"
        )
        st.stop()

    leads_all = pl.read_parquet(LEADS_PATH)

    # ── Sidebar de parámetros ──
    st.sidebar.header("⚙️ Tus parámetros")
    st.sidebar.markdown("Ajusta para simular el alcance de **tu** planta de gestión de residuos.")

    target_lers_input = st.sidebar.text_input(
        "Códigos LER objetivo (6 dígitos, separados por coma)",
        value="",
        placeholder="170504, 200201, 030307",
        help="Códigos LER que está autorizada a tratar tu planta. Vacío = TODOS.",
    )
    target_lers = [t.strip() for t in target_lers_input.split(",") if t.strip()]

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📍 Planta del cliente**")
    plant_lat = st.sidebar.number_input(
        "Latitud", value=41.3851, step=0.001, format="%.4f", help="Por defecto: Barcelona ciudad"
    )
    plant_lon = st.sidebar.number_input(
        "Longitud", value=2.1734, step=0.001, format="%.4f", help="Por defecto: Barcelona ciudad"
    )

    radius_km = st.sidebar.slider(
        "Radio operativo (km)",
        10,
        300,
        120,
        step=10,
        help="Distancia máxima rentable desde tu planta",
    )
    max_qty = int(leads_all["quantity_tonnes_last"].max() or 50000)
    cap_min, cap_max = st.sidebar.slider(
        "Capacidad t/año (rango óptimo)",
        min_value=0,
        max_value=max_qty,
        value=(0, min(50000, max_qty)),
        step=100,
        help="Volumen anual mínimo y máximo que tu planta puede absorber",
    )

    st.sidebar.markdown("---")
    show_gestores = st.sidebar.checkbox(
        "Incluir gestores sospechosos",
        value=False,
        help="Empresas con perfil de gestora (no productor real). Por defecto excluidas.",
    )
    only_with_precise_coord = st.sidebar.checkbox(
        "Solo coord exacta (sin centroide)",
        value=False,
        help="Más precisión geográfica, menos universo",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        ### 💬 ¿Te interesa?

        El dataset completo incluye:
        - **NIFs** identificados (Tier 1)
        - **Emails y teléfonos** corporativos
        - **URLs** verificadas
        - **Score personalizado** con tus parámetros

        📧 [{CONTACT_EMAIL}](mailto:{CONTACT_EMAIL}?subject=Producto%20leads%20residuos)
        """
    )

    # ── Filtrado ──
    filtered = leads_all.with_columns(
        pl.struct(["lat", "lon"])
        .map_elements(
            lambda x: _haversine(plant_lat, plant_lon, x["lat"], x["lon"])
            if x["lat"] is not None and x["lon"] is not None
            else None,
            return_dtype=pl.Float64,
        )
        .alias("dist_recalc_km"),
    )

    if target_lers:
        filtered = filtered.filter(pl.col("ler_code").is_in(target_lers))
    if not show_gestores:
        filtered = filtered.filter(pl.col("lead_type") == "productor")
    if only_with_precise_coord:
        filtered = filtered.filter(pl.col("huella_geocod").is_in(["accidents_greus", "geocoded"]))
    filtered = filtered.filter(
        (pl.col("dist_recalc_km").is_null()) | (pl.col("dist_recalc_km") <= radius_km)
    ).filter(
        (pl.col("quantity_tonnes_last").is_null())
        | (
            (pl.col("quantity_tonnes_last") >= cap_min)
            & (pl.col("quantity_tonnes_last") <= cap_max)
        )
    )

    # ── KPIs ──
    n_leads = filtered.height
    n_empresas = filtered["facility_id_external"].n_unique()
    n_lers = filtered["ler_code"].n_unique()
    t_total = filtered["quantity_tonnes_last"].sum() or 0
    avg_dist = filtered["dist_recalc_km"].mean() or 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Leads filtrados", f"{n_leads:,}")
    col2.metric("Empresas únicas", f"{n_empresas:,}")
    col3.metric("LERs distintos", f"{n_lers:,}")
    col4.metric("Toneladas/año", f"{t_total:,.0f}")
    col5.metric("Dist media (km)", f"{avg_dist:.1f}")

    if n_leads == 0:
        st.warning("Sin leads con los filtros actuales. Relaja los parámetros del lateral.")
        st.stop()

    # ── Tabs ──
    tab_table, tab_map, tab_dist = st.tabs(["📋 Top leads", "🗺️ Mapa", "📊 Distribuciones"])

    with tab_table:
        st.subheader(f"Top 100 leads de {n_leads:,} filtrados")
        display = (
            filtered.sort("score_prioridad", descending=True)
            .head(100)
            .select(
                pl.col("score_prioridad").round(1).alias("Score"),
                pl.col("facility_name").alias("Empresa"),
                pl.col("provincia").alias("Provincia"),
                pl.col("municipi").alias("Municipio"),
                pl.col("ler_code").alias("LER"),
                pl.col("descripcion_es").alias("Residuo"),
                pl.col("quantity_tonnes_last").round(1).alias("t/año"),
                pl.col("dist_recalc_km").round(1).alias("Dist_km"),
                pl.col("nivel_confianza_id").alias("Confianza"),
                pl.col("densidad_competencia").alias("Competencia"),
            )
        )
        st.dataframe(display.to_pandas(), use_container_width=True, height=600)

        st.info(
            "🔒 Columnas **Email**, **Teléfono**, **NIF** y **URL_Web** ocultas en la demo. "
            f"Disponibles en el dataset completo: [{CONTACT_EMAIL}]"
            f"(mailto:{CONTACT_EMAIL}?subject=Producto%20leads%20residuos)"
        )

    with tab_map:
        st.subheader("Mapa de leads filtrados")
        precise = filtered.filter(
            pl.col("huella_geocod").is_in(["accidents_greus", "geocoded"])
        ).select("lat", "lon", "score_prioridad")
        if precise.height > 0:
            st.map(
                precise.to_pandas(),
                latitude="lat",
                longitude="lon",
                size="score_prioridad",
                zoom=8,
            )
            st.caption(
                f"Mostrando {precise.height} leads con coord precisa. "
                f"({n_leads - precise.height} adicionales tienen coord aproximada de provincia)."
            )
        else:
            st.info("Sin coords precisas con los filtros actuales.")

    with tab_dist:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Por score")
            score_dist = filtered.select("score_prioridad").to_pandas()
            counts = score_dist["score_prioridad"].value_counts(bins=20).sort_index()
            st.bar_chart(counts)

            st.subheader("Por provincia")
            prov = filtered.group_by("provincia").len().sort("len", descending=True)
            st.bar_chart(prov.to_pandas().set_index("provincia"))

        with col_b:
            st.subheader("Top 15 LERs por t/año")
            top_ler = (
                filtered.group_by("ler_code", "descripcion_es")
                .agg(pl.col("quantity_tonnes_last").sum().alias("t"))
                .sort("t", descending=True)
                .head(15)
            )
            st.dataframe(top_ler.to_pandas(), use_container_width=True, hide_index=True)

            st.subheader("Por densidad de competencia")
            comp = filtered.group_by("densidad_competencia").len().sort("len", descending=True)
            st.dataframe(comp.to_pandas(), use_container_width=True, hide_index=True)

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align:center; color:#666; font-size:13px; padding:10px;">
        Pipeline de inteligencia comercial B2B desde registros públicos · 8 fuentes integradas ·
        21.600 leads · 889 empresas · 6 años de histórico (2019-2024) <br>
        Datos: PRTR-ES (MITECO) · Dades Obertes Catalunya (ARC) · RGPGRC ·
        OpenStreetMap contributors · ICGC ·
        Producto elaborado por Yoel Castaño, 2026.<br>
        <b>Contacto:</b> <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
