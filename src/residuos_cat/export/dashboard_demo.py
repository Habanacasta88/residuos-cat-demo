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

# Mapeos para mostrar etiquetas más amigables que los códigos crudos
CLASIFICACION_LABELS = {
    "P": "Peligroso (P)",
    "NP": "No peligroso (NP)",
    "DP": "Doble entrada / mixto (DP)",
}
TREATMENT_KIND_LABELS = {
    "R": "Valorización (R)",
    "D": "Eliminación (D)",
}
CONFIANZA_LABELS = {
    "alto": "Alto — match directo con registro oficial",
    "medio": "Medio — match probabilístico fuzzy",
}
COMPETENCIA_LABELS = {
    "baja": "Baja — pocas gestoras en la comarca",
    "media": "Media — competencia moderada",
    "alta": "Alta — saturado",
}


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

    @st.cache_data(show_spinner="Cargando dataset…")
    def _load_leads() -> pl.DataFrame:
        return pl.read_parquet(LEADS_PATH)

    leads_all = _load_leads()
    total_records = leads_all.height
    total_empresas = leads_all["facility_id_external"].n_unique()

    # ════════════════════════════════════════════════════════════
    # 🔍 BUSCADOR PRINCIPAL — PANEL CENTRAL ARRIBA
    # ════════════════════════════════════════════════════════════
    with st.container():
        st.markdown("### 🔍 Buscar empresa específica")
        cols_search = st.columns([3, 2, 1])
        with cols_search[0]:
            search_name = st.text_input(
                "Nombre de la empresa (búsqueda parcial, sin distinción mayús/minús)",
                value="",
                placeholder="Ej: Damm, Seat, Mango, Roca, Ercros...",
                help="Busca en el campo razón social. Coincidencias parciales (contiene).",
                key="search_name",
            )
        with cols_search[1]:
            search_nif = st.text_input(
                "NIF (búsqueda exacta)",
                value="",
                placeholder="A08015497",
                help=(
                    "Solo el 17% del demo tiene NIF asignado (Tier 1). "
                    "En el dataset completo: 100% identificados."
                ),
                key="search_nif",
            )
        with cols_search[2]:
            st.metric("Universo", f"{total_empresas:,}", help=f"De {total_records:,} leads totales")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # SIDEBAR — TODOS LOS FILTROS, ORGANIZADOS EN SECCIONES
    # ════════════════════════════════════════════════════════════
    st.sidebar.header("⚙️ Tus parámetros")
    st.sidebar.markdown(
        "Ajusta para simular el alcance de **tu** planta de gestión de residuos."
    )

    # ── Parámetros de tu planta (siempre visible) ──
    with st.sidebar.expander("📍 Tu planta + radio operativo", expanded=True):
        plant_lat = st.number_input(
            "Latitud planta", value=41.3851, step=0.001, format="%.4f",
            help="Por defecto: Barcelona centro"
        )
        plant_lon = st.number_input(
            "Longitud planta", value=2.1734, step=0.001, format="%.4f",
            help="Por defecto: Barcelona centro"
        )
        radius_km = st.slider(
            "Radio operativo (km)", 10, 300, 120, step=10,
            help="Distancia máxima rentable desde tu planta"
        )
        max_qty = int(leads_all["quantity_tonnes_last"].max() or 50000)
        cap_min, cap_max = st.slider(
            "Capacidad t/año (rango óptimo)",
            min_value=0,
            max_value=max_qty,
            value=(0, min(50000, max_qty)),
            step=100,
            help="Volumen anual mínimo y máximo que tu planta puede absorber",
        )

    # ── Filtros de residuo (LER) ──
    with st.sidebar.expander("♻️ Residuo (LER)", expanded=False):
        target_lers_input = st.text_input(
            "Códigos LER objetivo (6 dígitos, coma separados)",
            value="",
            placeholder="170504, 200201, 030307",
            help="Códigos LER que tu planta está autorizada a tratar. Vacío = TODOS.",
        )
        target_lers = [t.strip() for t in target_lers_input.split(",") if t.strip()]

        # Capítulo LER (primeros 2 dígitos)
        capitulos_disponibles = (
            leads_all.select(pl.col("ler_code").str.slice(0, 2).alias("cap"))
            .unique()
            .sort("cap")["cap"]
            .to_list()
        )
        capitulos_sel = st.multiselect(
            "Capítulo LER (familia)",
            options=capitulos_disponibles,
            default=[],
            help=(
                "Filtra por los primeros 2 dígitos del LER. "
                "Ej: 17 = construcción y demolición, 20 = urbanos."
            ),
        )

        clasif_opciones = sorted(leads_all["clasificacion"].unique().drop_nulls().to_list())
        clasif_sel = st.multiselect(
            "Clasificación",
            options=clasif_opciones,
            default=[],
            format_func=lambda x: CLASIFICACION_LABELS.get(x, x),
            help="Peligroso (P), No peligroso (NP), Doble entrada / mixto (DP)",
        )

        treat_opciones = sorted(leads_all["treatment_kind"].unique().drop_nulls().to_list())
        treat_sel = st.multiselect(
            "Tipo de tratamiento",
            options=treat_opciones,
            default=[],
            format_func=lambda x: TREATMENT_KIND_LABELS.get(x, x),
            help="R = valorización, D = eliminación",
        )

        search_descripcion = st.text_input(
            "Buscar en descripción de residuo",
            value="",
            placeholder="Ej: papel, plástico, lodo, hidrocarburo",
            help="Busca en la descripción ES (contiene, sin distinción mayús/minús)",
        )

    # ── Filtros geográficos ──
    with st.sidebar.expander("🗺️ Geografía", expanded=False):
        provincia_opciones = sorted(leads_all["provincia"].unique().drop_nulls().to_list())
        provincia_sel = st.multiselect(
            "Provincia",
            options=provincia_opciones,
            default=[],
            help="Filtra por provincia(s) catalana(s)",
        )

        # Comarcas filtradas por provincia (si hay)
        if provincia_sel:
            comarca_pool = (
                leads_all.filter(pl.col("provincia").is_in(provincia_sel))["comarca"]
                .unique()
                .drop_nulls()
                .to_list()
            )
        else:
            comarca_pool = leads_all["comarca"].unique().drop_nulls().to_list()
        comarca_sel = st.multiselect(
            "Comarca",
            options=sorted(comarca_pool),
            default=[],
            help="Subdivisiones de la provincia (limitadas a la provincia seleccionada si hay)",
        )

        # Municipios filtrados por provincia/comarca
        muni_filter = leads_all
        if provincia_sel:
            muni_filter = muni_filter.filter(pl.col("provincia").is_in(provincia_sel))
        if comarca_sel:
            muni_filter = muni_filter.filter(pl.col("comarca").is_in(comarca_sel))
        muni_pool = muni_filter["municipi"].unique().drop_nulls().to_list()
        muni_sel = st.multiselect(
            "Municipio",
            options=sorted(muni_pool),
            default=[],
            help="Solo aparecen los municipios geocodificados (~14% del demo)",
        )

    # ── Filtros temporales ──
    with st.sidebar.expander("📅 Tiempo", expanded=False):
        years_disponibles = sorted(leads_all["last_year"].unique().drop_nulls().to_list())
        years_sel = st.multiselect(
            "Año del último reporte",
            options=years_disponibles,
            default=[],
            help="Año en que la empresa reportó el residuo por última vez",
        )
        min_years_reported = st.slider(
            "Años reportados mínimo",
            min_value=1,
            max_value=6,
            value=1,
            step=1,
            help="Empresas que han reportado en al menos N años (más años = más estable)",
        )

    # ── Filtros de cualidad ──
    with st.sidebar.expander("🎯 Cualidad del lead", expanded=False):
        conf_opciones = sorted(leads_all["nivel_confianza_id"].unique().drop_nulls().to_list())
        conf_sel = st.multiselect(
            "Nivel de confianza identidad",
            options=conf_opciones,
            default=[],
            format_func=lambda x: CONFIANZA_LABELS.get(x, x),
            help="Cómo se cruzó la empresa con el registro oficial",
        )

        comp_opciones = sorted(leads_all["densidad_competencia"].unique().drop_nulls().to_list())
        comp_sel = st.multiselect(
            "Densidad de competencia",
            options=comp_opciones,
            default=[],
            format_func=lambda x: COMPETENCIA_LABELS.get(x, x),
            help="Densidad de gestoras competidoras en la comarca de la empresa",
        )

        score_min = st.slider(
            "Score mínimo",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=5.0,
            help="Score de prioridad mínimo (0-100). Mayor score = mejor lead.",
        )

    # ── Opciones avanzadas ──
    with st.sidebar.expander("⚙️ Opciones avanzadas", expanded=False):
        show_gestores = st.checkbox(
            "Incluir gestores sospechosos",
            value=False,
            help="Empresas con perfil de gestora (no productor real). Por defecto excluidas.",
        )
        only_with_precise_coord = st.checkbox(
            "Solo coord exacta (sin centroide)",
            value=False,
            help="Más precisión geográfica, menos universo",
        )
        top_n_display = st.select_slider(
            "Top a mostrar en tabla",
            options=[50, 100, 250, 500, 1000],
            value=100,
            help="Cuántas filas mostrar en la tabla de leads",
        )
        sort_by = st.radio(
            "Ordenar por",
            options=["score_prioridad", "quantity_tonnes_last", "dist_recalc_km", "facility_name"],
            index=0,
            format_func=lambda x: {
                "score_prioridad": "Score (desc)",
                "quantity_tonnes_last": "Toneladas (desc)",
                "dist_recalc_km": "Distancia (asc)",
                "facility_name": "Nombre (A-Z)",
            }[x],
        )

    # ── CTA de contacto ──
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        ### 💬 ¿Te interesa?

        El dataset completo incluye:
        - **NIFs** identificados (100% Tier 1)
        - **Emails y teléfonos** corporativos
        - **URLs** verificadas
        - **Score personalizado** con tus parámetros
        - **Exportable** a Excel + mapa Folium

        📧 [{CONTACT_EMAIL}](mailto:{CONTACT_EMAIL}?subject=Producto%20leads%20residuos)
        """
    )

    # ════════════════════════════════════════════════════════════
    # APLICAR FILTROS — orden importa para performance
    # ════════════════════════════════════════════════════════════
    filtered = leads_all

    # 1. Buscador por nombre (más selectivo, lo primero)
    if search_name.strip():
        filtered = filtered.filter(
            pl.col("facility_name").str.to_lowercase().str.contains(search_name.lower().strip())
        )

    # 2. Buscador por NIF exacto
    if search_nif.strip():
        nif_clean = search_nif.strip().upper().replace(" ", "").replace("-", "")
        filtered = filtered.filter(
            pl.col("nif").str.to_uppercase().str.replace_all(" ", "") == nif_clean
        )

    # 3. Geografía
    if provincia_sel:
        filtered = filtered.filter(pl.col("provincia").is_in(provincia_sel))
    if comarca_sel:
        filtered = filtered.filter(pl.col("comarca").is_in(comarca_sel))
    if muni_sel:
        filtered = filtered.filter(pl.col("municipi").is_in(muni_sel))

    # 4. LER
    if target_lers:
        filtered = filtered.filter(pl.col("ler_code").is_in(target_lers))
    if capitulos_sel:
        filtered = filtered.filter(pl.col("ler_code").str.slice(0, 2).is_in(capitulos_sel))
    if clasif_sel:
        filtered = filtered.filter(pl.col("clasificacion").is_in(clasif_sel))
    if treat_sel:
        filtered = filtered.filter(pl.col("treatment_kind").is_in(treat_sel))
    if search_descripcion.strip():
        filtered = filtered.filter(
            pl.col("descripcion_es").str.to_lowercase().str.contains(
                search_descripcion.lower().strip()
            )
        )

    # 5. Tiempo
    if years_sel:
        filtered = filtered.filter(pl.col("last_year").is_in(years_sel))
    if min_years_reported > 1:
        filtered = filtered.filter(pl.col("n_years_reported") >= min_years_reported)

    # 6. Cualidad
    if conf_sel:
        filtered = filtered.filter(pl.col("nivel_confianza_id").is_in(conf_sel))
    if comp_sel:
        filtered = filtered.filter(pl.col("densidad_competencia").is_in(comp_sel))
    if score_min > 0:
        filtered = filtered.filter(pl.col("score_prioridad") >= score_min)

    # 7. Avanzadas
    if not show_gestores:
        filtered = filtered.filter(pl.col("lead_type") == "productor")
    if only_with_precise_coord:
        filtered = filtered.filter(
            pl.col("huella_geocod").is_in(["accidents_greus", "geocoded"])
        )

    # 8. Distancia y capacidad (recalculadas con la planta del usuario)
    filtered = filtered.with_columns(
        pl.struct(["lat", "lon"])
        .map_elements(
            lambda x: _haversine(plant_lat, plant_lon, x["lat"], x["lon"])
            if x["lat"] is not None and x["lon"] is not None
            else None,
            return_dtype=pl.Float64,
        )
        .alias("dist_recalc_km"),
    )
    filtered = filtered.filter(
        (pl.col("dist_recalc_km").is_null()) | (pl.col("dist_recalc_km") <= radius_km)
    ).filter(
        (pl.col("quantity_tonnes_last").is_null())
        | (
            (pl.col("quantity_tonnes_last") >= cap_min)
            & (pl.col("quantity_tonnes_last") <= cap_max)
        )
    )

    # ════════════════════════════════════════════════════════════
    # KPIs
    # ════════════════════════════════════════════════════════════
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

    # ── Indicador especial cuando se hace búsqueda por nombre/NIF ──
    if search_name.strip() or search_nif.strip():
        if n_empresas > 0:
            st.success(
                f"✅ **{n_empresas} empresa(s)** encontradas con "
                f"{n_leads} líneas de residuo declaradas. "
                f"Ver detalle en la pestaña 'Top leads' debajo."
            )
        else:
            st.error(
                f"❌ Ningún resultado para tu búsqueda. "
                "Causas posibles: (a) la empresa no está en Cataluña, "
                "(b) no es productora de residuos declarante en PRTR/ARC, "
                "(c) escribió la razón social de otra forma — prueba con menos letras. "
                f"Si estás seguro de que debería aparecer: [escríbeme]"
                f"(mailto:{CONTACT_EMAIL}?subject=Empresa%20que%20no%20aparece%20en%20demo)."
            )

    if n_leads == 0:
        st.warning("Sin leads con los filtros actuales. Relaja los parámetros del lateral.")
        st.stop()

    # ════════════════════════════════════════════════════════════
    # TABS
    # ════════════════════════════════════════════════════════════
    tab_table, tab_map, tab_dist, tab_companies = st.tabs(
        ["📋 Top leads", "🗺️ Mapa", "📊 Distribuciones", "🏢 Por empresa"]
    )

    # ── Tab 1: Top leads ──
    with tab_table:
        st.subheader(f"Mostrando top {min(top_n_display, n_leads):,} de {n_leads:,} leads filtrados")
        sort_desc = sort_by != "facility_name" and sort_by != "dist_recalc_km"
        display = (
            filtered.sort(sort_by, descending=sort_desc, nulls_last=True)
            .head(top_n_display)
            .select(
                pl.col("score_prioridad").round(1).alias("Score"),
                pl.col("facility_name").alias("Empresa"),
                pl.col("provincia").alias("Provincia"),
                pl.col("municipi").alias("Municipio"),
                pl.col("comarca").alias("Comarca"),
                pl.col("ler_code").alias("LER"),
                pl.col("descripcion_es").alias("Residuo"),
                pl.col("clasificacion").alias("Clasif"),
                pl.col("treatment_kind").alias("Trat"),
                pl.col("quantity_tonnes_last").round(1).alias("t/año"),
                pl.col("last_year").alias("Año"),
                pl.col("n_years_reported").alias("Años_rep"),
                pl.col("dist_recalc_km").round(1).alias("Dist_km"),
                pl.col("nivel_confianza_id").alias("Confianza"),
                pl.col("densidad_competencia").alias("Competencia"),
            )
        )
        st.dataframe(display.to_pandas(), use_container_width=True, height=600)

        st.info(
            "🔒 Columnas **Email**, **Teléfono**, **NIF web** y **URL** ocultas en la demo. "
            f"Disponibles en el dataset completo: [{CONTACT_EMAIL}]"
            f"(mailto:{CONTACT_EMAIL}?subject=Producto%20leads%20residuos)"
        )

    # ── Tab 2: Mapa ──
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
                f"Mostrando {precise.height:,} leads con coord precisa. "
                f"({n_leads - precise.height:,} adicionales tienen coord aproximada de provincia)."
            )
        else:
            st.info(
                "Sin coords precisas con los filtros actuales. "
                "Quita 'Solo coord exacta' o relaja filtros geográficos."
            )

    # ── Tab 3: Distribuciones ──
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

            st.subheader("Por año de reporte")
            yr = filtered.group_by("last_year").len().sort("last_year")
            st.bar_chart(yr.to_pandas().set_index("last_year"))

        with col_b:
            st.subheader("Top 15 LERs por t/año")
            top_ler = (
                filtered.group_by("ler_code", "descripcion_es")
                .agg(pl.col("quantity_tonnes_last").sum().alias("t"))
                .sort("t", descending=True)
                .head(15)
            )
            st.dataframe(top_ler.to_pandas(), use_container_width=True, hide_index=True)

            st.subheader("Por clasificación de peligrosidad")
            clasif = filtered.group_by("clasificacion").len().sort("len", descending=True)
            st.dataframe(
                clasif.to_pandas().rename(columns={"len": "leads"}),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Por densidad de competencia")
            comp = filtered.group_by("densidad_competencia").len().sort("len", descending=True)
            st.dataframe(
                comp.to_pandas().rename(columns={"len": "leads"}),
                use_container_width=True,
                hide_index=True,
            )

    # ── Tab 4: Por empresa (NUEVO — agregado por facility) ──
    with tab_companies:
        st.subheader(f"Vista agregada por empresa ({n_empresas:,} únicas)")
        st.caption(
            "Una fila por empresa, con suma de toneladas y nº LERs distintos. "
            "Útil para evaluar el potencial total de cada cliente."
        )
        by_company = (
            filtered.group_by("facility_id_external", "facility_name", "provincia", "municipi")
            .agg(
                pl.col("score_prioridad").max().round(1).alias("Score_max"),
                pl.col("ler_code").n_unique().alias("N_LERs"),
                pl.col("quantity_tonnes_last").sum().round(1).alias("t/año_total"),
                pl.col("last_year").max().alias("Último_año"),
                pl.col("n_years_reported").max().alias("Años_rep"),
                pl.col("dist_recalc_km").min().round(1).alias("Dist_km"),
                pl.col("densidad_competencia").first().alias("Competencia"),
                pl.col("nivel_confianza_id").first().alias("Confianza"),
            )
            .sort("Score_max", descending=True)
            .head(top_n_display)
            .select(
                "Score_max",
                pl.col("facility_name").alias("Empresa"),
                pl.col("provincia").alias("Provincia"),
                pl.col("municipi").alias("Municipio"),
                "N_LERs",
                "t/año_total",
                "Último_año",
                "Años_rep",
                "Dist_km",
                "Confianza",
                "Competencia",
            )
        )
        st.dataframe(by_company.to_pandas(), use_container_width=True, height=600)

    # ════════════════════════════════════════════════════════════
    # FOOTER
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align:center; color:#666; font-size:13px; padding:10px;">
        Pipeline de inteligencia comercial B2B desde registros públicos · 8 fuentes integradas ·
        21.600 leads · 862 empresas · 6 años de histórico (2019-2024) <br>
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
