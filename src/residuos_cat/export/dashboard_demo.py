"""Dashboard PÚBLICO en modo demo (sin contactos sensibles).

Diferencias respecto a ``dashboard.py``:
- Lee desde ``data/demo/`` (anonimizado).
- Banner permanente arriba avisando que es DEMO.
- CTAs de contacto en sidebar y footer.
- Sin tabs de "competencia con NIF/contacto"; muestra todo agregado.
- Tooltip explicando que el dataset completo se entrega con contacto+NIF tras contratar.
- Búsqueda inteligente con cascada: exact → alias marca→razón social → fuzzy match.

Pensado para deploy en Streamlit Cloud o HuggingFace Spaces.

Lanzar localmente para test:
    uv run streamlit run src/residuos_cat/export/dashboard_demo.py
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from residuos_cat.export._aliases import find_aliased_terms

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = REPO_ROOT / "data" / "demo"

LEADS_PATH = DEMO_DIR / "leads_scored_demo.parquet"
GESTORES_PATH = DEMO_DIR / "gestores_demo.parquet"

CONTACT_EMAIL = "yoelcp1988@gmail.com"
FUZZY_SCORE_CUTOFF = 85  # 0-100, WRatio rapidfuzz (más estricto evita ruido)
FUZZY_MIN_QUERY_LEN = 4  # queries más cortas NO disparan fuzzy (evita falsos positivos)

# Etiquetas amigables
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
LEAD_TYPE_LABELS = {
    "productor": "🏭 Productor",
    "gestor_sospechoso": "♻️ Gestor",
}


def _strip_accents(s: str) -> str:
    """Quita tildes y diacríticos. 'Compañía' → 'Compania', 'Damm' → 'Damm'."""
    import unicodedata

    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia haversine en km entre dos puntos."""
    radius = 6371.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _smart_search(  # noqa: PLR0911 — cascada de 4 niveles, cada uno con return propio
    df: pl.DataFrame,
    query_name: str,
    query_nif: str,
) -> tuple[pl.DataFrame, str, str]:
    """Búsqueda en cascada: NIF exacto → contains → alias → fuzzy.

    Devuelve:
        (df filtrado, tipo de match para mostrar, mensaje contextual)

    tipo de match: "exact" / "alias" / "fuzzy" / "nif" / "none" / "empty"
    """
    if not query_name.strip() and not query_nif.strip():
        return df, "empty", ""

    # 1. NIF exacto (más restrictivo)
    if query_nif.strip():
        nif_clean = query_nif.strip().upper().replace(" ", "").replace("-", "")
        m = df.filter(
            pl.col("nif").str.to_uppercase().str.replace_all(" ", "").str.replace_all("-", "")
            == nif_clean
        )
        if m.height > 0:
            n_emp = m["facility_id_external"].n_unique()
            return m, "nif", f"✅ Match exacto por NIF `{nif_clean}` → {n_emp} empresa(s)"
        # Si NIF no encontró, sigue con búsqueda por nombre si existe

    # 2. Búsqueda por nombre (sin query: salir)
    if not query_name.strip():
        return df.filter(pl.lit(False)), "none", "❌ Ningún resultado para el NIF buscado."

    q = query_name.strip()
    q_upper = q.upper()
    q_normalized = _strip_accents(q.lower())

    # 2a. Contains directo en facility_name (case-insensitive + sin tildes)
    #     Polars no tiene strip_accents nativo, así que usamos map_elements para
    #     normalizar facility_name al vuelo.
    m1 = df.filter(
        pl.col("facility_name").map_elements(
            lambda s: q_normalized in _strip_accents(s.lower()) if s else False,
            return_dtype=pl.Boolean,
        )
    )
    if m1.height > 0:
        n_emp = m1["facility_id_external"].n_unique()
        return (
            m1,
            "exact",
            (
                f"✅ **{n_emp} empresa(s)** encontradas con coincidencia directa en razón social "
                f"para `{q}` ({m1.height:,} líneas)."
            ),
        )

    # 2b. Alias marca → razón social
    aliased_terms = find_aliased_terms(q)
    if aliased_terms:
        # OR de varios contains
        condition = None
        for term in aliased_terms:
            cond = pl.col("facility_name").str.to_uppercase().str.contains(term.upper())
            condition = cond if condition is None else (condition | cond)
        if condition is not None:
            m2 = df.filter(condition)
            if m2.height > 0:
                n_emp = m2["facility_id_external"].n_unique()
                terms_str = " / ".join(f"'{t}'" for t in aliased_terms)
                return (
                    m2,
                    "alias",
                    (
                        f"✅ **{n_emp} empresa(s)** encontradas vía alias conocido: "
                        f"`{q}` → {terms_str} ({m2.height:,} líneas).\n\n"
                        f"💡 *Nuestras fuentes públicas usan razón social oficial. "
                        f"Tu marca comercial está mapeada internamente.*"
                    ),
                )

    # 2c. Fuzzy match con rapidfuzz (solo si query >= FUZZY_MIN_QUERY_LEN chars)
    #     Queries muy cortas dan demasiados falsos positivos.
    if len(q) >= FUZZY_MIN_QUERY_LEN:
        try:
            from rapidfuzz import fuzz, process

            unique_names = df["facility_name"].unique().to_list()
            unique_upper = [n.upper() for n in unique_names]
            # WRatio: weighted combination de varios scorers, más equilibrado
            matches = process.extract(
                q_upper,
                unique_upper,
                scorer=fuzz.WRatio,
                limit=10,
                score_cutoff=FUZZY_SCORE_CUTOFF,
            )
            if matches:
                matched_names = [unique_names[idx] for _, _, idx in matches]
                avg_score = sum(score for _, score, _ in matches) / len(matches)
                m3 = df.filter(pl.col("facility_name").is_in(matched_names))
                if m3.height > 0:
                    n_emp = m3["facility_id_external"].n_unique()
                    top_examples = ", ".join(f"`{n}`" for n in matched_names[:3])
                    return (
                        m3,
                        "fuzzy",
                        (
                            f"🔍 **{n_emp} empresa(s)** encontradas por **similitud aproximada** "
                            f"(score medio {avg_score:.0f}/100) para `{q}`.\n\n"
                            f"Top candidatos: {top_examples}\n\n"
                            f"💡 *Si no es la empresa que buscas, prueba con la razón social oficial "
                            f"o más texto.*"
                        ),
                    )
        except ImportError:
            # rapidfuzz no instalado, ignorar fuzzy
            pass

    # 2d. Sin resultados ni en exact, ni alias, ni fuzzy
    return (
        df.filter(pl.lit(False)),
        "none",
        (
            f"❌ **Sin coincidencias** para `{q}`.\n\n"
            f"Causas posibles: (a) la empresa no opera en Cataluña, "
            f"(b) no está autorizada en RGPGRC ni reporta a PRTR-ES, "
            f"(c) la razón social oficial es muy distinta a la marca comercial.\n\n"
            f"📧 [Avísame para añadirla al diccionario de aliases]"
            f"(mailto:{CONTACT_EMAIL}?subject=Empresa%20que%20no%20aparece%20-%20{q})"
        ),
    )


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
    # 🔍 BUSCADOR PRINCIPAL
    # ════════════════════════════════════════════════════════════
    with st.container():
        st.markdown("### 🔍 Buscar empresa específica")
        st.caption(
            "Búsqueda inteligente con 3 niveles: "
            "(1) coincidencia directa, (2) diccionario de marcas→razón social, "
            "(3) similitud aproximada (fuzzy). Cuando buscas, **se incluyen automáticamente "
            "gestores y productores**."
        )
        cols_search = st.columns([3, 2, 1])
        with cols_search[0]:
            search_name = st.text_input(
                "Nombre o marca de la empresa",
                value="",
                placeholder="Ej: GERCO, Damm, Seat, Ercros, CELSA, Nestlé...",
                help=(
                    "Busca por nombre comercial o razón social. "
                    "Si buscas 'GERCO' encuentra 'GESTION DE RESIDUOS CONTAMINANTES'. "
                    "Si buscas 'Damm' encuentra 'COMPAÑIA CERVECERA DAMM'."
                ),
                key="search_name",
            )
        with cols_search[1]:
            search_nif = st.text_input(
                "NIF (búsqueda exacta)",
                value="",
                placeholder="B60866803",
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
    # SIDEBAR — FILTROS ADICIONALES
    # ════════════════════════════════════════════════════════════
    st.sidebar.header("⚙️ Tus parámetros")
    st.sidebar.markdown("Ajusta para simular el alcance de **tu** planta de gestión de residuos.")

    with st.sidebar.expander("📍 Tu planta + radio operativo", expanded=True):
        plant_lat = st.number_input(
            "Latitud planta",
            value=41.3851,
            step=0.001,
            format="%.4f",
            help="Por defecto: Barcelona centro",
        )
        plant_lon = st.number_input(
            "Longitud planta",
            value=2.1734,
            step=0.001,
            format="%.4f",
            help="Por defecto: Barcelona centro",
        )
        radius_km = st.slider(
            "Radio operativo (km)",
            10,
            300,
            120,
            step=10,
            help="Distancia máxima rentable desde tu planta",
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

    with st.sidebar.expander("♻️ Residuo (LER)", expanded=False):
        target_lers_input = st.text_input(
            "Códigos LER objetivo (6 dígitos, coma separados)",
            value="",
            placeholder="170504, 200201, 030307",
            help="Códigos LER que tu planta está autorizada a tratar. Vacío = TODOS.",
        )
        target_lers = [t.strip() for t in target_lers_input.split(",") if t.strip()]

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

    with st.sidebar.expander("🗺️ Geografía", expanded=False):
        provincia_opciones = sorted(leads_all["provincia"].unique().drop_nulls().to_list())
        provincia_sel = st.multiselect(
            "Provincia",
            options=provincia_opciones,
            default=[],
            help="Filtra por provincia(s) catalana(s)",
        )

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

    # ── Empresas específicas (multiselect con typeahead) ──
    with st.sidebar.expander("🏢 Empresas específicas", expanded=False):
        st.caption(
            "Selecciona una o varias empresas concretas. Las demás se ocultan. "
            "Útil para auditar tu cartera de clientes/prospects."
        )
        empresas_pool = sorted(leads_all["facility_name"].unique().drop_nulls().to_list())
        empresas_sel = st.multiselect(
            f"Empresas ({len(empresas_pool):,} disponibles)",
            options=empresas_pool,
            default=[],
            help=(
                "Escribe para buscar (typeahead). Marca varias para verlas todas juntas. "
                "Si dejas vacío, no se aplica este filtro."
            ),
            placeholder="Ej: ECOPARC, GESTION DE RESIDUOS CONTAMINANTES…",
        )
        if empresas_sel:
            st.success(f"{len(empresas_sel)} empresa(s) seleccionada(s)")

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

    with st.sidebar.expander("⚙️ Opciones avanzadas", expanded=False):
        show_gestores_default = st.checkbox(
            "Incluir gestores sospechosos (por defecto cuando NO hay búsqueda)",
            value=False,
            help=(
                "Empresas con perfil de gestora (no productor). "
                "**Cuando buscas por nombre/NIF, se incluyen automáticamente** "
                "para no perder coincidencias."
            ),
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
    # APLICAR BÚSQUEDA INTELIGENTE + FILTROS
    # ════════════════════════════════════════════════════════════
    has_search = bool(search_name.strip() or search_nif.strip())

    # 1. Búsqueda inteligente (puede ser sobre TODO el dataset incluyendo gestores)
    filtered, match_type, match_msg = _smart_search(leads_all, search_name, search_nif)

    # 2. Auto-include gestores si hay búsqueda activa
    #    Si NO hay búsqueda, respeta el checkbox del usuario
    if not has_search and not show_gestores_default:
        filtered = filtered.filter(pl.col("lead_type") == "productor")

    # 3. Geografía
    if provincia_sel:
        filtered = filtered.filter(pl.col("provincia").is_in(provincia_sel))
    if comarca_sel:
        filtered = filtered.filter(pl.col("comarca").is_in(comarca_sel))
    if muni_sel:
        filtered = filtered.filter(pl.col("municipi").is_in(muni_sel))
    if empresas_sel:
        filtered = filtered.filter(pl.col("facility_name").is_in(empresas_sel))

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
            pl.col("descripcion_es")
            .str.to_lowercase()
            .str.contains(search_descripcion.lower().strip())
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

    # 7. Coord exacta
    if only_with_precise_coord:
        filtered = filtered.filter(pl.col("huella_geocod").is_in(["accidents_greus", "geocoded"]))

    # 8. Distancia y capacidad
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
    # MENSAJE DE BÚSQUEDA + KPIs
    # ════════════════════════════════════════════════════════════
    if has_search and match_msg:
        if match_type in {"exact", "nif"}:
            st.success(match_msg)
        elif match_type == "alias":
            st.info(match_msg)
        elif match_type == "fuzzy":
            st.warning(match_msg)
        elif match_type == "none":
            st.error(match_msg)

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
        st.warning(
            "Sin leads con los filtros actuales. Relaja los parámetros del lateral "
            "(amplía radio, sube rango de toneladas, o quita filtros de comarca/municipio)."
        )
        st.stop()

    # ════════════════════════════════════════════════════════════
    # TABS
    # ════════════════════════════════════════════════════════════
    tab_table, tab_map, tab_dist, tab_companies = st.tabs(
        ["📋 Top leads", "🗺️ Mapa", "📊 Distribuciones", "🏢 Por empresa"]
    )

    with tab_table:
        st.subheader(
            f"Mostrando top {min(top_n_display, n_leads):,} de {n_leads:,} leads filtrados"
        )
        sort_desc = sort_by not in ("facility_name", "dist_recalc_km")
        display = (
            filtered.sort(sort_by, descending=sort_desc, nulls_last=True)
            .head(top_n_display)
            .with_columns(
                pl.col("lead_type").replace(LEAD_TYPE_LABELS).alias("Tipo"),
            )
            .select(
                "Tipo",
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

    with tab_companies:
        st.subheader(f"Vista agregada por empresa ({n_empresas:,} únicas)")
        st.caption(
            "Una fila por empresa, con suma de toneladas y nº LERs distintos. "
            "Útil para evaluar el potencial total de cada cliente."
        )
        by_company = (
            filtered.group_by(
                "facility_id_external", "facility_name", "provincia", "municipi", "lead_type"
            )
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
            .with_columns(
                pl.col("lead_type").replace(LEAD_TYPE_LABELS).alias("Tipo"),
            )
            .select(
                "Tipo",
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
