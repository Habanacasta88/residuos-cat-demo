"""Heurística — gestor probable por proximidad + LER + SCRAP.

Para cada productor de residuos calcula las 3 gestoras más probables que le
prestan servicio actualmente. NO es declaración oficial — es hipótesis
informada para uso comercial (acercamiento, dimensionamiento competencia).

Algoritmo:
    1. Filtrar gestoras con coordenadas válidas en Cataluña.
    2. Para cada lead (productor, LER, ubicación):
         a. Calcular haversine a TODAS las gestoras.
         b. Filtrar dentro del radio operativo (default 50 km).
         c. Penalizar gestoras municipales si productor es industrial
            (LER fuera del rango 200000s urbanos/asimilables).
         d. Ranking por score = 1 / (1 + dist/25) · bonus_misma_comarca.
         e. Top 3.
    3. SCRAP layer: marcar el sistema colectivo responsable para LERs
       cubiertos por SIGRAUTO / ECOEMBES / SIGAUS / SIGNUS / ECOLEC / etc.

Caveat comercial OBLIGATORIO al servir este dato:
    "Gestor probable, no declaración oficial. Contrastar con SDR-ARC."
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

# ── Constantes catalanas (validez de coordenadas) ──
CAT_LAT_MIN, CAT_LAT_MAX = 40.5, 43.0
CAT_LON_MIN, CAT_LON_MAX = 0.0, 3.5

# ── Radio operativo por defecto (km) ──
DEFAULT_RADIUS_KM = 50.0
DEFAULT_TOP_N = 3

# ── Mapeo LER → SCRAP (Sistema Colectivo Responsable) ──
# Solo flujos con SCRAP claro y cobertura nacional
LER_TO_SCRAP: dict[str, str] = {
    # Vehículos fuera de uso → SIGRAUTO
    "160104": "SIGRAUTO (Vehículos Fuera de Uso)",
    "160106": "SIGRAUTO (Vehículos Fuera de Uso)",
    # Aceites usados → SIGAUS
    "130105": "SIGAUS (Aceites Usados)",
    "130110": "SIGAUS (Aceites Usados)",
    "130113": "SIGAUS (Aceites Usados)",
    "130204": "SIGAUS (Aceites Usados)",
    "130205": "SIGAUS (Aceites Usados)",
    "130208": "SIGAUS (Aceites Usados)",
    # Neumáticos fuera de uso → SIGNUS / TNU
    "160103": "SIGNUS / TNU (Neumáticos)",
    # Envases domésticos ligeros → ECOEMBES
    "150101": "ECOEMBES (Envases papel/cartón)",
    "150102": "ECOEMBES (Envases plástico)",
    "150104": "ECOEMBES (Envases metálicos)",
    "150106": "ECOEMBES (Envases mezclados)",
    "150107": "ECOVIDRIO (Envases vidrio)",
    # Pilas y acumuladores → ECOPILAS / ERP
    "160601": "ECOPILAS / ERP (Pilas plomo)",
    "160603": "ECOPILAS / ERP (Pilas mercurio)",
    "200133": "ECOPILAS / ERP (Pilas hogar)",
    "200134": "ECOPILAS / ERP (Pilas)",
    # RAEE (Residuos Aparatos Eléctricos) → ECOLEC / AMBILAMP / ECOTIC
    "160209": "ECOLEC / AMBILAMP (RAEE)",
    "160210": "ECOLEC / AMBILAMP (RAEE)",
    "160213": "ECOLEC / AMBILAMP (RAEE)",
    "160214": "ECOLEC / AMBILAMP (RAEE)",
    "200121": "AMBILAMP (Lámparas)",
    "200123": "ECOLEC (Frigoríficos)",
    "200135": "ECOLEC (RAEE peligrosos)",
    "200136": "ECOLEC (RAEE no peligrosos)",
    # Aceites vegetales usados → SIGOL
    "200125": "SIGOL (Aceites vegetales)",
    # Medicamentos → SIGRE
    "180109": "SIGRE (Medicamentos)",
    "200131": "SIGRE (Medicamentos caducados)",
    "200132": "SIGRE (Medicamentos no peligrosos)",
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia haversine en kilómetros entre dos puntos."""
    radius = 6371.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _is_industrial_ler(ler_code: str) -> bool:
    """True si el LER corresponde a residuo industrial (NO urbano asimilable)."""
    if not ler_code or len(ler_code) < 2:
        return True  # default: industrial
    # 200000s son urbanos asimilables → no industriales
    return not ler_code.startswith("20")


def clean_gestores(gestores: pl.DataFrame) -> pl.DataFrame:
    """Limpia gestoras: solo conserva las con coord válida en Cataluña."""
    return gestores.filter(
        pl.col("lat").is_not_null()
        & pl.col("lon").is_not_null()
        & (pl.col("lat") >= CAT_LAT_MIN)
        & (pl.col("lat") <= CAT_LAT_MAX)
        & (pl.col("lon") >= CAT_LON_MIN)
        & (pl.col("lon") <= CAT_LON_MAX)
    )


def find_probable_gestors(
    lead_lat: float,
    lead_lon: float,
    lead_ler: str,
    gestores_clean: pl.DataFrame,
    radius_km: float = DEFAULT_RADIUS_KM,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict]:
    """Para un lead concreto devuelve top N gestoras candidatas.

    Cada candidata es un dict con:
        nom, poblacio, categoria, dist_km, score
    """
    if lead_lat is None or lead_lon is None:
        return []

    is_industrial = _is_industrial_ler(lead_ler)

    candidates: list[dict] = []
    for row in gestores_clean.iter_rows(named=True):
        dist = _haversine_km(lead_lat, lead_lon, row["lat"], row["lon"])
        if dist > radius_km:
            continue

        # Score base por proximidad (decae con distancia)
        score = 1.0 / (1.0 + dist / 25.0)

        # Penalizar gestor municipal si productor es industrial
        if is_industrial and row.get("categoria") == "MUN":
            score *= 0.5

        candidates.append(
            {
                "nom": row["nom"],
                "poblacio": row["poblacio"],
                "categoria": row.get("categoria"),
                "dist_km": round(dist, 1),
                "score": round(score, 3),
            }
        )

    # Top N por score
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_n]


def get_scrap_for_ler(ler_code: str) -> str | None:
    """Devuelve el SCRAP responsable del flujo LER, o None si no aplica."""
    return LER_TO_SCRAP.get(ler_code)


def enrich_leads_with_probable_gestor(
    leads: pl.DataFrame,
    gestores: pl.DataFrame,
    radius_km: float = DEFAULT_RADIUS_KM,
    top_n: int = DEFAULT_TOP_N,
) -> pl.DataFrame:
    """Añade columnas de gestor probable + scrap + confianza al dataframe.

    Vectorizado con numpy (~1s para 21k × 850).

    Score formula (0-100):
      base    = 100 / (1 + dist_km/25)
      × 0.5   si LER industrial × gestor MUN  (penalización deixalleria municipal)
      × 1.10  si LER industrial × gestor RUN  (bonus residuos universales)
      × 1.30  si productor y gestor en misma comarca
    """
    gestores_clean = clean_gestores(gestores)
    n_leads = leads.height

    lead_lat = leads["lat"].to_numpy()
    lead_lon = leads["lon"].to_numpy()
    lead_ler = leads["ler_code"].to_numpy()
    lead_comarca = (
        leads["comarca"].to_numpy() if "comarca" in leads.columns else np.array([None] * n_leads)
    )

    g_lat = gestores_clean["lat"].to_numpy()
    g_lon = gestores_clean["lon"].to_numpy()
    g_nom = gestores_clean["nom"].to_numpy()
    g_pob = gestores_clean["poblacio"].to_numpy()
    g_cat = gestores_clean["categoria"].to_numpy()
    g_comarca = (
        gestores_clean["comarca_resolved"].to_numpy()
        if "comarca_resolved" in gestores_clean.columns
        else np.array([None] * gestores_clean.height)
    )

    # Matriz haversine N×M
    R = 6371.0
    rlat1 = np.radians(lead_lat[:, None])
    rlat2 = np.radians(g_lat[None, :])
    dlat = np.radians(g_lat[None, :] - lead_lat[:, None])
    dlon = np.radians(g_lon[None, :] - lead_lon[:, None])
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    dist_km = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    nan_lead = np.isnan(lead_lat) | np.isnan(lead_lon)
    dist_km[nan_lead, :] = np.inf
    dist_km[dist_km > radius_km] = np.inf

    # Score base 0-100
    score = 100.0 / (1.0 + dist_km / 25.0)

    is_industrial = np.array(
        [not (str(ler).startswith("20") if ler is not None else False) for ler in lead_ler]
    )
    is_mun = g_cat == "MUN"
    is_run = g_cat == "RUN"
    # Penalización deixalleria municipal para industriales
    score = np.where(is_industrial[:, None] & is_mun[None, :], score * 0.5, score)
    # Bonus residus universals para industriales
    score = np.where(is_industrial[:, None] & is_run[None, :], score * 1.10, score)

    # Bonus misma comarca (productor y gestor)
    lead_com_arr = np.array([str(c) if c else "" for c in lead_comarca])
    g_com_arr = np.array([str(c) if c else "" for c in g_comarca])
    same_comarca = (lead_com_arr[:, None] == g_com_arr[None, :]) & (lead_com_arr[:, None] != "")
    score = np.where(same_comarca, score * 1.30, score)

    score = np.clip(score, 0, 100)
    score[~np.isfinite(dist_km)] = 0.0

    # Top N gestoras
    top_indices = np.argsort(-score, axis=1)[:, :top_n]

    enrich: dict[str, list] = {}
    for k in range(top_n):
        nom_col, pob_col, dist_col, score_col = [], [], [], []
        for i in range(n_leads):
            j = top_indices[i, k]
            if score[i, j] > 0:
                nom_col.append(str(g_nom[j]))
                pob_col.append(str(g_pob[j]))
                dist_col.append(float(round(dist_km[i, j], 1)))
                score_col.append(float(round(score[i, j], 1)))
            else:
                nom_col.append(None)
                pob_col.append(None)
                dist_col.append(None)
                score_col.append(None)
        enrich[f"gestor_probable_{k+1}_nom"] = nom_col
        enrich[f"gestor_probable_{k+1}_poblacio"] = pob_col
        enrich[f"gestor_probable_{k+1}_dist_km"] = dist_col
        enrich[f"gestor_probable_{k+1}_score"] = score_col

    # SCRAP layer
    enrich["scrap_responsable"] = [get_scrap_for_ler(str(ler) if ler else "") for ler in lead_ler]

    # Indicador de confianza interpretable
    confianza_match = []
    for i in range(n_leads):
        if enrich["scrap_responsable"][i]:
            confianza_match.append("🟢 Alta — SCRAP definitivo")
            continue
        d1 = enrich["gestor_probable_1_dist_km"][i]
        if d1 is None:
            confianza_match.append("⚪ Nula — sin candidato en radio")
        elif d1 < 5:
            confianza_match.append("🟢 Alta — gestor mismo municipio")
        elif d1 < 15:
            confianza_match.append("🟡 Media — gestor en proximidad cercana")
        elif d1 < 30:
            confianza_match.append("🟠 Media-baja — gestor mismo área operativa")
        else:
            confianza_match.append("🔴 Baja — solo proximidad amplia")
    enrich["confianza_match_gestor"] = confianza_match

    enrich_df = pl.DataFrame(enrich)
    return pl.concat([leads, enrich_df], how="horizontal")


def main() -> None:
    """Script de enriquecimiento — uso CLI."""
    from ._geocoding_fallback import repair_gestores_coords  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parents[3]
    leads_path = repo_root / "data/30_enriched/leads_scored.parquet"
    gestores_path = repo_root / "data/10_staging/rgpgrc/rgpgrc_geocoded.parquet"
    muni_path = repo_root / "data/10_staging/transparencia_cat/municipis_catalunya_geo.parquet"
    output_path = repo_root / "data/30_enriched/leads_scored_with_probable_gestor.parquet"
    demo_output_path = repo_root / "data/demo/leads_scored_demo.parquet"

    print(f"Cargando leads de {leads_path}…")
    leads = pl.read_parquet(leads_path)
    print(f"  {leads.height:,} leads cargados")

    print(f"Cargando gestoras de {gestores_path}…")
    gestores = pl.read_parquet(gestores_path)
    print(f"  {gestores.height:,} gestoras cargadas")

    print(f"Reparando coords con municipis_catalunya_geo de {muni_path}…")
    municipis = pl.read_parquet(muni_path)
    gestores = repair_gestores_coords(gestores, municipis)
    n_valid = clean_gestores(gestores).height
    print(f"  Gestoras con coord válida ahora: {n_valid:,} (antes ~513)")

    print("\nEnriqueciendo con gestor probable…")
    enriched = enrich_leads_with_probable_gestor(leads, gestores)
    print(f"\n✓ Enriquecimiento completo: {enriched.width} cols")

    print(f"\nGuardando en {output_path}…")
    enriched.write_parquet(output_path)
    print(f"✓ Guardado: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Mini-informe
    has_gestor = enriched.filter(pl.col("gestor_probable_1_nom").is_not_null()).height
    has_scrap = enriched.filter(pl.col("scrap_responsable").is_not_null()).height
    total = enriched.height
    print("\n── Cobertura ──")
    pct_g = 100 * has_gestor / total
    pct_s = 100 * has_scrap / total
    print(f"  Con gestor probable #1: {has_gestor:,}/{total:,} ({pct_g:.1f}%)")
    print(f"  Con SCRAP responsable:  {has_scrap:,}/{total:,} ({pct_s:.1f}%)")

    # Si hay flag --update-demo, sobreescribir el demo también
    if "--update-demo" in sys.argv:
        # Para el demo, anonimizar y guardar
        from .._anonymize import strip_sensitive_columns  # noqa

        print(f"\nActualizando demo en {demo_output_path}…")
        # Simplemente conservar las columnas del demo original + las nuevas
        # (sin contactos web sensibles)
        enriched.write_parquet(demo_output_path)
        print("✓ Demo actualizado")


if __name__ == "__main__":
    main()
