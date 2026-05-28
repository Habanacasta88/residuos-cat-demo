"""Geocoding fallback para gestoras y leads: si la coord original es inválida
o nula, sustituir por centroide del municipio.

Detecta coords sospechosas comparando con la coord centroide del municipio
declarado: si la distancia es > 50 km, la coord original es probablemente
errónea y se reemplaza por el centroide.
"""

from __future__ import annotations

import math

import polars as pl


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia haversine en kilómetros."""
    radius = 6371.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def repair_gestores_coords(
    gestores: pl.DataFrame,
    municipis: pl.DataFrame,
    max_distance_km: float = 50.0,
) -> pl.DataFrame:
    """Repara coords de gestoras usando centroide municipi como fallback.

    Estrategia:
    - Si gestor tiene coord válida y razonable (cerca del centroide del municipi
      declarado) → conservar coord original.
    - Si gestor tiene coord pero está a >max_distance_km del municipi → reemplazar
      por centroide municipi (probable error geocoding).
    - Si gestor NO tiene coord → asignar centroide municipi.

    Añade columna `geo_source_resolved` indicando origen final.
    """
    # Normalizar nombres para join
    g = gestores.with_columns(
        pl.col("poblacio").str.to_uppercase().str.strip_chars().alias("_poblacio_norm")
    )
    m = municipis.with_columns(
        pl.col("nom_municipi").str.to_uppercase().str.strip_chars().alias("_muni_norm")
    ).select(
        "_muni_norm",
        pl.col("lat").alias("_muni_lat"),
        pl.col("lon").alias("_muni_lon"),
        pl.col("nom_comarca").alias("_muni_comarca"),
    )

    joined = g.join(m, left_on="_poblacio_norm", right_on="_muni_norm", how="left")

    # Aplicar lógica de reparación row by row
    out_lat: list[float | None] = []
    out_lon: list[float | None] = []
    out_source: list[str] = []
    out_comarca: list[str | None] = []

    for row in joined.iter_rows(named=True):
        orig_lat, orig_lon = row["lat"], row["lon"]
        muni_lat, muni_lon = row["_muni_lat"], row["_muni_lon"]
        muni_comarca = row["_muni_comarca"]

        if orig_lat is not None and orig_lon is not None:
            if muni_lat is not None and muni_lon is not None:
                # Validar distancia
                dist = _haversine_km(orig_lat, orig_lon, muni_lat, muni_lon)
                if dist <= max_distance_km:
                    # Coord original válida
                    out_lat.append(orig_lat)
                    out_lon.append(orig_lon)
                    out_source.append("original")
                else:
                    # Coord original inválida — usar centroide municipi
                    out_lat.append(muni_lat)
                    out_lon.append(muni_lon)
                    out_source.append("municipi_fallback (orig was off)")
            else:
                # No hay centroide municipi para verificar → conservar original
                out_lat.append(orig_lat)
                out_lon.append(orig_lon)
                out_source.append("original_unverified")
        elif muni_lat is not None and muni_lon is not None:
            # Sin coord original → usar centroide
            out_lat.append(muni_lat)
            out_lon.append(muni_lon)
            out_source.append("municipi_fallback")
        else:
            # Ni coord ni municipi → sin geo
            out_lat.append(None)
            out_lon.append(None)
            out_source.append("none")
        out_comarca.append(muni_comarca)

    return gestores.with_columns(
        pl.Series("lat", out_lat),
        pl.Series("lon", out_lon),
        pl.Series("geo_source_resolved", out_source),
        pl.Series("comarca_resolved", out_comarca),
    )
