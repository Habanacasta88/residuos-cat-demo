"""Mapping de marcas/nombres comerciales → razones sociales en el dataset.

Permite al cliente buscar por la marca que conoce (GERCO, Damm, CELSA) y
encontrar la empresa aunque el dataset la tenga registrada con razón social
distinta (porque las fuentes públicas usan razón social oficial PRTR-ES/ARC).

Cada clave (marca) puede mapear a una lista de patrones — se hace
case-insensitive contains contra ``facility_name``.

Pensado para crecer iterativamente: cada vez que un prospect pregunte por
una empresa que no encuentra, añadirla aquí y redesplegar.
"""

from __future__ import annotations

ALIASES: dict[str, list[str]] = {
    # Gestoras conocidas
    "GERCO": ["GESTION DE RESIDUOS CONTAMINANTES"],
    "PREZERO": ["PREZERO"],
    "FERIMET": ["FERIMET"],
    "VIUDA DE LAURO": ["VIUDA DE LAURO CLARIANA"],
    "ECOPARC": ["ECOPARC 1", "ECOPARC 2", "ECOPARC 3", "ECOPARC 4"],
    "MARCEL NAVARRO": ["RECUPERACIONS MARCEL NAVARRO"],
    "EDAR BESOS": ["EDAR BESÒS"],
    "EDAR EL PRAT": ["EDAR EL PRAT"],
    "TERSA": ["PLANTA DE VALORITZACIO ENERGETICA DE SANT ADRIA DE BESOS"],
    "TRARGISA": ["CENTRE DE VALORITZACIO DE LLORET DE MAR"],
    "CTR VALLES": ["UTE CTR VALLÈS"],
    "CTR MARESME": ["CENTRE INTEGRAL DE VALORITZACIO DE RESIDUS DEL MARESME"],
    # Industrial — productoras grandes
    "CELSA": ["COMPAÑIA ESPAÑOLA DE LAMINACION", "CELSA"],
    "DAMM": ["COMPAÑIA CERVECERA DAMM", "ESTRELLA DE LEVANTE"],
    "ESTRELLA DAMM": ["COMPAÑIA CERVECERA DAMM"],
    "SEAT": ["SEAT"],
    "VOLKSWAGEN": ["SEAT", "VOLKSWAGEN"],
    "NESTLE": ["NESTLE"],
    "NESTLÉ": ["NESTLE"],
    "ROCA": ["ROCA SANITARIO", "ROCA CORPORACION"],
    "ERCROS": ["ERCROS"],
    "HENKEL": ["HENKEL IBERICA"],
    "BASF": ["BASF ESPAÑOLA"],
    "BAYER": ["BAYER HISPANIA"],
    "FREIXENET": ["FREIXENET"],
    "CODORNIU": ["CODORNIU"],
    "CODORNÍU": ["CODORNIU"],
    "MIQUEL Y COSTAS": ["MIQUEL Y COSTAS"],
    "VICHY CATALAN": ["VICHY CATALAN"],
    "FONT VELLA": ["AGUAS DE FONT VELLA"],
    "REPSOL": ["REPSOL"],
    "CEPSA": ["CEPSA"],
    "ALMIRALL": ["ALMIRALL"],
    "ESTEVE": ["ESTEVE"],
    "GRIFOLS": ["GRIFOLS"],
    "FERRER": ["FERRER INTERNACIONAL"],
    "URIACH": ["URIACH"],
    "REIG JOFRE": ["REIG JOFRE"],
    "TORRAS PAPEL": ["TORRAS PAPEL"],
    "LECITRAILER": ["LECITRAILER"],
    "FICOSA": ["FICOSA"],
    "FAGOR": ["FAGOR"],
    "AGBAR": ["AGBAR", "AGUAS DE BARCELONA"],
}


def find_aliased_terms(query: str) -> list[str]:
    """Si la query coincide con una marca conocida, devuelve los patrones de
    razón social a buscar. Sino devuelve lista vacía.

    Búsqueda case-insensitive contra las CLAVES del diccionario.
    """
    if not query or not query.strip():
        return []
    q = query.strip().upper()
    # Match exacto primero
    if q in ALIASES:
        return ALIASES[q]
    # Match parcial: la query CONTIENE la clave o la clave CONTIENE la query
    matched: list[str] = []
    for marca, razones in ALIASES.items():
        if marca in q or q in marca:
            matched.extend(razones)
    return matched
