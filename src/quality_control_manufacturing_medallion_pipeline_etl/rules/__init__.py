"""
rules/__init__.py
=================
Punto de entrada unificado para todas las reglas de calidad.
"""

from rules.inspections import get_rules as get_inspection_rules
from rules.labels import get_rules as get_label_rules


def get_all_rules():
    """Devuelve todas las reglas combinadas."""
    return {**get_inspection_rules(), **get_label_rules()}


def get_rules_by_tag(tag: str) -> dict:
    """Filtra reglas por tag (inspections | labels)."""
    return {
        name: rule
        for name, rule in get_all_rules().items()
        if rule["tag"] == tag
    }