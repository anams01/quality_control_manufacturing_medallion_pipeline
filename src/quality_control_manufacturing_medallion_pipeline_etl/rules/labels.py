"""
rules/labels.py
===============
Reglas de calidad (expectations) para la tabla bronze_labels.
"""

def get_rules():
    return {
        "unit_id_not_null": {
            "constraint": "unit_id IS NOT NULL",
            "tag": "labels",
        },
        "is_defective_valid": {
            "constraint": "is_defective IN (0, 1)",
            "tag": "labels",
        },
        "label_available_date_not_null": {
            "constraint": "label_available_date IS NOT NULL",
            "tag": "labels",
        },
    }