"""
rules/inspections.py
====================
Reglas de calidad (expectations) para la tabla bronze_inspections.
"""

def get_rules():
    return {
        # --- Identificadores ---
        "unit_id_not_null": {
            "constraint": "unit_id IS NOT NULL",
            "tag": "inspections",
        },
        # --- Sensores físicos ---
        "temperature_valid": {
            "constraint": "temperature_celsius BETWEEN 150 AND 300",
            "tag": "inspections",
        },
        "pressure_valid": {
            "constraint": "pressure_bar BETWEEN 0.5 AND 10.0",
            "tag": "inspections",
        },
        "vibration_valid": {
            "constraint": "vibration_mm_s BETWEEN 0.0 AND 15.0",
            "tag": "inspections",
        },
        "voltage_valid": {
        "constraint": "voltage_v BETWEEN 2.5 AND 4.5",
        "tag": "inspections",
        },
        "humidity_valid": {
            "constraint": "humidity_pct BETWEEN 20.0 AND 80.0",
            "tag": "inspections",
        },
        "solder_thickness_valid": {
            "constraint": "solder_thickness_um BETWEEN 50 AND 250",
            "tag": "inspections",
        },
        "alignment_error_valid": {
            "constraint": "alignment_error_um BETWEEN 0 AND 100",
            "tag": "inspections",
        },
        # --- Parámetros de proceso ---
        "machine_id_not_null": {
            "constraint": "machine_id IS NOT NULL",
            "tag": "inspections",
        },
        "line_id_valid": {
            "constraint": "line_id IN ('LINE_A', 'LINE_B', 'LINE_C', 'LINE_D')",
            "tag": "inspections",
        },
        "shift_valid": {
            "constraint": "shift IN ('morning', 'afternoon', 'night')",
            "tag": "inspections",
        },
        "tool_wear_valid": {
            "constraint": "tool_wear_pct BETWEEN 0 AND 100",
            "tag": "inspections",
        },
        "production_speed_valid": {
            "constraint": "production_speed_pct BETWEEN 50 AND 130",
            "tag": "inspections",
        },
        # --- Timestamp ---
        "timestamp_not_null": {
            "constraint": "timestamp IS NOT NULL",
            "tag": "inspections",
        },
    }