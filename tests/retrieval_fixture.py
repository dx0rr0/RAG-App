"""Tiny deterministic retrieval fixture; it is not representative model data."""

DOCUMENT_IDS = ["D1", "D2", "D3", "D4", "D5"]
DOCUMENTS = {
    "D1": "insulina regula glucosa sangre células",
    "D2": "ejercicio mejora sensibilidad insulina glucosa",
    "D3": "vitamina salud ósea luz solar",
    "D4": "carbohidratos elevan niveles glucosa comida",
    "D5": "dormir mejora memoria atención concentración",
}

QUERIES = {
    "q1": {
        "text": "insulina glucosa",
        "relevance": {"D1": 3, "D2": 2, "D4": 1},
    },
    "q2": {
        "text": "ejercicio sensibilidad insulina",
        "relevance": {"D2": 3, "D1": 1},
    },
    "q3": {
        "text": "memoria concentración",
        "relevance": {"D5": 3},
    },
}

# Hand-authored deterministic scores, not embedding-model output.
VECTOR_SCORES = {
    "q1": {"D1": 0.80, "D2": 0.75, "D3": 0.10, "D4": 0.95, "D5": 0.05},
    "q2": {"D1": 0.70, "D2": 0.85, "D3": 0.03, "D4": 0.96, "D5": 0.01},
    "q3": {"D1": 0.02, "D2": 0.03, "D3": 0.45, "D4": 0.01, "D5": 0.98},
}
