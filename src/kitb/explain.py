"""Feature-Einfluss je Vorhersage als kompakter Text, z. B. ``%-rsi-14:+0.42;%-roc-14:-0.10``.

Wird von den FreqAI-Modellen als zusätzliche Vorhersagespalte ``kitb_explain`` ausgegeben und
landet so im Entscheidungsprotokoll.
"""

import numpy as np

EXPLAIN_COLUMN = "kitb_explain"


def top_contributions(contrib: np.ndarray, names: list[str], k: int = 3) -> list[str]:
    """contrib: (n_rows, n_features) Beitrag jedes Features zur Klasse ``up``."""
    contrib = np.asarray(contrib, dtype=float)
    out = []
    for row in contrib:
        idx = np.argsort(-np.abs(row))[:k]
        out.append(";".join(f"{names[i]}:{row[i]:+.3f}" for i in idx))
    return out


def parse(text: str) -> dict[str, float]:
    if not isinstance(text, str) or not text:
        return {}
    return {name: float(val) for name, val in (part.rsplit(":", 1) for part in text.split(";"))}
