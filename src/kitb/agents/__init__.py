"""Phase 5: spezialisierte Analyse-Agenten (lokal, regelbasiert, ohne externe API-Aufrufe).

Agenten geben nur Empfehlungen ab. Sie können Einstiege ausschließlich **unterdrücken**
(UND-Verknüpfung), nie neue Einstiege erzeugen, Positionsgrößen erhöhen oder Stoploss/Risiko-
limits ändern. Eingesetzt werden sie erst, wenn ``kitb agents-evaluate`` einen messbaren Nutzen
gegenüber der einfacheren Architektur (ohne Agenten) belegt.
"""

from kitb.agents.core import AGENT_COLUMNS, Agent, Coordinator, CycleError, default_agents

__all__ = ["AGENT_COLUMNS", "Agent", "Coordinator", "CycleError", "default_agents"]
