"""Mixin: beschränkt die Einstiege einer bestehenden Strategie auf die vom Coordinator erlaubten Kerzen."""

from pandas import DataFrame

from kitb.agents.core import Coordinator, default_agents


class AgentGateMixin:
    """Vor die Basisstrategie schalten: ``class X(AgentGateMixin, BaseStrategy)``.

    Nur UND-Verknüpfung auf ``enter_long``; Ausstiege, Stoploss, Positionsgröße bleiben unverändert.
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        return Coordinator(default_agents()).run(dataframe)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        if "enter_long" in dataframe:
            blocked = (dataframe["enter_long"] == 1) & ~dataframe["agent_allow_entry"].astype(bool)
            dataframe.loc[blocked, ["enter_long", "enter_tag"]] = (0, None)
        return dataframe
