"""LightGBM-Klassifikator für Bot B mit Erklärung je Vorhersage.

LightGBM liefert mit ``pred_contrib=True`` additive Beiträge je Feature (TreeSHAP) für die
positive Klasse; daraus werden die drei stärksten Einflüsse in ``kitb_explain`` geschrieben.
"""

import numpy as np
from pandas import DataFrame

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.LightGBMClassifier import LightGBMClassifier
from kitb.explain import EXPLAIN_COLUMN, top_contributions


class KitbLightGBMClassifier(LightGBMClassifier):
    def predict(self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs):
        pred_df, do_predict = super().predict(unfiltered_df, dk, **kwargs)
        X = dk.data_dictionary["prediction_features"]
        contrib = np.asarray(self.model.predict(X, pred_contrib=True))[:, :-1]  # letzte Spalte = Bias
        sign = 1.0 if list(self.model.classes_)[1] == "up" else -1.0
        pred_df[EXPLAIN_COLUMN] = top_contributions(sign * contrib, list(X.columns))
        return pred_df, do_predict
