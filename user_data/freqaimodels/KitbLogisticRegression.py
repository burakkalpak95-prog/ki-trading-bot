"""Einfaches, nachvollziehbares FreqAI-Modell für Bot A: logistische Regression (scikit-learn).

FreqAI normalisiert die Features bereits. Der Beitrag eines Features zur Klasse ``up`` ist
``Koeffizient × Featurewert`` und wird je Vorhersage in ``kitb_explain`` ausgegeben.
"""

import logging
from typing import Any

import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from kitb.explain import EXPLAIN_COLUMN, top_contributions

logger = logging.getLogger(__name__)


class KitbLogisticRegression(BaseClassifierModel):
    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"].to_numpy()[:, 0]
        model = LogisticRegression(**self.model_training_parameters)
        model.fit(X, y, sample_weight=data_dictionary["train_weights"])
        if len(data_dictionary.get("test_features", [])):
            score = model.score(data_dictionary["test_features"],
                                data_dictionary["test_labels"].to_numpy()[:, 0])
            logger.info("KitbLogisticRegression %s: Genauigkeit auf internem Eval-Set %.3f", dk.pair, score)
        return model

    def predict(self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs):
        pred_df, do_predict = super().predict(unfiltered_df, dk, **kwargs)
        X = dk.data_dictionary["prediction_features"]
        coef = self.model.coef_[0]  # binär: Koeffizienten für classes_[1]
        sign = 1.0 if list(self.model.classes_)[1] == "up" else -1.0
        contrib = sign * X.to_numpy() * coef
        pred_df[EXPLAIN_COLUMN] = top_contributions(contrib, list(X.columns))
        return pred_df, do_predict
