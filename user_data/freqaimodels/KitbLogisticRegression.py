"""Einfaches, nachvollziehbares FreqAI-Modell für Bot A: logistische Regression (scikit-learn).

FreqAI normalisiert die Features bereits; die Koeffizienten sind direkt als Feature-Einfluss lesbar.
"""

import logging
from typing import Any

from sklearn.linear_model import LogisticRegression

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)


class KitbLogisticRegression(BaseClassifierModel):
    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        X = data_dictionary["train_features"].to_numpy()
        y = data_dictionary["train_labels"].to_numpy()[:, 0]
        model = LogisticRegression(**self.model_training_parameters)
        model.fit(X, y, sample_weight=data_dictionary["train_weights"])
        if len(data_dictionary.get("test_features", [])):
            score = model.score(data_dictionary["test_features"].to_numpy(),
                                data_dictionary["test_labels"].to_numpy()[:, 0])
            logger.info("KitbLogisticRegression %s: Genauigkeit auf internem Eval-Set %.3f", dk.pair, score)
        return model
