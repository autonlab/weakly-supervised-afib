import pandas as pd
from pathlib import Path

from .labelmodel import LabelModelCustom

# see [here](https://arxiv.org/abs/1810.02840)
def trainlm(train_features: pd.DataFrame, test_features: pd.DataFrame, test_labels):
    fitModel = LabelModelCustom()
    fitModel.fit(train_features)
    predictions = fitModel.predict(test_features)
    confidences = fitModel.predict_proba(test_features)
    return {
        'model': fitModel,
        'predictions': predictions,
        'confidences': confidences,
        'labels': test_labels
    }