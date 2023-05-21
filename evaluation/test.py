import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import PrecisionRecallDisplay, classification_report, precision_recall_curve


# locally-defined-package imports
from data.utilities import getDataConfig
from evaluation.roc import roc

def showResults(trainingResults: dict, title=None, posLabel=None, feature_importance=False):
    """_summary_

    Args:
        trainingResults (dict): _description_
        title (_type_, optional): _description_. Defaults to None.
        posLabel (_type_, optional): _description_. Defaults to None.
        feature_importance (bool, optional): _description_. Defaults to False.

    Returns:
        List[str]: list of created assets, so mlflow can log those artifacts
    """
    resultantModel = trainingResults['model']
    newDecisionThreshold = .5
    # print(resultantModel.classes_)

    # features = [f.lower() for f in mu.getModelConfig().features_nk]

    def bestThreshold(labels, probabilities):
        precision, recall, thresholds = precision_recall_curve(labels, probabilities, pos_label=posLabel)
        f1Scores = 2*precision*recall/(precision+recall)
        bestThresh = thresholds[np.argmax(f1Scores)]
        bestF1Score = max(f1Scores)
        return bestThresh, bestF1Score
    thres, f1 = bestThreshold(trainingResults['labels'], trainingResults['confidences'][:, 1])
    # print(thres, f1)
    bestThreshPreds = [1 if i else 0 for i in (trainingResults['confidences'][:, 1] >= thres).astype(bool)]

    cr_defaultThreshold = classification_report(
        y_true=trainingResults['labels'],
        y_pred=trainingResults['predictions']
        )
    # print(f'{title if title else "nameless"} classification report:\n{cr}')
    cr_optimalThreshold = classification_report(
        y_true=trainingResults['labels'],
        y_pred=bestThreshPreds
        )
    # print(f'{title if title else "nameless"} optimized threshold ({thres:.2}classification report:\n{cr}')
    if (feature_importance):
        featImport, featImportSorted = permutationFeatureImportance(resultantModel, trainingResults['testData'], trainingResults['testLabels'], feature_subset=features, n_repeats=10)
        print('\n\n----- Feature importances -----\n\n')
        newlinetab = "\n\t"
        fiSorted = [f'{name}: {importance:.2}' for name, importance in featImportSorted]
        print(f'Top features:{newlinetab}{newlinetab.join(fiSorted)}')
    yTest = trainingResults['labels']
    yScore = trainingResults['confidences'][:, 1]
    crDefDst = str(Path(__file__).parent / 'assets' / f'cr_default_threshold_{title if title else "nameless"}.txt')
    crOptDst = str(Path(__file__).parent / 'assets' / f'cr_optimal_threshold_{title if title else "nameless"}.txt')
    with open(crDefDst, 'w+') as writefile:
        writefile.write(cr_defaultThreshold)
    with open(crOptDst, 'w+') as writefile:
        writefile.write(cr_optimalThreshold)
    prDst = str(Path(__file__).parent / 'assets' / f'pr_{title if title else "nameless"}.png')
    rocDst = str(Path(__file__).parent / 'assets' / f'roc_{title if title else "nameless"}.png')
    PrecisionRecallDisplay.from_predictions(yTest, yScore); plt.savefig(prDst); plt.clf()
    roc(yTest, yScore, f'ROC {title if title else "plot"}', dst= rocDst)
    return [prDst, rocDst, crDefDst, crOptDst]