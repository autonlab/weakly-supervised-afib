from sklearn.metrics import PrecisionRecallDisplay, classification_report, precision_recall_curve
import numpy as np


# locally-defined-package imports
from data.utilities import getDataConfig

def showResults(trainingResults: dict, title=None, posLabel=None, feature_importance=False):
    resultantModel = trainingResults['m']
    newDecisionThreshold = .5
    # print(resultantModel.classes_)

    # features = [f.lower() for f in mu.getModelConfig().features_nk]

    def bestThreshold(labels, probabilities):
        precision, recall, thresholds = precision_recall_curve(labels, probabilities, pos_label=posLabel)
        f1Scores = 2*precision*recall/(precision+recall)
        bestThresh = thresholds[np.argmax(f1Scores)]
        bestF1Score = max(f1Scores)
        return bestThresh, bestF1Score
    thres, f1 = bestThreshold(trainingResults['testLabels'], trainingResults['testPredProbabilities'][:, 1])
    # print(thres, f1)
    bestThreshPreds = [1 if i else 0 for i in (trainingResults['testPredProbabilities'][:, 1] >= thres).astype(bool)]

    cr = classification_report(
        y_true=trainingResults['testLabels'],
        y_pred=trainingResults['testPredictions']
        )
    print(f'{title if title else "nameless"} classification report:\n{cr}')
    cr = classification_report(
        y_true=trainingResults['testLabels'],
        y_pred=bestThreshPreds
        )
    print(f'{title if title else "nameless"} optimized threshold classification report:\n{cr}')
    if (feature_importance):
        featImport, featImportSorted = permutationFeatureImportance(resultantModel, trainingResults['testData'], trainingResults['testLabels'], feature_subset=features, n_repeats=10)
        print('\n\n----- Feature importances -----\n\n')
        newlinetab = "\n\t"
        fiSorted = [f'{name}: {importance:.2}' for name, importance in featImportSorted]
        print(f'Top features:{newlinetab}{newlinetab.join(fiSorted)}')
    yTest = trainingResults['testLabels']
    yScore = trainingResults['testPredProbabilities'][:, 1]
    roc(yTest, yScore, f'ROC {title if title else "plot"}', dstTitle=f'roc_{title if title else "nameless"}.png', posLabel=posLabel)