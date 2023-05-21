import datetime as dt
from functools import reduce
from joblib import delayed, Parallel
import numpy as np
import pandas as pd

# local package loads
from . import computers
def nk_featurizer(fins: list[int], times: list[dt.datetime], slices: np.array, samplerate: float, features: list[str], n_jobs=-1):
    """Given fins, times, and data slices returns neurokit featurization in dataframe.

    Args:
        fins (list[int]): fin_study_ids of patients
        times (list[dt.datetime]): corresponding datum endpoints
        slices (np.array): data slices for selected series
        samplerate (float): signal samplerate
        features (list[str]): list of desired features

    Returns:
        _type_: _description_
    """
    ## TODO Remove single threaded code or refactor into more readable function
    itemized = Parallel(n_jobs=n_jobs)(delayed(nk_featurizer_itemized)(fins[i], times[i], slices[i], samplerate, features) for i in range(len(fins)))
    def dictionaryReducer(d1, d2):
        for k in d1.keys():
            d1[k].append(d2[k])
        return d1
    initialDict = dict(
        [(k, []) for k in itemized[0].keys()]
        )
    dResult = reduce(dictionaryReducer, itemized, initialDict)
    return pd.DataFrame(dResult)

def nk_featurizer_itemized(fin, time, slice, samplerate, features):
    featurizedResult = dict()
    featurizedResult['time'] = list(); featurizedResult['fin_study_id'] = list()
    for feature in features:
        featurizedResult[feature] = list()
    shortSegmentFeatures = computers.featurize_nk(slice, samplerate)
    longSegmentFeatures = computers.featurize_longertimewindow_nk(slice, samplerate)
    if (shortSegmentFeatures and longSegmentFeatures):
        for feature in shortSegmentFeatures:
            featurizedResult[feature].append(shortSegmentFeatures[feature])
        for feature in longSegmentFeatures:
            featurizedResult[feature].append(longSegmentFeatures[feature])
        featurizedResult['time'].append(time)
        featurizedResult['fin_study_id'].append(fin)
    return featurizedResult

'''
    ## initial impl below
    # featurizedResult = {
    #     'fin_study_id': list(),
    #     'time': list(),
    # }

    # for feature in features:
    #     featurizedResult[feature] = list()
    # for i, series_II_slice in enumerate(slices):
    #     shortSegmentFeatures = computers.featurize_nk(series_II_slice, samplerate)
    #     longSegmentFeatures = computers.featurize_longertimewindow_nk(series_II_slice, samplerate)
    #     if (shortSegmentFeatures and longSegmentFeatures):
    #         for feature in shortSegmentFeatures:
    #             featurizedResult[feature].append(shortSegmentFeatures[feature])
    #         for feature in longSegmentFeatures:
    #             featurizedResult[feature].append(longSegmentFeatures[feature])
    #         featurizedResult['time'].append(times[i])
    #         featurizedResult['fin_study_id'].append(fins[i])
    # return pd.DataFrame(featurizedResult)
'''