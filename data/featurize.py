import datetime as dt
import numpy as np
import pandas as pd

# local package loads
from . import computers
def nk_featurizer(fins: list[int], times: list[dt.datetime], slices: np.array, samplerate: float, features: list[str]):
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
    featurizedResult = {
        'fin_study_id': list(),
        'time': list(),
    }

    for feature in features:
        featurizedResult[feature] = list()
    for i, series_II_slice in enumerate(slices[:,0]):
        shortSegmentFeatures = computers.featurize_nk(series_II_slice, samplerate)
        longSegmentFeatures = computers.featurize_longertimewindow_nk(series_II_slice, samplerate)
        if (shortSegmentFeatures and longSegmentFeatures):
            for feature in shortSegmentFeatures:
                featurizedResult[feature].append(shortSegmentFeatures[feature])
            for feature in longSegmentFeatures:
                featurizedResult[feature].append(longSegmentFeatures[feature])
            featurizedResult['time'].append(times[i])
            featurizedResult['fin_study_id'].append(fins[i])
    return pd.DataFrame(featurizedResult)
