from pathlib import Path
import pandas as pd
import pickle
from typing import Union

## local package loads
import data.utilities as du

def loadFrom(src: Union[Path, str]):
    return pickle.load(open(src, 'rb'))

def loadTestset():
    df = pd.read_csv(Path(__file__).parent / 'assets' / 'final_annotations_featurized_nk.csv')
    def labelSifter(l):
        if l == 'ATRIAL_FIBRILLATION':
            return 1
        return 0
    labels = df['label'].apply(labelSifter)
    featureSets = dict()
    for featureDict in du.getDataConfig().features: featureSets = {**featureSets,**featureDict}
    features = featureSets['features_nk']
    data = df[features]
    return data, labels

def loadTestsetWaveforms():
    df = pd.read_csv(Path(__file__).parent / 'assets' / 'final_annotations_featurized_nk.csv',
                     parse_dates=['start', 'stop'])
    def labelSifter(l):
        if l == 'ATRIAL_FIBRILLATION':
            return 1
        return 0
    labels = df['label'].apply(labelSifter)
    res = list()
    h5_search_dir = '/zfs2/mladi/viewer/projects/mladi/originals/'
    for idx, row in df.iterrows():
        fin, start, end = row['fin_study_id'], row['start'], row['stop']
        x, samplerate = du.getSlicesFIN(
            fin,
            [du.HR_SERIES_II],
            #  [utilities.HR_SERIES_II# , utilities.HR_SERIES_V],
            start,
            end,
            h5_search_dir)
        res.append((x[0, :], samplerate))
    return df, res