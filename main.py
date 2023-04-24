import argparse
import mlflow
from pathlib import Path
import pandas as pd
import pickle
from os import path


# local package loads
from data import preprocess, featurize, utilities


if __name__=='__main__':
    config = utilities.getDataConfig()
    parser = argparse.ArgumentParser(
        description='Implements afib forecasting pipeline'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=[
            'lm',
        ],
        help='Choice of labelmodel or no other choice')
    parser.add_argument('--src', type=str,
        help='Source csv of fin_study_ids and times')
    parser.add_argument('--datum_size_minutes', type=int,
        help='Window length, in minutes')
    parser.add_argument('--h5_dir', type=str,
        help='Directory holding h5 files')
    parser.add_argument('--feature_set', type=str,
        choices=[list(fDict.keys())[0] for fDict in utilities.getDataConfig().features],
        help='Feature set to use (see `data/config.yml` for possible sets)')
    args = parser.parse_args()
    model, src, winLength = args.model, args.src, args.datum_size_minutes
    featureSet = args.feature_set

    #get features from data config
    featureSets = dict()
    for featureDict in config.features: featureSets = {**featureSets,**featureDict}
    features = featureSets[featureSet]

    src = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / src,
        parse_dates=['time']
    )
    N_JOBS = -1
    normalizedSignalsPath = Path(__file__).parent / 'intermediate-outputs' / 'normalized.pkl'
    ## TODO add a memoized function wrapper instead of this messiness below
    if path.exists(normalizedSignalsPath):
        fins, times, slices, samplerate = pickle.load(open(normalizedSignalsPath, 'rb'))
    else:
        fins, times, slices, samplerate = preprocess.extractAndNormalize(src, winLength, Path(args.h5_dir), N_JOBS)
        objToStore = (fins, times, slices, samplerate)
        with open(normalizedSignalsPath, 'wb+') as writefile:
            pickle.dump(objToStore, writefile)
    mlflow.log_artifact(
        normalizedSignalsPath,
        'normalized-signal-dir')

    featurized: pd.DataFrame = featurize.nk_featurizer(fins, times, slices, samplerate, features, N_JOBS)

    featPath = Path(__file__).parent / 'intermediate-outputs' / 'featurized.csv'
    featurized.to_csv(featPath)
    mlflow.log_artifact(
        featPath,
        'featurized-data-dir')

    print('-- featurization complete --')