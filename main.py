import argparse
import mlflow
import numpy as np
from pathlib import Path
import pandas as pd
from pprint import pformat
import pickle
from os import path


# local package loads
from data import preprocess, featurize, utilities, savers, loaders
from evaluation import test
from model.multitaskws19.train import trainlm

def memoized_consumption(function, cachedFile):
    pth = Path(__file__).parent / 'intermediate-outputs' / cachedFile
    if (path.exists(pth)):
        return loaders.loadFrom(pth)

    res = function()
    savers.saveTo(res, pth)
    return res


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
    parser.add_argument('-j','--n_jobs', type=int,
        help='Number of cores to use when parallelization is possible. Use -1 to use all available.')
    parser.add_argument('-t','--title', type=str,
        help='Identifier for model run results, takes form of `performance_indicator_{title}.png`.')
    args = parser.parse_args()
    model, src, winLength, runTitle = args.model, args.src, args.datum_size_minutes, args.title
    n_jobs = args.n_jobs
    featureSet = args.feature_set

    #get features from data config
    featureSets = dict()
    for featureDict in config.features: featureSets = {**featureSets,**featureDict}
    features = featureSets['features_nk']

    src = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / src,
        parse_dates=['time']
    )

    # fins, times, slices, samplerate = preprocess.extractAndNormalize(src, winLength, Path(args.h5_dir), N_JOBS)
    # objToStore = (fins, times, slices, samplerate)

    # normalizedSignalsPath = Path(__file__).parent / 'intermediate-outputs' / 'normalized.pkl'
    fins, times, slices, samplerate = memoized_consumption(
        lambda : preprocess.extractAndNormalize(src, winLength, Path(args.h5_dir), n_jobs),
        'normalized.pkl')
    # mlflow.log_artifact(
    #     normalizedSignalsPath,
    #     'normalized-signal-dir')
    featurized = memoized_consumption(
        lambda : featurize.nk_featurizer(fins, times, slices, samplerate, features, n_jobs),
        'featurized.pkl'
    )
    # for feat in features:
    #     print(feat)
    #     featurized[feat] = featurized[feat].apply(float)
    #     from numpy import nan, inf
    #     featurized[feat] = featurized[feat].apply(lambda x: eval(x, None, locals())[0])
    # featurized: pd.DataFrame = featurize.nk_featurizer(fins, times, slices, samplerate, features, n_jobs)

    # featPath = Path(__file__).parent / 'intermediate-outputs' / 'featurized.csv'
    # featurized.to_csv(featPath)
    # mlflow.log_artifact(
    #     featPath,
    #     'featurized-data-dir')

    print('-- featurization complete --')

    train_idx = np.random.uniform(size=len(featurized)) > .0
    #load annotation 
    test_data, test_labels = loaders.loadTestset()
    lmResultDict = trainlm(featurized.iloc[train_idx], test_data, test_labels)
    heuristicWeights = lmResultDict['model'].getHeuristicFittedWeights()
    hPath = Path(__file__).parent / 'intermediate-outputs' / ('heuristic_weights_%s.txt' % runTitle)
    savers.saveTo(pformat(dict(list(heuristicWeights))), hPath, writetype='w+')
    mlflow.log_artifact(
        hPath,
        'performance-indicators')

    artifacts = test.showResults(lmResultDict, title=runTitle)
    for a in artifacts:
        mlflow.log_artifact(
            a,
            'performance-indicators')