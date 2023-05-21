import argparse
import datetime as dt
import mlflow
from pathlib import Path
import pandas as pd
import pickle
from os import path


# local package loads
from data import preprocess, featurize, utilities, savers, loaders


if __name__=='__main__':
    config = utilities.getDataConfig()
    parser = argparse.ArgumentParser(
        description='Implements patient filter to exclude chronic afib patients'
    )
    parser.add_argument('-n', '--num_patients', type=int,
        help='Quantity of patients to sift through')
    parser.add_argument('-v','--variability_threshold', type=float,
        help='Percentage of time to allow in first two hours of patient signal')
    parser.add_argument('-h5','--h5_dir', type=str,
        help='Directory holding h5 files')
    parser.add_argument('-j','--n_jobs', type=int,
        help='Number of cores to use when parallelization is possible. Use -1 to use all available.')
    args = parser.parse_args()
    n, v, h5_dir = args.num_patients, args.variability_threshold, args.h5_dir
    n_jobs = args.n_jobs

    ## find all possible h5 files
    finFromFile = lambda x: utilities.getFINFromPath(x)
    allFins = list()
    minI, maxI = 10000, 0
    for i, h5File in enumerate(Path(args.h5_dir).glob('*.h5')):
        if i < 800:
            continue
        minI = min(i, minI)
        maxI = max(i, maxI)
        allFins.append(finFromFile(h5File))
    print('%i many patients being parsed' % (maxI - minI))
    allFins = list(set(allFins))

    if len(allFins) < n:
        print('You requested %i though there are only %i unique patients in directory %s, will revert to %i instead' %
              (n, len(allFins), str(h5_dir), len(allFins)))
        n = len(allFins)
    i = 0
    usableFins = list(); starts = list()
    while ((i < n) and (len(allFins) > 0)):
        fin = allFins.pop()
        start, stop = utilities.seriesBounds(fin, utilities.HR_SERIES_II, h5_dir)
        length_s = utilities.signalLength(fin, utilities.HR_SERIES_II, h5_dir)
        if (type(length_s) == type(None)): continue
        length = dt.timedelta(seconds = length_s)
        print('%s total length: %s, Start from stop: %s' % (str(fin), str(length), str(stop-start)))
        if length < dt.timedelta(hours=2):
            continue
        usableFins.append(fin)
        starts.append(start)
        i += 1
    dfs = list()
    for i in range(len(usableFins)):
        fin, start = usableFins[i], starts[i]
        ts = pd.date_range(start, start+dt.timedelta(hours=2), freq=dt.timedelta(minutes=10))
        dfs.append(pd.DataFrame({
            'fin_study_id': [fin for i in range(len(ts))],
            'time': map(lambda x: x + dt.timedelta(minutes=10), list(ts))
        }))
    dfToFeaturize = pd.concat(dfs)
    print(dfToFeaturize)
    print('Extracting %i slices and normalizing.' % len(dfToFeaturize))
    fins, times, slices, samplerate = preprocess.extractAndNormalize(dfToFeaturize, 10, Path(args.h5_dir), n_jobs)
    print('Featurizing %i data points.' % len(fins))
    config = utilities.getDataConfig()
    featureSets = dict()
    for featureDict in config.features: featureSets = {**featureSets,**featureDict}
    features = featureSets['features_nk']
    featurized: pd.DataFrame = featurize.nk_featurizer(fins, times, slices, samplerate, features, n_jobs)


    featPath = Path(__file__).parent / 'intermediate-outputs' / ('120_minutes_%i_patients.csv' % len(usableFins))
    featurized.to_csv(featPath)
    mlflow.log_artifact(
        featPath,
        'featurized-data-dir')

    print('-- featurization complete --')