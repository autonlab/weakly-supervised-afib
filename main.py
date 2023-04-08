import argparse
from pathlib import Path
import pandas as pd

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
    print(args)
    model, src, winLength = args.model, args.src, args.datum_size_minutes
    featureSet = args.feature_set

    #get features from data config
    featureSets = dict()
    for featureDict in config.features: featureSets = {**featureSets,**featureDict}
    print(featureSets)
    features = featureSets[featureSet]
    print(features)

    src = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / src,
        parse_dates=['time']
    )
    fins, times, slices, samplerate = preprocess.extractAndNormalize(src, winLength, Path(args.h5_dir))
    featurized = featurize.nk_featurizer(fins, times, slices, samplerate, features)
    print(slices.shape)
    print(slices)