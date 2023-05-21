import audata as aud
import datetime as dt
from joblib import delayed, Parallel
import numpy as np
import pandas as pd
from pathlib import Path
# locally defined packages
from . import utilities



def extractAndNormalize(src: pd.DataFrame, desired_minutes: int, h5_search_dir: Path, n_jobs=-1):
    """_summary_

    Args:
        src (pd.DataFrame): with columns fin_study_id, time
        desired_minutes (int): minute length of window
        h5_search_dir (Path): directory to find h5 source files in

    Returns:
        Tuple[List, List, np.array, float]: fin_study_ids, times, data slices, and samplerate for specified request
    """
    fins, times, slices = list(), list(), list()

    results = Parallel(n_jobs=n_jobs)(
        delayed(extractAndNormalize_itemized)(fin, subDF, desired_minutes, h5_search_dir) for fin, subDF in src.groupby('fin_study_id'))
    for (fs, ts, ss, samplerate) in results:
        fins += fs
        times += ts
        slices += ss
    return fins, times, np.array(slices), samplerate

def extractAndNormalize_itemized(fin, df, desired_minutes, h5_search_dir):
    ## TODO Remove commented code
    slices, times, fins = list(), list(), list()
    for i, row in df.iterrows():
        fin, time = row['fin_study_id'], row['time']
        start, end = time - dt.timedelta(minutes=desired_minutes), time
        try:
            x, samplerate = utilities.getSlicesFIN(
                fin,
                [utilities.HR_SERIES_II],
                #  [utilities.HR_SERIES_II# , utilities.HR_SERIES_V],
                start,
                end,
                h5_search_dir)

            if (type(x) == type(None)): continue
            x = x[0, :]

            ## iqr normalization
            mn = x.mean(0)
            q25 = np.quantile(x,0.25,axis=0)
            q75 = np.quantile(x,0.75,axis=0)
            # x = (x - mn[None, :])/(q75-q25+0.1)[None,:]
            # x = (x - mn[None, :])/(q75-q25+0.1)[None,:]
            x = (x - mn)/(q75-q25+0.1)
            x = (x - mn)/(q75-q25+0.1)
        except:
            x = np.array; time = None
        slices.append(x)
        times.append(time)
        fins.append(fin)
    print('Finished %s with %i slices' % (str(fin), len(slices)))
    return fins, times, slices, samplerate

'''
    # for i, row in src.iterrows():
    #     fin, time = row['fin_study_id'], row['time']
    #     start, end = time - dt.timedelta(minutes=desired_minutes), time
    #     x, samplerate = utilities.getSlicesFIN(
    #         fin,
    #         [utilities.HR_SERIES_II],
    #       #  [utilities.HR_SERIES_II# , utilities.HR_SERIES_V],
    #         start,
    #         end,
    #         h5_search_dir)

    #     if (type(x) == type(None)): continue
    #     x = x[0, :]

    #     ## iqr normalization
    #     mn = x.mean(0)
    #     q25 = np.quantile(x,0.25,axis=0)
    #     q75 = np.quantile(x,0.75,axis=0)
    #     # x = (x - mn[None, :])/(q75-q25+0.1)[None,:]
    #     # x = (x - mn[None, :])/(q75-q25+0.1)[None,:]
    #     x = (x - mn)/(q75-q25+0.1)
    #     x = (x - mn)/(q75-q25+0.1)
    #     slices.append(x)
    #     times.append(time)
    #     fins.append(fin)
'''