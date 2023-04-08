import audata as aud
import numpy as np
from pathlib import Path
import pandas as pd
from joblib import delayed, Parallel
import datetime as dt
# locally defined packages
from . import utilities


def extractAndNormalize(src: pd.DataFrame, desired_minutes: int, h5_search_dir: Path) -> np.array:
    """_summary_

    Args:
        src (pd.DataFrame): with columns fin_study_id, time
        desired_minutes (int): minute length of window
        h5_search_dir (Path): directory to find h5 source files in

    Returns:
        np.array: normalized source signals at specified times and window length
    """
    fins, times, slices = [], [], []

    for i, row in src.iterrows():
        fin, time = row['fin_study_id'], row['time']
        start, end = time - dt.timedelta(minutes=desired_minutes), time
        slice, samplerate = utilities.getSlicesFIN(
            fin,
            [utilities.HR_SERIES_II, utilities.HR_SERIES_V],
            start,
            end,
            h5_search_dir)

        if (type(slice) == type(None)): continue
        print(i)
        print(slice.shape)
        print(slice)

        ## iqr normalization
        x = slice
        mn = x.mean(0)
        q25 = np.quantile(x,0.25,axis=0)
        q75 = np.quantile(x,0.75,axis=0)
        x = (x - mn[None, :])/(q75-q25+0.1)[None,:]
        slices.append(slice)
        times.append(time)
        fins.append(fin)
    return fins, times, np.array(slices), samplerate