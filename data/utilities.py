
import audata as aud
import datetime as dt
import numpy as np
import os
from pathlib import Path
import pytz
import random
import yaml

"""
    Miscellaneous utilities for afib detection and risk prediction model formation
"""

## TODO organize this file

#https://stackoverflow.com/questions/6866600/how-to-parse-read-a-yaml-file-into-a-python-object
class Struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

def getDataConfig():
    with open(Path(__file__).parent / 'config.yml', 'r') as yamlfile:
        configContents = yaml.safe_load(yamlfile)
    return Struct(**configContents)

HR_SERIES_II = "/data/waveforms/II"
HR_SERIES_V = "/data/waveforms/V"
B2B_SERIES = "/data/numerics/HR.BeatToBeat"
def getSF(series):
    """Infer sampling frequency from series (TODO: can be more efficient)

    Args:
        series (Union[List[dt.datetime],np.array[dt.datetime]]): time series

    Returns:
        integer: sampling frequency for series
    """
    distSums, numDists = 0, 0
    for i in range(len(series)-1):
        cur, next = series[i:i+2]
        numDists += 1
        distSums += (next - cur).total_seconds()
        # print(next-cur)
    samplingFrequency = numDists/distSums#500#samples/s   numDists / distSums
    samplingFrequency = round(samplingFrequency)
    return samplingFrequency

def binary_search_timeseries(searchtime, auf, lo=0):
    length = auf[HR_SERIES_II].nrow
    def get(i):
        b = auf[HR_SERIES_II][i]['time'].item().to_pydatetime()
        b = b.replace(tzinfo=pytz.UTC)
        return b
    if searchtime < get(lo):
        return lo
    elif searchtime > get(length-1):
        return length-1
    hi = length-1
    while (lo <= hi):
        mid = (hi + lo) // 2

        if (searchtime < get(mid)):
            hi = mid-1
        elif (searchtime > get(mid)):
            lo = mid+1
        else:
            return mid
    if (abs(get(lo)-searchtime) < abs(get(hi)-searchtime)):
        return lo
    else:
        return hi

def getSliceFIN(fin, series, starttime, stoptime, searchDir):
    file = findFileByFIN(str(fin), searchDir)
    dataslice, samplerate = getSlice(file, series, starttime, stoptime)
    return dataslice, samplerate

def getSlicesFIN(fin, series, starttime, stoptime, searchDir):
    file = findFileByFIN(str(fin), searchDir)

    if (type(file) == type(None)): return None, None

    dataslice, samplerate = getSliceMultiseries(file, series, starttime, stoptime)
    return dataslice, samplerate

def getSlice(file, series, starttime, stoptime):
    auf = aud.File.open(file)
    basetime = auf[series][0]['time'].item().to_pydatetime()
    basetime = basetime.replace(tzinfo=pytz.UTC)
    starttime, stoptime = starttime.replace(tzinfo=pytz.UTC), stoptime.replace(tzinfo=pytz.UTC)

    sIdx = binary_search_timeseries(starttime, auf)
    eIdx = binary_search_timeseries(stoptime, auf)
    # s = auf[series][sIdx:eIdx]['time']
    # print(s, type(s[0]))
    # samplerate = getSF(auf[series][sIdx:eIdx]['time'])#hp.get_samplerate_mstimer(auf[series][sIdx:eIdx]['time'])
    samplerate = (eIdx-sIdx) / (stoptime-starttime).total_seconds()

    return auf[series][sIdx:eIdx]['value'], samplerate

def getSliceMultiseries(file, series, starttime, stoptime):
    auf = aud.File.open(file)
    basetime = auf[series[0]][0]['time'].item().to_pydatetime()
    basetime = basetime.replace(tzinfo=pytz.UTC)
    starttime, stoptime = starttime.replace(tzinfo=pytz.UTC), stoptime.replace(tzinfo=pytz.UTC)

    sIdx = binary_search_timeseries(starttime, auf)
    eIdx = binary_search_timeseries(stoptime, auf)
    # s = auf[series][sIdx:eIdx]['time']
    # print(s, type(s[0]))
    # samplerate = getSF(auf[series][sIdx:eIdx]['time'])#hp.get_samplerate_mstimer(auf[series][sIdx:eIdx]['time'])
    samplerate = (eIdx-sIdx) / (stoptime-starttime).total_seconds()

    return np.array([auf[s][sIdx:eIdx]['value'] for s in series]), samplerate

def getFINFromPath(path):
    if (isinstance(path, Path)):
        path = str(path)
    return path.split(os.sep)[-1].split('.')[0].split('_')[-1]

def findFileByFIN(finStr, searchDir):
    if isinstance(finStr, int):
        finStr = str(finStr)
    if isinstance(searchDir, str):
        searchDir = Path(searchDir)
    for f in searchDir.glob('*%s.h5' % finStr):
        return str(f)
    return None

def addSlicesToDictionary(file, seriesOfInterest, start, end, id, d, slice_size_s=10):
    # file = findFileByFIN(str(FIN_ID))
    h5file = aud.File.open(file, readonly=True)
    start = start.timestamp()
    end = end.timestamp()
    sliceStart = start; sliceEnd = sliceStart+slice_size_s
    while (sliceEnd <= end):
        # d['fin_id'].append(FIN_ID)
        d['start'].append( dt.datetime.fromtimestamp(sliceStart) )
        d['end'].append( dt.datetime.fromtimestamp(sliceEnd) )
        d['window_id'].append(id)
        sliceStart = sliceEnd; sliceEnd += slice_size_s
    return


def satisfiesDistanceFromAlerts(allAlertsInFile, prospectiveStart, prospectiveEnd):
    #ensure it doesn't intersect with any other alert, or even within 1 hour buffer
    for idx, alertEntry in allAlertsInFile.iterrows():
        start = alertEntry['newstartalert'] - dt.timedelta(hours=1)
        end = alertEntry['newendalert'] + dt.timedelta(hours=1)

        if (not (start > prospectiveEnd or prospectiveStart > end)):
            return False
    return True

def isValid(file, start, stop):
    sIdx = binary_search_timeseries(start, file)
    eIdx = binary_search_timeseries(stop, file)
    if (sIdx == eIdx): return False
    def get(i):
        b = file[HR_SERIES_II][i]['time'].item().to_pydatetime()
        b = b.replace(tzinfo=pytz.UTC)
        return b
    def diffLessThanOne(t1, t2):
        return (t1-t2) < dt.timedelta(seconds=1)

    res =  diffLessThanOne(get(sIdx), start) and diffLessThanOne(get(eIdx), stop)
    return res

def getRandomSlice(fin, duration_s, patientSeriesSearchDirectory, seriesOfInterest=HR_SERIES_II, tryLimit=50):
    f = findFileByFIN(fin, patientSeriesSearchDirectory)
    try:
        f = aud.File.open(f)

        fileStartTime = f[seriesOfInterest][0]['time'].item().to_pydatetime()
        fileEndTime = f[seriesOfInterest][-1]['time'].item().to_pydatetime()
    except:
        return None, None
    fileStartTime, fileEndTime = fileStartTime.replace(tzinfo=pytz.UTC), fileEndTime.replace(tzinfo=pytz.UTC)

    prospectiveStart = dt.datetime.fromtimestamp(random.randint(int(fileStartTime.timestamp()), int(fileEndTime.timestamp())))
    prospectiveEnd = prospectiveStart + dt.timedelta(seconds=10)
    prospectiveStart, prospectiveEnd = prospectiveStart.replace(tzinfo=pytz.UTC), prospectiveEnd.replace(tzinfo=pytz.UTC)
    isntValid = lambda x, y, z: not isValid(x,y,z)
    count = 0
    while (not isValid(f, prospectiveStart, prospectiveEnd) and (count < tryLimit)):
        prospectiveStart = dt.datetime.fromtimestamp(random.randint(int(fileStartTime.timestamp()), int(fileEndTime.timestamp())))
        prospectiveEnd = prospectiveStart + dt.timedelta(seconds=duration_s)
        prospectiveStart, prospectiveEnd = prospectiveStart.replace(tzinfo=pytz.UTC), prospectiveEnd.replace(tzinfo=pytz.UTC)
        count += 1
    if (count >= tryLimit):
        return None, None
    return prospectiveStart, prospectiveEnd