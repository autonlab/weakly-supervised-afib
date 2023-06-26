from data.loaders import loadTestsetWaveforms

import neurokit2 as nk
import numpy as np
import scipy as sp
from typing import List

def getBaselinePWaves(baselineSignals, samplerates):
    commonSampleRate = min([sr for sr in samplerates])
    print(f'Common sample rate for these badboys: {commonSampleRate}')
    maxAfter, maxBefore = 0, 0
    pwaves = list()

    distances = list()
    for i, signal in enumerate(baselineSignals):
        samplerate = samplerates[i]
        if (samplerate > commonSampleRate):
            #resample them to same samplerate and average them
            ##how to: https://stackoverflow.com/questions/37120969/how-can-we-use-scipy-signal-resample-to-downsample-the-speech-signal-from-44100
            signalSeconds = len(signal) / samplerate
            samples = signalSeconds * commonSampleRate
            afterSignal, samplerate = sp.signal.resample(signal, samples), commonSampleRate
            print(len(afterSignal) ,commonSampleRate * signalSeconds)

        #Ensure peaks are aligned 
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=samplerate)
        _, waves_peak = nk.ecg_delineate(signal, rpeaks, sampling_rate=samplerate, method="peak")
        #collect array of p_wave signal slices
        for j in range(len(waves_peak['ECG_P_Onsets'])):
            startIdx, peakIdx, endIdx = waves_peak['ECG_P_Onsets'][j], waves_peak['ECG_P_Peaks'][j], waves_peak['ECG_P_Offsets'][j]
            if (np.isnan(startIdx)):
                continue
            distanceBefore, distanceAfter = peakIdx - startIdx, endIdx - peakIdx
            distances.append((distanceBefore, distanceAfter))
            maxBefore, maxAfter = max(distanceBefore, maxBefore), max(distanceAfter, maxAfter)
            pwaves.append(signal[startIdx:endIdx])
    pWavesAligned = list()
    for i, pwave in enumerate(pwaves):
        print(type(pwave), len(pwave))
        distanceBefore, distanceAfter = distances[i]
        beforeBuffer, afterBuffer = maxBefore - distanceBefore, maxAfter - distanceAfter
        signal = np.zeros(beforeBuffer) + signal + np.zeros(afterBuffer)
        print(len(signal))
        pWavesAligned.append(signal)
    return pWavesAligned


def getPTemplate(baselinePWaves) -> tuple:
    """_summary_

    Args:
        baselineSignalsAndSampleRates (_type_): _description_

    Returns:
        tuple: of the p wave template (signal, samplerate)
    """
    pWaves = list()
    commonSampleRate = min([sr for (_, sr) in baselinePWaves])
    for signal, samplerate in baselinePWaves:
        #collect all pwaves
        if (samplerate > commonSampleRate):
            #resample them to same samplerate and average them
            ##how to: https://stackoverflow.com/questions/37120969/how-can-we-use-scipy-signal-resample-to-downsample-the-speech-signal-from-44100
            signalSeconds = len(signal) / samplerate
            samples = signalSeconds * commonSampleRate
            signal, samplerate = sp.signal.resample(signal, samples), commonSampleRate

        pWaves.append(signal)
    template = np.mean(pWaves)
    return template, commonSampleRate

def pWavePresent(signal, samplerate, pwaves: List[tuple], p_template=None, template_samplerate=None):
    """_summary_

    Args:
        signal (_type_): _description_
        samplerate (_type_): _description_
        pwaves (List[tuple]): sequence of pwave (start, end) indices in `signal`
        p_template (_type_, optional): _description_. Defaults to None.
    """

    corrCoefs = list()
    for pStart, pStop in pwaves:
        corrCoef = np.corrcoef(signal[pStart:pStop], p_template)
        corrCoefs.append(corrCoef)
    return corrCoefs

if __name__=='__main__':
    #collect some sinus rhythm segments
    df, waveforms = loadTestsetWaveforms()
    df.reset_index(inplace=True)
    print(df)

    stable = df[df['label'] == 'SINUS']
    print(stable)
    signals, samplerates = list(), list()
    for i, row in stable.iterrows():
        signal, sr = waveforms[i]
        signals.append(signal)
        samplerates.append(sr)
    print(len(signals), len(samplerates))
    # 
    # getBaselinePWaves    