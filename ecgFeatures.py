"""
This file extracts features from a dataset and saves them as a new feature set.              

Author: Josh Atwal                        
"""
from biosppy.signals import ecg
from scipy.io import loadmat, savemat
import numpy as np
import pywt as pw
import scipy.stats as st

dataPath = '../data/'
dset = ['training', 'testing']

sr = 300.0 #sampling rate

#For each of the training and testing sets
for s in dset:
    
    # Load dataset
    dataset = loadmat(dataPath + s + 'Set.mat')
    if s=='training':
        X = dataset['train'][0]
    else:
        X = dataset['test'][0]
    
    # Number of signals
    n = len(X)

    # Preallocate vectors
    R_avg, R_std, HR_avg, HR_std = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    ST,ST_std, TT_std, TT_avg, PR, PR_std, QT, QT_std, PP, PP_std = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    aMean, aVar, aSkew, aKurtosis, aSum, aRMS, aMAD, aMin, aMax = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    dMean, dVar, dSkew, dKurtosis, dSum, dRMS, dMAD, dMin, dMax = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    PPeak, PPeak_std, RPeak, RPeak_std, TPeak, TPeak_std  = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    # Extract features for every signal
    for i in range(n):
        
        # Extract basic ECG information
        ecgInfo =  ecg.ecg(signal=X[i][0], sampling_rate=sr,show=False)
        R_peaks = ecgInfo['rpeaks']
        R_distances = R_peaks[1:] - R_peaks[:-1]
        R_avg[i], R_std[i] = np.mean(R_distances), np.std(R_distances)
        heartRates = ecgInfo['heart_rate']
        HR_avg[i], HR_std[i] = np.mean(heartRates), np.std(heartRates)

        # Wavelet transform
        
        nWaves = ecgInfo['templates'].shape[0]
        
        detailCoeffs = np.zeros([len(ecgInfo['templates_ts']), nWaves])
        approxCoeffs = np.zeros([len(ecgInfo['templates_ts']), nWaves])
        for j in range(nWaves):
            approxCoeffs[:,j] = pw.swt(ecgInfo['templates'][j], wavelet = "db4", level=2, start_level=0, axis=-1)[1][0]
            detailCoeffs[:,j] = pw.swt(ecgInfo['templates'][j], wavelet = "db4", level=2, start_level=0, axis=-1)[1][1]
    
        # Statistics of detail wavelet coefficients
        dMean[i] = np.mean(detailCoeffs)
        dVar[i] = np.var(detailCoeffs)
        dSkew[i] = st.skew(np.ravel(detailCoeffs))
        dKurtosis[i] = st.kurtosis(np.ravel(detailCoeffs))
        dSum[i] = np.sum(detailCoeffs)
        dRMS[i] = np.sqrt(np.mean(detailCoeffs**2))
        dMAD[i] = np.mean(np.absolute(detailCoeffs - np.mean(detailCoeffs)))
        dMin[i] = np.min(detailCoeffs)
        dMax[i] = np.max(detailCoeffs)
        
        # Statistics of approximate wavelet coefficients
        aMean[i] = np.mean(approxCoeffs)
        aVar[i] = np.var(approxCoeffs)
        aSkew[i] = st.skew(np.ravel(approxCoeffs))
        aKurtosis[i] = st.kurtosis(np.ravel(approxCoeffs))
        aSum[i] = np.sum(approxCoeffs)
        aRMS[i] = np.sqrt(np.mean(approxCoeffs**2))
        aMAD[i] = np.mean(np.absolute(approxCoeffs - np.mean(approxCoeffs)))
        aMin[i] = np.min(approxCoeffs)
        aMax[i] = np.max(approxCoeffs)

        # Statistics of the ECG peaks
        ecgDelin = loadmat(dataPath + 'delineationResults/A%.5d_ECG_delineation.mat' % (i+1))['wavedet']['ECG'][0][0][0]
        ST[i] = np.nanmean(ecgDelin['T'][0][0]-ecgDelin['S'][0][0])
        ST_std[i] = np.nanstd(ecgDelin['T'][0][0]-ecgDelin['S'][0][0])
        TT_avg[i] = np.nanmean(ecgDelin['T'][0][0][1:] - ecgDelin['T'][0][0][:-1])
        TT_std[i] = np.nanstd(ecgDelin['T'][0][0][1:] - ecgDelin['T'][0][0][:-1])
        PR[i] = np.nanmean(ecgDelin['R'][0][0]-ecgDelin['P'][0][0])
        PR_std[i] = np.nanstd(ecgDelin['R'][0][0]-ecgDelin['P'][0][0])
        QT[i] = np.nanmean(ecgDelin['R'][0][0]-ecgDelin['P'][0][0])
        QT_std[i] = np.nanstd(ecgDelin['R'][0][0]-ecgDelin['P'][0][0])
        PP[i] = np.nanmean(ecgDelin['P'][0][0][1:] - ecgDelin['P'][0][0][:-1])
        PP_std[i] = np.nanstd(ecgDelin['P'][0][0][1:] - ecgDelin['P'][0][0][:-1])

        PPeak[i] = np.nanmean(ecgDelin['P'][0][0])
        PPeak_std[i] = np.nanstd(ecgDelin['P'][0][0])
        RPeak[i] = np.nanmean(ecgDelin['R'][0][0])
        RPeak_std[i] = np.nanstd(ecgDelin['R'][0][0])
        TPeak[i] = np.nanmean(ecgDelin['T'][0][0])
        TPeak_std[i] = np.nanstd(ecgDelin['T'][0][0])          

    # Combine all feature vectors into a matrix    
    F = np.vstack((R_avg, R_std, HR_avg, HR_std, aMean, aVar, aSkew, aKurtosis, aSum, aRMS, aMAD, aMin, aMax, dMean, dVar, dSkew, dKurtosis, dSum, dRMS, dMAD, dMin, dMax, ST,ST_std, TT_avg, TT_std, PR, PR_std, QT, QT_std, PP, PP_std, PPeak,PPeak_std, RPeak, RPeak_std, TPeak, TPeak_std))

    # Save as output matrix
    savemat(dataPath+s+"Features.mat", {"F":F})






