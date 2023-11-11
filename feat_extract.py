# feat_extract.py
# Created 2023 by Jaycee Kaufman, Klick Inc. 
# Contact: jmorgankaufman@klick.com

'''
Versions for Packages
praat-parselmouth==0.4.3
'''

import parselmouth
from parselmouth.praat import call

def measure_feats(voiceID, f0min, f0max, unit):
    '''
    Function to calculate the acoustic features of a voice file
    
    Inputs:
    voiceID is the path to the voice file (.wav)

    f0min and f0max are minimum and maximum F0 values used to calculate the pitch. 
    Defaults in Praat are:
        f0min: 75 Hz
        f0max: 500 Hz

    unit is the units of the pitch, typically 'Hertz'

    Outputs:
    14 voice features corresponding to pitch, intensity, harmonic noise ratio (HNR), jitter, and shimmer
    '''
    try:
        sound = parselmouth.Sound(voiceID) # read the sound

        pitch = call(sound, "To Pitch", 0.0, f0min, f0max) # create a praat pitch object

        # Fundamental Frequency 
        meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
        stdevF0 = call(pitch, "Get standard deviation", 0, 0, unit) # get standard deviation of pitch

        # Intensity
        intensity = call(sound, "To Intensity", 75, 0, "yes")
        meanI = call(intensity, "Get mean", 0, 0, "energy") # get mean intensity
        stdevI = call(intensity, "Get standard deviation", 0 ,0, "energy") # get standard deviation of intensity

        # Harmonics-to-Noise Ratio
        hnr = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1)

        # Jitter measurements
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) 
        localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

        # Shimmer measurements
        localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        return meanF0, stdevF0, meanI, stdevI, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer
    except parselmouth.PraatError: 
        print(voiceID)
        return 0,0,0,0,0,0,0,0,0,0,0,0,0,0

# Define feature names
feature_names = [
    "meanF0", "stdevF0", "meanIntensity", "stdevIntensity", "HNR", 
    "localJitter", "localabsoluteJitter", "rapJitter", "ppq5Jitter", 
    "localShimmer", "localdbShimmer", "apq3Shimmer", "aqpq5Shimmer", "apq11Shimmer"
]
