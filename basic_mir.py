from pathlib import Path
import numpy, scipy, matplotlib.pyplot as plt, sklearn, urllib, IPython.display as ipd
import librosa, librosa.display
import stanford_mir; stanford_mir.init()
kick_signals = [
    librosa.load(p)[0] for p in Path().glob('audio/drum_samples/train/kick_*.mp3')
]
snare_signals = [
    librosa.load(p)[0] for p in Path().glob('audio/drum_samples/train/snare_*.mp3')
]


def extract_features(signal):
    return [
        librosa.feature.zero_crossing_rate(signal)[0, 0],
        librosa.feature.spectral_centroid(signal)[0, 0],
    ]

kick_features = numpy.array([extract_features(x) for x in kick_signals])
snare_features = numpy.array([extract_features(x) for x in snare_signals])


feature_table = numpy.vstack((kick_features, snare_features))

scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
training_features = scaler.fit_transform(feature_table)

plt.scatter(training_features[:10,0], training_features[:10,1], c='b')
plt.scatter(training_features[10:,0], training_features[10:,1], c='r')
plt.xlabel('Zero Crossing Rate')
plt.ylabel('Spectral Centroid')

plt.show()