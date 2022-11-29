import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import dataset as dset
from torch.utils.data import DataLoader
from core import PitchEstimator

nsynth_test=dset.NSynth('./nsynth-test/examples.json','./nsynth-test/audio')
test_dataloader = DataLoader(nsynth_test, batch_size=64, shuffle=True)

test_features, test_labels = next(iter(test_dataloader))
audio = test_features[0]
pitch = test_labels[0]
print(f"Pitch: {pitch}")

AI = PitchEstimator()
time, frequency, confidence = AI.predict(audio)


# can add some post-processing

# interpolation
length = audio.shape[0] // 100
if time.shape[-1] != length:
    frequency = np.interp(
        np.linspace(0, 1, length, endpoint=False),
        np.linspace(0, 1, time.shape[-1], endpoint=False),
        frequency,
    )
    confidence = np.interp(
        np.linspace(0, 1, length, endpoint=False),
        np.linspace(0, 1, time.shape[-1], endpoint=False),
        confidence,
    )

# divide frequency in to lcf and hcf
lcf = np.copy(frequency) # frequncy with low confidence value
hcf = np.copy(frequency) # frequncy with high confidence value
for i in range(length):
    if(confidence[i]<0.5): hcf[i]=np.nan
    else: lcf[i]=np.nan

plt.plot(frequency,'g')
plt.plot(lcf,'r')
plt.plot(hcf,'b')
plt.show()
pass