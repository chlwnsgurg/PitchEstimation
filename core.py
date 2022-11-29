import os
import numpy as np
from numpy.lib.stride_tricks import as_strided
import torch
import torch.nn.functional as F
from  model import Crepe
from utils import to_local_average_cents, cents_to_frequency, frequency_to_cents

class PitchEstimator:
    """
    General TorchCrepe Predictor.
    """
    def __init__(self):

        self.model = Crepe()
        self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'assets', 'pre-trained.pth')))  # or 'user-trained.pth'
        self.model.eval()

    def predict(self, audio, sr=16000, step_size=10):
        """
        Perform pitch estimation on given audio
        Parameters
        ----------
        audio : np.ndarray [shape=(N,) or (N, C)]
            The audio samples. Multichannel audio will be downmixed.
        sr : int
            Sample rate of the audio samples. The audio will be resampled if
            the sample rate is not 16 kHz, which is expected by the model.
        step_size : int
            The step size in milliseconds for running pitch estimation.
        Returns
        -------
        A 3-tuple consisting of:
            time: np.ndarray [shape=(T,)]
                The timestamps on which the pitch was estimated
            frequency: np.ndarray [shape=(T,)]
                The predicted pitch values in Hz
            confidence: np.ndarray [shape=(T,)]
                The confidence of voice activity, between 0 and 1
        """
        
        # make 1024-sample frames of the audio with hop length of 10 milliseconds
        x = np.pad(audio, 512, mode='constant', constant_values=0)
        hop_length = int(sr * step_size / 1000)     # step_size = int(1000 * 160 / 16000)
        n_frames = 1 + int((len(x) - 1024) / hop_length)
        frames = as_strided(x, shape=(1024, n_frames),
                            strides=(x.itemsize, hop_length * x.itemsize))
        frames = frames.transpose().copy()

        # normalize each frame -- this is expected by the model
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        frames /= np.std(frames, axis=1)[:, np.newaxis]

        frames = torch.tensor(frames)
        
        y = self.model(frames)

        confidence = y.detach().numpy().max(axis=1)

        cents = to_local_average_cents(y.detach().numpy())
        
        frequency = 10 * 2 ** (cents / 1200)
        frequency[np.isnan(frequency)] = 0

        time = np.arange(confidence.shape[0]) * step_size / 1000.0

        return time, frequency, confidence