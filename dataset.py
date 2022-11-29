import os
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset

class NSynth(Dataset):
    def __init__(self, annotations_file, audio_dir):
        self.audio_labels = pd.read_json(annotations_file,orient='index')
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_labels.iloc[idx].name)+'.wav'
        audio, sample_rate = librosa.load(audio_path,sr=16000)
        pitch = self.audio_labels.iloc[idx, 1]
        return audio, pitch