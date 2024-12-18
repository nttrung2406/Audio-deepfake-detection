import os
import torch
from torch.utils.data import Dataset
import torchaudio

root = os.getcwd()
train_path = os.path.join(root, 'data', 'audios', 'train')
test_path = os.path.join(root, 'data', 'audios', 'test')
TARGET_SAMPLE_RATE = 16000

class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.classes = ["real", "fake"]
        self.audio_files = []
        self.labels = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith("wav"):
                    file_path = os.path.join(class_dir, file)
                    try:
                        # Test loading the file to ensure it's valid
                        torchaudio.info(file_path)
                        self.audio_files.append(file_path)
                        self.labels.append(class_idx)
                    except RuntimeError:
                        print(f"Skipping invalid file: {file_path}")
        
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
        )

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]

        try:
            audio, sr = torchaudio.load(audio_file)
            if audio.shape[1] == 0:
                raise ValueError(f"File {audio_file} contains zero samples.")
        except Exception as e:
            raise ValueError(f"Error loading audio file {audio_file}: {e}")

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0).unsqueeze(0)

        if sr != TARGET_SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)(audio)

        fixed_length = TARGET_SAMPLE_RATE * 3
        if audio.shape[1] < fixed_length:
            audio = torch.nn.functional.pad(audio, (0, fixed_length - audio.shape[1]))
        else:
            audio = audio[:, :fixed_length]

        audio = self.mel_spectrogram(audio)

        return audio, label

class TestDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.audio_files = []

        for file in os.listdir(data_dir):
            if file.endswith(".wav"):
                self.audio_files.append(os.path.join(data_dir, file)) 

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
        )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]

        audio, sr = torchaudio.load(audio_file)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0).unsqueeze(0)

        if sr != TARGET_SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)(audio)

        fixed_length = TARGET_SAMPLE_RATE * 3
        if audio.shape[1] < fixed_length:
            audio = torch.nn.functional.pad(audio, (0, fixed_length - audio.shape[1]))
        else:
            audio = audio[:, :fixed_length]

        audio = self.mel_spectrogram(audio)

        return audio, audio_file