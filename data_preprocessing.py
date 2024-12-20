#data_preprocessing.py

import librosa
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import subprocess as sp
import random
import torch
from torchvision.transforms import ToTensor
from tqdm import tqdm


# Audio Reading and Preprocessing
def read_audio(config, pathname, trim_long_data=True):
    """
    Reads and trims or pads audio data to a fixed length.
    """
    y, sr = librosa.load(pathname, sr=config['sampling_rate'])
    if len(y) > 0:
        y, _ = librosa.effects.trim(y)

    if len(y) > config['samples']:
        if trim_long_data:
            y = y[:config['samples']]
    else:
        padding = config['samples'] - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, config['samples'] - len(y) - offset), config['pad_mode'])

    return y


def audio_to_melspectrogram(config, audio):
    """
    Converts audio waveform to a Mel spectrogram.
    """
    spectrogram = librosa.feature.melspectrogram(
        audio,
        sr=config['sampling_rate'],
        n_mels=config['n_mels'],
        hop_length=config['hop_length'],
        n_fft=config['n_fft'],
        fmin=config['fmin'],
        fmax=config['fmax']
    )
    return librosa.power_to_db(spectrogram).astype(np.float32)


def read_as_melspectrogram(config, pathname, trim_long_data=True, debug_display=False):
    """
    Reads audio and converts it to a Mel spectrogram with optional debug visualization.
    """
    x = read_audio(config, pathname, trim_long_data)
    mels = audio_to_melspectrogram(config, x)

    if debug_display:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mels, sr=config['sampling_rate'], x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.show()

    return mels


# Data Augmentation
def augment_audio_with_sox(input_path, output_path, augment_cmd):
    """
    Applies audio augmentations using SoX.
    """
    cmd = ["sox", input_path, output_path] + augment_cmd
    sp.run(cmd)


def apply_augmentation(config, input_path, augment_type="pitch", params=None):
    """
    Applies predefined augmentations like pitch shift, fade, reverb, etc.
    """
    output_path = "augmented_output.wav"
    if augment_type == "pitch":
        augment_cmd = ["pitch", str(params.get("cent", 500))]
    elif augment_type == "fade":
        augment_cmd = ["fade", params.get("type", "q"), str(params.get("duration", 3))]
    elif augment_type == "reverb":
        augment_cmd = ["reverb"]
    elif augment_type == "gain":
        augment_cmd = ["gain", "-h", "treble", str(params.get("treble", "+20")), "bass", str(params.get("bass", "+20"))]
    elif augment_type == "equalizer":
        augment_cmd = ["equalizer", str(params.get("freq", 2400)), "3q", str(params.get("gain", 8))]
    else:
        raise ValueError("Invalid augmentation type!")

    augment_audio_with_sox(input_path, output_path, augment_cmd)
    return output_path


# Image Processing
def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    """
    Converts a mono spectrogram to a color image format.
    """
    mean = mean or X.mean()
    std = std or X.std()
    X_std = (X - mean) / (std + eps)

    _min, _max = X_std.min(), X_std.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min

    if (_max - _min) > eps:
        V = 255 * (X_std - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X_std, dtype=np.uint8)

    return np.stack([V, V, V], axis=-1)


def convert_wav_to_image(config, df, source_dir):
    """
    Converts WAV files to Mel spectrogram images.
    """
    X = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        mels = read_as_melspectrogram(config, Path(source_dir) / row['fname'], trim_long_data=False)
        color_image = mono_to_color(mels)
        X.append(color_image)
    return X


# Advanced Cropping Techniques
def strength_adaptive_crop(mels, crop_size):
    """
    Applies strength-based adaptive cropping to prioritize loud regions.
    """
    db_sums = np.sum(mels, axis=0)
    top_regions = np.argsort(db_sums)[-crop_size:]
    cropped = mels[:, min(top_regions):max(top_regions)]
    return cropped


# Dataset Preparation
class FATDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transforms=None):
        """
        Custom dataset for FAT2019.
        """
        self.images = images
        self.labels = labels
        self.transforms = transforms or ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]

        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(label, dtype=torch.float32)
