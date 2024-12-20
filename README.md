# Audio Tagging with Noisy Labels and Minimal Supervision using InceptionV3 with CNN

This repository contains the implementation and results of an audio tagging task, inspired by the "Freesound Audio Tagging 2019" challenge hosted on Kaggle. The project was conducted as part of the course **CSC 5351 01** at **Al Akhawayn University in Ifrane**. It explores audio classification using audio tagging in the presence of noisy and minimally supervised data.

## Project Overview

The task aims to classify audio clips into one or more of 80 everyday sound categories defined by the AudioSet Ontology. The dataset includes:
- A small manually-labeled curated dataset.
- A larger noisy-labeled dataset derived from web audio sources.

Our objective was to handle noisy labels effectively and leverage minimal supervision to train a robust audio classifier.

## Dataset

The dataset used is the **FSDKaggle2019** dataset, comprising:
- **Curated subset**: 4,970 clips, ~10.5 hours, high-quality manual labels.
- **Noisy subset**: 19,815 clips, ~80 hours, automatic labels with potential noise.

For detailed dataset information, visit the [Freesound Audio Tagging 2019 Kaggle Page](https://www.kaggle.com/c/freesound-audio-tagging-2019).

![audio_spectrogram_batch](https://github.com/user-attachments/assets/58ac10db-69cd-4781-aa0a-d2e1cfe242eb)


## Implementation

The project is organized into the following scripts:

1. **`config.py`**: Contains configurations such as paths, audio processing parameters, and training hyperparameters.
2. **`data_preprocessing.py`**: Includes functions for loading, preprocessing, and augmenting audio data into Mel spectrograms.
3. **`models.py`**: Defines the custom convolutional neural network (CNN) and integrates an InceptionV3-based architecture.
4. **`train.py`**: Implements the training and validation loops, including LWLRAP (Label-Weighted Label Ranking Average Precision) computation.
5. **`evaluate.py`**: Performs model evaluation using validation data and test-time augmentation (TTA).
6. **`main.py`**: Provides an entry point to train and evaluate the model sequentially.

## Results

The primary evaluation metric for the project is **LWLRAP**. The best model achieved the following:
- **Training Loss**: 0.217
- **Validation Loss**: 0.242
- **Average LWLRAP**: 0.8824

Per-class LWLRAP metrics and weights are stored for further analysis.

## How to Reproduce Evaluation 

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/audio-tagging-noisy-labels.git
   cd audio-tagging-noisy-labels

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Place the FSDKaggle2019 dataset in the appropriate directories as specified in
   ```bash
   config.py:

* Curated data: Store in the directory defined by DATA_PATH.
* Noisy data: Store in the same directory.

4. Preprocess the data (if applicable):
   ```bash
   python data_preprocessing.py


5. Train the model:
   ```bash
   python main.py

6. Evaluate the model:
   ```bash
   python evaluate.py
   
* Computes LWLRAP scores for the validation dataset.
* Generates per-class metrics and logs results.


## References

This work draws upon methodologies and publicly available resources, including:

- Eduardo Fonseca et al., Audio tagging with noisy labels and minimal supervision. DCASE 2019 Proceedings.

- Jort F. Gemmeke et al., Audio set: an ontology and human-labeled dataset for audio events. ICASSP 2017.

- Kaggle Kernels and Notebooks:

  - CNN 2D Basic Solution by @daisukelab

  - Simple 2D CNN Classifier by @mhiro2

  - Inception v3 by @sailorwei
