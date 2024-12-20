# Audio Tagging with Noisy Labels using InceptionV3 with CNN

This repository contains the implementation and results of an audio tagging task, inspired by the "Freesound Audio Tagging 2019" challenge hosted on Kaggle. The project was conducted as part of the course **CSC 5351 01** at **Al Akhawayn University in Ifrane**. It explores audio classification using CNN to depict spectrograms in the presence of noisy and minimally supervised data.

## Problem Statement
The task aims to address the challenge of exploiting small, reliable, manually labeled data alongside larger, noisy web audio datasets in an audio tagging setting. Additionally, the project incorporates domain adaptation techniques to handle potential domain mismatches between different data sources.


## Kaggle Task Overview
![lala](https://github.com/user-attachments/assets/4fdcd8c3-3b49-4f4a-9c40-235db8e59795)

The task aims to classify audio clips into one or more of 80 everyday sound categories defined by the AudioSet Ontology. The dataset includes:
- A small manually-labeled curated dataset.
- A larger noisy-labeled dataset derived from web audio sources.

Our objective was to handle noisy labels effectively and leverage minimal supervision to train a robust audio classifier.

## Dataset

The dataset used is the **FSDKaggle2019** dataset, comprising:
- **Curated subset**: 4,970 clips, ~10.5 hours, high-quality varying in length from 0.3 to 30 seconds, labelled by hand.
- **Noisy subset**: 19,815 clips, ~80 hours, automatic labels with potential noise.

For detailed dataset information, visit the [Freesound Audio Tagging 2019 Kaggle Page](https://www.kaggle.com/c/freesound-audio-tagging-2019).

## Data Preprocessing
We employ Log-Mel spectrogram representations of audio data, generated using the `librosa` library. Key preprocessing steps include:
1. **Audio Loading and Trimming**: Each audio file is trimmed or padded to a fixed length of 2 seconds.
2. **Log-Mel Spectrogram Conversion**: Raw audio is converted into spectrograms with 128 Mel bands, using parameters such as FFT size, hop length, and frequency range tailored for optimal performance.
4. **Strength Adaptive Crop**: Regions with high dB values are preferentially cropped to ensure important information is preserved.

![audio_spectrogram_batch](https://github.com/user-attachments/assets/58ac10db-69cd-4781-aa0a-d2e1cfe242eb)

## Data Augmentation using SOX
We implemented SOX augmentation techniques:
- **Pitch Shift**: Alters the pitch while preserving the class.
  
- **Fade**: Adds fade-in and fade-out effects.
- **Reverb**: Simulates natural reverberation.
- **Treble and Bass Adjustment**: Adjusts the frequency emphasis.
- **Equalization**: Focuses on specific frequency bands.

These augmentations were applied to both noisy and curated datasets, resulting in datasets 8x larger than the original curated dataset.

## Model Architecture
### CVSSP Baseline
The baseline model, based on the CVSSP architecture, includes convolutional layers with pooling policies. This structure was used as a starting point for further enhancement.

### CustomCNN
- Enhanced with **spatial** and **channel attention mechanisms**.
- ConvBlock structure adapted from the Kaggle Salt Segmentation competition by @phalanx.
- A five-block architecture designed to capture primitive patterns in time and frequency axes.

### InceptionV3
- Two versions: a standard three-channel version and a modified one-channel variant.
- Borrowed from torchvision with customized input layers for audio data.

## Training Pipeline

![image](https://github.com/user-attachments/assets/0dfed930-e9e1-40a3-bffa-f59aad2abff6)


The training process consists of three stages:

1. **Stage 1**: Pre-training on noisy datasets using Log-Mel spectrograms with pitch augmentation.
2. **Stage 2**: Fine-tuning on curated datasets with RandomResizedCrop and additional augmentations.
3. **Stage 3**: Final fine-tuning on curated datasets without RandomResizedCrop.



During validation, 20 test-time augmentations (TTA) are applied for each model to stabilize predictions.

## Results performance

The primary evaluation metric for the project is **LWLRAP**. The best model achieved the following:
- **Training Loss**: 0.217
- **Validation Loss**: 0.242
- **Average LWLRAP**: 0.8824

| Model           | Dataset       | CV LWLRAP | LB LWLRAP |
|-----------------|---------------|-----------|-----------|
| CustomCNN       | Noisy + Curated | 0.88      | 0.87      |
| InceptionV3 3ch | Noisy + Curated | 0.87      | 0.86      |
| InceptionV3 1ch | Noisy + Curated | 0.86      | 0.85      |

The ensemble of these models with equal weights achieved further improvements in both CV and LB scores.

## Implementation

The project is organized into the following scripts:

1. **`config.py`**: Contains configurations such as paths, audio processing parameters, and training hyperparameters.
2. **`data_preprocessing.py`**: Includes functions for loading, preprocessing, and augmenting audio data into Mel spectrograms.
3. **`models.py`**: Defines the custom convolutional neural network (CNN) with Attention mechanism and integrates an InceptionV3-based architecture.
4. **`train.py`**: Implements the training and validation loops, including LWLRAP (Label-Weighted Label Ranking Average Precision) computation.
5. **`evaluate.py`**: Performs model evaluation using validation data and test-time augmentation (TTA).
6. **`main.py`**: Provides an entry point to train and evaluate the model sequentially.

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

4. Preprocess the data:
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

- CNN 2D Basic Solution by @daisukelab

- Simple 2D CNN Classifier by @mhiro2
 
- DaisukeLab's dataset and preprocessing: [Kaggle Kernel](https://www.kaggle.com/daisukelab/fat2019_prep_mels1)
- CVSSP Baseline: [Kaggle Discussion](https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/89382#latest-534349)
- Attention Mechanisms: [Salt Segmentation](https://github.com/ybabakhin/kaggle_salt_bes_phalanx)
- InceptionV3 Adaptation: [Kaggle Kernel](https://www.kaggle.com/sailorwei/fat2019-2d-cnn-with-mixup-lb-0-673)
- Data Augmentation Techniques: [SoX Documentation](http://sox.sourceforge.net/)

