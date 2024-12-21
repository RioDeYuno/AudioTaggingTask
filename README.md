# Audio Tagging with Noisy Labels using InceptionV3 with CNN

This repository contains the implementation and results of an audio tagging task, inspired by the "Freesound Audio Tagging 2019" [1] challenge hosted on Kaggle. The project was conducted as part of the course **CSC 5351 01** at **Al Akhawayn University in Ifrane**. It explores audio classification using CNN to depict spectrograms in the presence of noisy and minimally supervised data.

## Problem Statement
The task aims to address the challenge of exploiting small, reliable, manually labeled data alongside larger, noisy web audio datasets in an audio tagging setting. Additionally, the project incorporates domain adaptation techniques to handle potential domain mismatches between different data sources.


## Kaggle Task Overview
![lala](https://github.com/user-attachments/assets/4fdcd8c3-3b49-4f4a-9c40-235db8e59795)

The task aims to classify audio clips into one or more of 80 everyday sound categories defined by the AudioSet Ontology. The dataset includes:
- A small manually-labeled curated dataset.[1]
- A larger noisy-labeled dataset derived from web audio sources. [2]

Our objective was to handle noisy labels effectively and leverage minimal supervision to train a robust audio classifier.

## Dataset

The dataset used is the **FSDKaggle2019** dataset [1], comprising:
- **Curated subset**: 4,970 clips, ~10.5 hours, high-quality varying in length from 0.3 to 30 seconds, labelled by hand.
- **Noisy subset**: 19,815 clips, ~80 hours, automatic labels with potential noise. [2]

To optimize model performance, the noisy dataset was employed for pretraining, effectively serving as a "warm-up" phase. This approach leverages the large volume of the noisy dataset to initialize the model, followed by fine-tuning with the curated dataset to enhance precision and reduce the impact of label noise. This dual-phase training strategy allows the model to generalize better across diverse acoustic conditions and label qualities.

For detailed dataset information, visit the [Freesound Audio Tagging 2019 Kaggle Page](https://www.kaggle.com/c/freesound-audio-tagging-2019).

## Data Preprocessing [1]
We employ Log-Mel spectrogram representations of audio data, generated using the `librosa` library. Key preprocessing steps include:
1. **Audio Loading and Trimming**: Each audio file is trimmed or padded to a fixed length of 2 seconds.
2. **Log-Mel Spectrogram Conversion**: Raw audio is converted into spectrograms with 128 Mel bands, using parameters such as FFT size, hop length, and frequency range tailored for optimal performance.
4. **Strength Adaptive Crop**: Regions with high dB values are preferentially cropped to ensure important information is preserved inspired from Uraratsu's work [7]

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
### CVSSP Baseline (inspired) [6]
The baseline model, based on the CVSSP architecture, includes log-mel spectrogram with 9 convolutional layers with 2x2 pooling policies. This structure was used as a starting point for further enhancement.

![image](https://github.com/user-attachments/assets/df28e04e-03ca-49f3-9b8c-76fc3a957658)


### CustomCNN (inspired from [2] & [3])
- Enhanced with **spatial** and **channel attention mechanisms**. inspired from [8]
- ConvBlock structure adapted from the Kaggle Salt Segmentation competition by @phalanx.
- A five-block architecture designed to capture primitive patterns in time and frequency axes.



### InceptionV3  (inspired from [9])
- Two versions: a standard three-channel version and a modified one-channel variant.
- Borrowed from torchvision with customized input layers for audio data.

## Training Pipeline [7] (inspired from Uraratsu's work)
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

- [1] Eduardo Fonseca et al., Audio tagging with noisy labels and minimal supervision. DCASE 2019 Proceedings.

- [2] Jort F. Gemmeke et al., Audio set: an ontology and human-labeled dataset for audio events. ICASSP 2017.

- [3] CNN 2D Basic Solution by @daisukelab

- [4] Simple 2D CNN Classifier by @mhiro2
 
- [5] DaisukeLab's dataset and preprocessing: [Kaggle Kernel](https://www.kaggle.com/daisukelab/fat2019_prep_mels1)
- [6] CVSSP Baseline: [Kaggle Discussion](https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/89382#latest-534349)
- [7] Uratatsu, "7th Place Solution for Freesound Audio Tagging 2019," Speaker Deck. Available: https://speakerdeck.com/uratatsu/7th-place-solution-for-freesound-audio-tagging-2019?slide=13. 
- [8] Attention Mechanisms: [Salt Segmentation](https://github.com/ybabakhin/kaggle_salt_bes_phalanx)
- [9] InceptionV3 Adaptation: [Kaggle Kernel](https://www.kaggle.com/sailorwei/fat2019-2d-cnn-with-mixup-lb-0-673)
- [10] Data Augmentation Techniques: [SoX Documentation](http://sox.sourceforge.net/)


