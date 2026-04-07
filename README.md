# End-to-End Audio Recognition Pipeline (Using Deep Learning AI) 

This repository implements a **scalable audio tagging pipeline** designed to handle noisy and weakly-labeled data, inspired by the "Freesound Audio Tagging 2019" [1] challenge hosted on Kaggle **(co-hosted by Google Research)**. The project was conducted and supervised on the course **CSC 5351 01** at **Al Akhawayn University in ifrane** with an emphasis on real-world engineering constraints.

## Project Highlights
- Built an **end-to-end audio ML pipeline** handling curated and noisy datasets
- Implemented **large-scale preprocessing and augmentation** for audio data
- Trained deep CNN architectures including **CustomCNN with attention** and **InceptionV3**
- Applied **multi-stage training (pretraining + fine-tuning)** to mitigate noisy labels
- Achieved **0.88+ LWLRAP** using cross-validation, test-time augmentation, and ensembling

## Tech Stack
- Language: Python
- Audio Processing: Librosa, SoX
- Deep Learning: PyTorch, Torchvision
- Data Handling: NumPy, Pandas

## System Architecture
| Stage | Description |
|------|------------|
| Input | Raw audio files (WAV / MP3) |
| Preprocessing | Log-Mel spectrograms, adaptive cropping, audio augmentation |
| Training | (CNN + Attention) and InceptionV3 |
| Evaluation | LWLRAP + test-time augmentation |
| Ensembling | Combining model outputs for final predictions |


## Kaggle Task Overview
![lala](https://github.com/user-attachments/assets/4fdcd8c3-3b49-4f4a-9c40-235db8e59795)

The task aims to classify audio clips into one or more of 80 everyday sound categories defined by the AudioSet Ontology. 

Our objective was to handle noisy labels effectively and leverage minimal supervision to train a robust audio classifier.

## Google Research Challenges
- Exploiting small, reliable, manually labeled data alongside larger, noisy web audio data.
- Handling domain adaptation between different data sources.

## Dataset

The dataset used is the **FSDKaggle2019** dataset [1], comprising:
- **Curated subset**: 4,970 clips, ~10.5 hours, high-quality varying in length from 0.3 to 30 seconds, labelled by hand.
- **Noisy subset**: 19,815 clips, ~80 hours, automatic labels with potential noise. [2]

The noisy dataset is used as a **warm-up training phase**, followed by fine-tuning on curated data.


## Data Preprocessing [1]
We employ Log-Mel spectrogram representations of audio data, generated using the `librosa` library. Key preprocessing steps include:
1. **Audio Loading and Trimming**: Each audio file is trimmed or padded to a fixed length of 2 seconds.
2. **Log-Mel Spectrogram Conversion (128 Mel bands)**: parameters; FFT size, hop length, and frequency range
4. **Strength Adaptive Crop**: Regions with high dB values (info) are cropped [7]

![audio_spectrogram_batch](https://github.com/user-attachments/assets/58ac10db-69cd-4781-aa0a-d2e1cfe242eb)

## Data Augmentation using SOX 
- **Pitch Shift**
- **Fade**
- **Reverb**
- **Treble and Bass Adjustment**
- **Equalization**

These augmentations were applied to both noisy and curated datasets, resulting in datasets 8x larger -> Improves robustness

## Model Architecture
### CVSSP Baseline (inspired) [6]
The baseline model, based on the CVSSP architecture, used as a starting point:

![image](https://github.com/user-attachments/assets/df28e04e-03ca-49f3-9b8c-76fc3a957658)


- **CustomCNN**
  - Multi-block CNN with spatial and channel attention
  - Optimized for time–frequency representations
- **InceptionV3**
  - Adapted for audio spectrogram inputs
  - Both 1-channel and 3-channel variants
- **Ensemble**
  - Equal-weight ensembling for improved stability and performancea.

## Training Pipeline [7]
![image](https://github.com/user-attachments/assets/0dfed930-e9e1-40a3-bffa-f59aad2abff6)


The training process consists of three stages:

1. **Stage 1**: Pre-training on noisy data with augmentation
2. **Stage 2**: Fine-tuning on curated datasets with spatial augmentation
3. **Stage 3**: Final fine-tuning without aggressive cropping
4. **Evaluation** using 20× Test-Time Augmentation (TTA)


## Results performance

The best model achieved the following:
- **Training Loss**: 0.217
- **Validation Loss**: 0.242
- **Average LWLRAP**: 0.8824

| Model           | Dataset       | CV LWLRAP | LB LWLRAP |
|-----------------|---------------|-----------|-----------|
| CustomCNN       | Noisy + Curated | 0.88      | 0.87      |
| InceptionV3 3ch | Noisy + Curated | 0.87      | 0.86      |
| InceptionV3 1ch | Noisy + Curated | 0.86      | 0.85      |

The ensemble of these models with equal weights achieved further improvements in both CV and LB scores.

## Future Improvements
- Distributed training using multi-GPU or Spark integration
- Model serving via REST API
- CI/CD for automated training and evaluation
- Cloud storage integration (S3 / GCS)

## Implementation

The project is organized into the following scripts:

1. **`config.py`**: Contains configurations such as paths, audio processing parameters, and training hyperparameters.
2. **`data_preprocessing.py`**: Includes functions for loading, preprocessing, and augmenting audio data into Mel spectrograms.
3. **`models.py`**: Defines the custom convolutional neural network (CNN) with Attention mechanism and integrates an InceptionV3-based architecture.
4. **`train.py`**: Implements the training and validation loops, including LWLRAP (Label-Weighted Label Ranking Average Precision) computation.
5. **`evaluate.py`**: Performs model evaluation using validation data and test-time augmentation (TTA).
6. **`main.py`**: Provides an entry point to train and evaluate the model sequentially.


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


