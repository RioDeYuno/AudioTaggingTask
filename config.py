# config.py

# Paths
DATA_PATH = "../input/freesound-audio-tagging-2019/"
AUGMENTED_DATA_PATH = "../input/fat2019_prep_mels1/"
MODEL_SAVE_PATH = "./results/models/"
RESULTS_PATH = "./results/"
AUGMENTED_FILES_PATH = "./augmented_files/"

# Audio processing configurations
SAMPLING_RATE = 44100  # Audio sampling rate
DURATION = 2  # Duration of audio in seconds
HOP_LENGTH = 347 * DURATION  # Hop length for spectrogram
N_MELS = 128  # Number of Mel bands
N_FFT = N_MELS * 20  # FFT window size
FMIN = 20  # Minimum frequency for Mel scale
FMAX = SAMPLING_RATE // 2  # Maximum frequency for Mel scale
PAD_MODE = "constant"  # Padding mode
SAMPLES = SAMPLING_RATE * DURATION  # Total number of samples in an audio clip

# Data augmentation configurations
AUGMENTATION_TYPES = [
    "pitch_shift",
    "fade",
    "reverb",
    "treble_bass",
    "equalizing",
]
PITCH_SHIFT_RANGE = [-500, 500]  # Cent range for pitch shift
FADE_TYPE = "q"  # Fade type for audio transformation
TREBLE_GAIN = [20, -30]  # Treble gain adjustment
BASS_GAIN = [20, -30]  # Bass gain adjustment
EQUALIZER_FREQ = 2400  # Frequency for equalizer adjustment
EQUALIZER_GAIN = [3, 8]  # Gain adjustment for equalizer

# Training configurations
BATCH_SIZE = 32  # Batch size for training
EPOCHS = 50  # Number of training epochs
LEARNING_RATE = 1e-3  # Learning rate
LOSS_FUNCTION = "BCEWithLogitsLoss"  # Loss function for optimization
OPTIMIZER = "Adam"  # Optimizer
LR_SCHEDULER = {
    "type": "CosineAnnealingLR",
    "T_max": 10,
    "eta_min": 1e-5,
}  # Learning rate scheduler configuration
VALIDATION_TTA = 5  # Test Time Augmentation iterations during validation

# Crop policies
CROP_POLICY = "strength_adaptive"  # Strength adaptive crop policy
IMAGE_SIZE = (128, 128)  # Target image size for spectrograms

# Model configurations
MODELS = {
    "CustomCNN": {
        "num_blocks": 5,
        "kernel_size": (2, 2),
        "pooling_policy": ["both", "max", "avg"],
        "num_classes": 80,
    },
    "InceptionV3": {
        "use_pretrained": True,
        "input_channels": 1,
    },
}

# Ensemble configurations
ENSEMBLE_METHOD = "equal_weight"  # Method for ensemble blending
NUM_MODELS = 3  # Number of models in the ensemble
TTA_COUNT = 20  # Number of TTA iterations per model during ensemble

# Logging and results
LOG_FILE = "./results/training_log.txt"
SAVE_METRICS = ["lwlrap", "accuracy"]  # Metrics to save in results

# Miscellaneous
DEBUG_MODE = False  # Enable debugging
RANDOM_SEED = 42  # Random seed for reproducibility
