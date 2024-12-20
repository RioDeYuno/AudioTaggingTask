import torch
from config import *
from train import train_model
from evaluate import evaluate_model
from data_preprocessing import convert_wav_to_image
from models import get_model
from torch.utils.data import DataLoader
from data_preprocessing import FATTrainDataset, FATValidDataset
from tqdm import tqdm
import logging
import os

# Logging setup
def setup_main_logger():
    logger = logging.getLogger("MainLogger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


logger = setup_main_logger()

def main():
    """
    Main function to manage training, validation, and evaluation processes.
    """
    # Ensure paths exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    logger.info("Starting main pipeline...")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Preprocess and prepare datasets
    logger.info("Preparing datasets...")

    # Preprocessing noisy and curated datasets
    if DEBUG_MODE:
        logger.info("Debug mode is active. Using smaller datasets for faster processing.")

    x_train_curated, y_train_curated = convert_wav_to_image(config=CONFIG, df=train_curated_df, source_dir=DATA_PATH)
    x_val, y_val = convert_wav_to_image(config=CONFIG, df=val_df, source_dir=DATA_PATH)

    # Define datasets and data loaders
    train_dataset = FATTrainDataset(
        images=x_train_curated,
        labels=y_train_curated,
        transforms=transforms_dict["train"],
        strength_list=strength_list_train,
    )

    val_dataset = FATValidDataset(
        fnames=val_fnames,
        mels=x_val,
        labels=y_val,
        transforms=transforms_dict["valid"],
        strength_list=strength_list_val,
        tta=VALIDATION_TTA,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info("Datasets prepared. Starting training...")

    # Train the model
    train_model()

    # Evaluate the model
    logger.info("Training completed. Starting evaluation...")

    model = get_model("CustomCNN", num_classes=80).to(device)
    model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/best_model.pth"))

    average_lwlrap, per_class_lwlrap, weight_per_class = evaluate_model(
        model, val_loader, device, tta=VALIDATION_TTA
    )

    logger.info(f"Evaluation Results: Average LWLRAP: {average_lwlrap:.4f}")
    logger.info(f"Per Class LWLRAP: {per_class_lwlrap}")
    logger.info(f"Class Weights: {weight_per_class}")

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
