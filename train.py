#train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from data_preprocessing import FATTrainDataset, FATValidDataset, convert_wav_to_image
from models import get_model
from config import *
import numpy as np
import logging


# Logging setup
def setup_logger():
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


logger = setup_logger()


# Train Loop
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for data, labels in tqdm(train_loader, desc="Training"):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss


# Validation Loop
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc="Validation"):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            all_preds.extend(outputs.sigmoid().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader)
    per_class_lwlrap, weight_per_class = calculate_lwlrap(
        np.array(all_labels), np.array(all_preds)
    )
    lwlrap_score = np.sum(per_class_lwlrap * weight_per_class)
    return epoch_loss, lwlrap_score


# Learning Rate Scheduler
def get_scheduler(optimizer, config):
    if config["type"] == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["T_max"], eta_min=config["eta_min"]
        )
    else:
        raise ValueError("Unsupported scheduler type")


# Label Weighted Label Ranking Average Precision
import numpy as np


def _one_sample_positive_class_precisions(scores, truth):
    """
    Calculate precisions for each true class for a single sample.

    Args:
        scores: np.array of (num_classes,) giving the individual classifier scores.
        truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
        pos_class_indices: np.array of indices of the true classes for this sample.
        pos_class_precisions: np.array of precisions corresponding to each of those classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)

    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)

    retrieved_classes = np.argsort(scores)[::-1]
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)

    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True

    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    precision_at_hits = (
        retrieved_cumulative_hits[class_rankings[pos_class_indices]]
        / (1 + class_rankings[pos_class_indices].astype(np.float))
    )
    return pos_class_indices, precision_at_hits


def calculate_lwlrap(truth, scores):
    """
    Calculate label-weighted label-ranking average precision.

    Args:
        truth: np.array of (num_samples, num_classes) giving boolean ground-truth
               of presence of that class in that sample.
        scores: np.array of (num_samples, num_classes) giving the classifier-under-
                test's real-valued score for each class for each sample.

    Returns:
        per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each class.
        weight_per_class: np.array of (num_classes,) giving the prior of each class
                          within the truth labels.
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape

    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))

    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(
            scores[sample_num, :], truth[sample_num, :]
        )
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = precision_at_hits

    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))

    per_class_lwlrap = (
        np.sum(precisions_for_samples_by_classes, axis=0) / np.maximum(1, labels_per_class)
    )

    return per_class_lwlrap, weight_per_class


# Main Training Function
def train_model():
    # Load Configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dataset and DataLoader Preparation
    train_dataset = FATTrainDataset(
        mels=x_train, labels=y_train, transforms=transforms_dict["train"]
    )
    val_dataset = FATValidDataset(
        fnames=val_fnames,
        mels=x_val,
        labels=y_val,
        transforms=transforms_dict["valid"],
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model Initialization
    model = get_model("CustomCNN", num_classes=80).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(optimizer, LR_SCHEDULER)

    best_lwlrap = 0.0

    # Training Loop
    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, lwlrap_score = validate(model, val_loader, criterion, device)

        logger.info(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LWLRAP: {lwlrap_score:.4f}"
        )

        # Update Learning Rate
        scheduler.step()

        # Save Best Model
        if lwlrap_score > best_lwlrap:
            best_lwlrap = lwlrap_score
            torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/best_model.pth")
            logger.info("Saved best model.")

    logger.info("Training completed.")


if __name__ == "__main__":
    train_model()
