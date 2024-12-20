#evaluate.py

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from data_preprocessing import FATValidDataset
from models import get_model
from config import *
from tqdm import tqdm


def calculate_lwlrap(truth, scores):
    """
    Calculate label-weighted label-ranking average precision (LWLRAP).
    """
    num_samples, num_classes = scores.shape
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))

    for sample_num in range(num_samples):
        pos_class_indices = np.flatnonzero(truth[sample_num] > 0)
        if len(pos_class_indices) == 0:
            continue
        retrieved_classes = np.argsort(scores[sample_num])[::-1]
        class_rankings = np.zeros(num_classes, dtype=np.int)
        class_rankings[retrieved_classes] = np.arange(num_classes)

        retrieved_class_true = np.zeros(num_classes, dtype=bool)
        retrieved_class_true[class_rankings[pos_class_indices]] = True

        retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
        precisions_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]]
            / (1 + class_rankings[pos_class_indices].astype(np.float))
        )
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = precisions_at_hits

    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / np.sum(labels_per_class)
    per_class_lwlrap = np.sum(precisions_for_samples_by_classes, axis=0) / np.maximum(
        1, labels_per_class
    )

    return per_class_lwlrap, weight_per_class


def evaluate_model(model, val_loader, device, tta=5):
    """
    Evaluate the model using validation data with Test Time Augmentation (TTA).

    Args:
        model: Trained model to evaluate.
        val_loader: Validation DataLoader.
        device: Device to run the evaluation on.
        tta: Number of TTA iterations.

    Returns:
        average_lwlrap: Average LWLRAP score.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for _ in range(tta):
            for data, labels, _ in tqdm(val_loader, desc="Evaluating"):
                data = data.to(device)
                outputs = model(data)
                all_preds.append(outputs.sigmoid().cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    per_class_lwlrap, weight_per_class = calculate_lwlrap(all_labels, all_preds)
    average_lwlrap = np.sum(per_class_lwlrap * weight_per_class)

    return average_lwlrap, per_class_lwlrap, weight_per_class


if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Validation Dataset
    val_dataset = FATValidDataset(
        fnames=val_fnames,
        mels=x_val,
        labels=y_val,
        transforms=transforms_dict["valid"],
        strength_list=strength_list_val,
        tta=VALIDATION_TTA,
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load Model
    model = get_model("CustomCNN", num_classes=80)
    model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/best_model.pth"))
    model.to(device)

    # Evaluate Model
    average_lwlrap, per_class_lwlrap, weight_per_class = evaluate_model(
        model, val_loader, device, tta=VALIDATION_TTA
    )

    # Print Results
    print(f"Average LWLRAP: {average_lwlrap:.4f}")
    print("Per Class LWLRAP:", per_class_lwlrap)
    print("Class Weights:", weight_per_class)
