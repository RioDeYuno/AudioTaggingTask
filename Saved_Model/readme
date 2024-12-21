
# Model Evaluation Guide

## Requirements
Ensure the following dependencies are installed:
- Python 3.8+
- PyTorch
- `transformers`
- `numpy`
- `scipy`

Install requirements using:
```bash
pip install torch transformers numpy scipy
```

---

## Files Description
1. **`Final_Model.safetensors`**: Contains the trained model weights.
2. **`config.yaml`**: Configuration file describing the model architecture and settings.

---

## Steps to Evaluate the Model

1. **Download and Prepare the Dataset**:
   Download the preprocessed Freesound Audio Tagging 2019 dataset from Kaggle:
   [Freesound Audio Tagging 2019 Mel-Spectrogram](https://www.kaggle.com/datasets/daisukelab/fat2019_prep_mels1)

   Place the dataset in your working directory, ensuring the paths to the `.npy` files are accessible.

2. **Load the Model**:
   Use the following script to load the model and configuration:
   ```python
   import torch
   from transformers import AutoModel

   # Load configuration
   import yaml
   with open("config.yaml", "r") as file:
       config = yaml.safe_load(file)

   # Load the model
   model = AutoModel.from_pretrained(".", config=config)
   model.eval()
   ```

3. **Load Preprocessed Audio Data**:
   Example for loading Mel-Spectrogram data:
   ```python
   import numpy as np

   # Path to mel-spectrogram data
   mel_path = "./path_to_dataset/mel_spectrogram_file.npy"
   mel_data = np.load(mel_path)

   # Convert to PyTorch tensor
   mel_tensor = torch.tensor(mel_data).unsqueeze(0)  # Add batch dimension
   ```

4. **Run the Model and Generate Predictions**:
   Use the model to generate predictions:
   ```python
   with torch.no_grad():
       outputs = model(mel_tensor)
   predictions = outputs.logits.argmax(dim=1)  # Adjust based on task
   ```

5. **Evaluate Model Performance**:
   Compare predictions with ground truth to calculate LWLRAP or other metrics:
   ```python
   from sklearn.metrics import label_ranking_average_precision_score

   # Example ground truth (one-hot encoded labels)
   ground_truth = np.array([...])  # Replace with true labels

   # Calculate LWLRAP
   lwlrap_score = label_ranking_average_precision_score(ground_truth, predictions.numpy())
   print(f"LWLRAP Score: {lwlrap_score}")
   ```

---

This guide provides a concise path to evaluate the model using the preprocessed Freesound dataset. Adjust file paths and code snippets to match your environment.
