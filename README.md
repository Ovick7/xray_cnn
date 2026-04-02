# Chest X-Ray Pneumonia Classification using CNN

## Overview

This project builds a Convolutional Neural Network (CNN) from scratch to classify paediatric chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**. The dataset contains 5,863 JPEG images sourced from paediatric patients aged one to five years, organised into train, validation, and test splits.

---

## Dataset

| Split      | Path                  |
|------------|-----------------------|
| Training   | `chest_xray/train/`   |
| Validation | `chest_xray/val/`     |
| Test       | `chest_xray/test/`    |

Each split contains two subfolders: `NORMAL` and `PNEUMONIA`.

The dataset is imbalanced, with pneumonia cases outnumbering normal cases roughly 3:1 in training. This is handled through weighted sampling and class-weighted loss.

---

## CNN Architecture

The model (`ChestXRayCNN`) is built entirely from scratch using PyTorch. No pre-trained weights are used.

### Feature Extractor

Four convolutional blocks, each containing:

```
Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout2d
```

| Block | Input Channels | Output Channels | Spatial Reduction |
|-------|---------------|-----------------|-------------------|
| 1     | 3             | 32              | 2x                |
| 2     | 32            | 64              | 2x                |
| 3     | 64            | 128             | 2x                |
| 4     | 128           | 256             | 2x                |

After block 4, an `AdaptiveAvgPool2d(4, 4)` reduces the spatial dimensions to a fixed 4x4 output regardless of input size.

### Classifier Head

```
Flatten -> Linear(4096, 512) -> BatchNorm1d -> ReLU -> Dropout(0.5)
        -> Linear(512, 128)  -> BatchNorm1d -> ReLU -> Dropout(0.3)
        -> Linear(128, 2)
```

### Design Choices

- **Double convolution per block**: Two consecutive conv layers per block increase receptive field and depth without excessive pooling.
- **BatchNorm after every conv**: Stabilises training, reduces sensitivity to learning rate, and acts as a regulariser.
- **Dropout2d in conv blocks**: Randomly zeros entire feature maps, acting as stronger spatial regularisation than standard Dropout.
- **AdaptiveAvgPool2d**: Decouples the classifier from fixed input size and reduces spatial dimensions smoothly before the fully connected layers.
- **Progressive channel doubling**: 32 -> 64 -> 128 -> 256 follows the standard CNN design principle of increasing depth while reducing spatial dimensions.

---

## Approach and Methodology

### Class Imbalance Handling

Two complementary strategies are applied simultaneously:

1. **WeightedRandomSampler**: Over-samples the minority class during training so each batch has balanced class representation.
2. **Weighted CrossEntropyLoss**: Assigns a higher loss penalty to the minority class, making the model more sensitive to misclassifying normal cases.

In medical imaging, recall for the PNEUMONIA class is critical. Missing a pneumonia diagnosis (false negative) is clinically far more harmful than a false alarm.

### Data Augmentation

Applied only to the training set to improve generalisation:

- Random horizontal flip
- Random rotation up to 10 degrees
- Colour jitter on brightness and contrast by 0.2
- Normalisation using ImageNet channel statistics

Validation and test sets use only resize and normalisation.

### Training Configuration

| Parameter      | Value                                        |
|----------------|----------------------------------------------|
| Image size     | 224 x 224                                    |
| Batch size     | 32                                           |
| Epochs         | 30                                           |
| Optimiser      | Adam (weight decay = 1e-4)                   |
| Initial LR     | 1e-3                                         |
| LR scheduler   | ReduceLROnPlateau (patience=4, factor=0.5)   |
| Loss function  | Weighted CrossEntropyLoss                    |

The learning rate scheduler halves the LR when validation loss stops improving for 4 consecutive epochs, allowing finer convergence in later stages.

### Model Selection

The checkpoint with the lowest validation loss is saved during training. This checkpoint is reloaded for final test evaluation, preventing the reporting of results from an overfitted epoch.

---

## Project Structure

```
chest_xray_cnn/
    model.py             - CNN architecture definition
    train.py             - Training and evaluation pipeline
    predict.py           - Single-image inference script
    requirements.txt     - Python dependencies
    README.md            - This file

Generated after training:
    best_model.pth       - Best model weights
    training_history.png - Loss and accuracy curves
    confusion_matrix.png - Confusion matrix on test set
    roc_curve.png        - ROC curve with AUC score
```

---

## Setup and Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Ensure your dataset is structured as follows:

```
chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
```

Update `DATASET_PATH` in `train.py` to point to your local dataset location.

### 3. Train the model

```bash
python train.py
```

Training prints per-epoch metrics, saves the best checkpoint, and generates evaluation plots.

### 4. Run inference on a single image

```bash
python predict.py path/to/xray.jpg --checkpoint best_model.pth
```

---

## Evaluation Metrics

The following metrics are computed on the held-out test set:

- **Accuracy**: Overall fraction of correctly classified images.
- **Precision, Recall, F1-score**: Per-class breakdown from the classification report.
- **Confusion matrix**: Visualises TP, TN, FP, FN counts.
- **ROC-AUC**: Measures the model's ability to discriminate between classes across all thresholds.

Recall for the PNEUMONIA class is the most clinically meaningful metric.

---

## Limitations

- The original validation set is very small (16 images). Validation metrics may have high variance. Consider carving out a larger validation split from training data for more reliable model selection.
- The model is trained from scratch on ~5,800 images, which is relatively small for a deep CNN. Performance may plateau compared to transfer learning approaches.
- This model is for research and educational purposes only and has not undergone clinical validation.
