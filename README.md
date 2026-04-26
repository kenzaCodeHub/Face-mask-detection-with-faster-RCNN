# Face Mask Detection with Faster R-CNN

Object detection model that identifies whether people are wearing face masks correctly, incorrectly, or not at all, using **Faster R-CNN (ResNet50-FPN)** fine-tuned on the Face Mask Detection dataset.

## Classes

| Class | Description |
|---|---|
| `with_mask` | Mask worn correctly |
| `without_mask` | No mask |
| `mask_weared_incorrect` | Mask worn incorrectly |

## Approach

1. **Data**: [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) — 853 images with Pascal VOC XML bounding box annotations
2. **Split**: 70% train (597) / 15% validation (128) / 15% test (128)
3. **Augmentation**: color jitter, auto-contrast, sharpness, grayscale, Gaussian blur
4. **Model**: Faster R-CNN with ResNet50-FPN backbone pre-trained on COCO, classification head replaced for 4 classes (3 + background)
5. **Training**: SGD (lr=5e-4, momentum=0.9, weight decay=1e-4), 15 epochs on GPU
6. **Evaluation**: test loss, per-class Precision / Recall / F1-Score (IoU ≥ 0.5), visual predictions with bounding boxes

## Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| `with_mask` | 82.4% | 87.6% | **84.9%** |
| `without_mask` | 75.9% | 85.6% | **80.5%** |
| `mask_weared_incorrect` | 0.0% | 0.0% | 0.0% |

**Test loss: 0.263**

The `mask_weared_incorrect` class has zero detections due to severe class imbalance (~3% of all annotations). Possible improvements include oversampling, focal loss, or targeted augmentation for this minority class.

## Tech Stack

Python, PyTorch, torchvision (Faster R-CNN), Matplotlib, PIL, Google Colab (GPU)
