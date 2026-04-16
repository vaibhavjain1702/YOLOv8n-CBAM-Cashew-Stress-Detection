# YOLOv8n-CBAM: Cashew Crop Stress Detection

Modified YOLOv8n object detection architecture with **CBAM (Convolutional Block Attention Module)** for automated cashew crop stress identification from drone imagery.

## Overview

- **Dataset:** [CADI-AI](https://www.kaggle.com/datasets/karaagroaiprojects/cadi-ai) (4,736 images, 3 classes: abiotic, insect, disease)
- **Baseline:** YOLOv8n (pretrained on COCO)
- **Modification:** CBAM injected at backbone layers 4 (P3, 64ch) and 6 (P4, 128ch) via runtime wrapping — preserves 100% pretrained weights
- **Environment:** Google Colab (T4 GPU)

## Results

| Metric | Baseline | Modified (CBAM) | Improvement |
|:-------|:--------:|:---------------:|:-----------:|
| mAP@50 | 0.1723 | **0.3425** | +99% |
| mAP@50-95 | 0.0569 | **0.1309** | +130% |
| Precision | 0.2783 | **0.4197** | +51% |
| Recall | 0.2090 | **0.3506** | +68% |
| F1 Score | 0.2387 | **0.3820** | +60% |

## Key Features

- **Runtime CBAM Injection** — preserves pretrained weights by wrapping C2f layers instead of modifying YAML config
- **XAI Attention Heatmaps** — visualizes where the model focuses using channel-weighted spatial attention maps
- **Complete Pipeline** — dataset download, preprocessing, training, evaluation, and visualization in a single notebook

## How to Run

1. Open `DL_Assignment.ipynb` in Google Colab
2. Run all cells sequentially (requires T4 GPU runtime)
3. Training takes ~2 hours for both models (30 epochs each)

## Tech Stack

`Python` · `PyTorch` · `Ultralytics YOLOv8` · `OpenCV` · `Matplotlib`
