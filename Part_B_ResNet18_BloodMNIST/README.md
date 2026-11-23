# Part B: ResNet-18 Blood Cell Classification

## Project Overview
This part of the Biomedical Image Analysis project investigates the classification of eight blood cell types using transfer learning with a pretrained ResNet-18 model.

## Research Question
To what extent does fine-tuning an ImageNet-pretrained ResNet-18 improve multiclass blood cell classification compared with using a frozen pretrained backbone?

## Dataset
- **Dataset:** BloodMNIST (128×128)
- **Number of images:** 17,092
- **Number of classes:** 8
- **Input format:** RGB microscopy images
- **Splits:** Official training, validation and test sets

## Models
1. Frozen pretrained ResNet-18
2. Fine-tuned pretrained ResNet-18

## Project Structure

- `Part_B_ResNet18_BloodMNIST.ipynb` — Main implementation and experiments
- `README.md` — Project documentation
- `requirements.txt` — Python dependencies
- `figures/` — Dataset visualisations, confusion matrices, training curves and Grad-CAM images
- `results/` — Evaluation metrics, performance summaries and bootstrap confidence intervals
  - `histories/` — Training histories for frozen and fine-tuned models across three random seeds

## Evaluation
Models are evaluated using accuracy, macro precision, macro recall and macro F1-score across three random seeds.

## Explainability
Grad-CAM is used to visualise image regions influencing model predictions.


