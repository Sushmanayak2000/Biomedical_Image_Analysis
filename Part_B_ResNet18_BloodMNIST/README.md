# Part B: Eight-Class Blood Cell Classification Using ResNet-18

## 1. Project Overview

This project investigates the classification of eight types of blood cells from microscopic images using transfer learning with an ImageNet-pretrained ResNet-18 convolutional neural network.

The study compares two approaches:

1. **Frozen ResNet-18:** The pretrained feature extraction layers are frozen, and only the final classification layer is trained.
2. **Fine-tuned ResNet-18:** All layers of the pretrained network are trainable and updated during training.

The objective is to investigate whether fine-tuning an ImageNet-pretrained model improves multiclass blood cell classification compared with using a frozen pretrained backbone.

## 2. Research Question

To what extent does fine-tuning an ImageNet-pretrained ResNet-18 improve multiclass blood cell classification compared with using a frozen pretrained backbone?

## 3. Dataset Description

The experiments use the BloodMNIST dataset from the MedMNIST collection.

| Property | Description |
|---|---|
| Dataset | BloodMNIST |
| Image resolution | 128 × 128 pixels |
| Image channels | RGB |
| Total images | 17,092 |
| Number of classes | 8 |
| Training images | 11,959 |
| Validation images | 1,712 |
| Test images | 3,421 |

### Blood Cell Classes

The dataset contains the following eight classes:

1. Basophil
2. Eosinophil
3. Erythroblast
4. Immature Granulocytes
5. Lymphocyte
6. Monocyte
7. Neutrophil
8. Platelet

### Dataset Source

- Dataset: BloodMNIST (128 × 128)
- Download source: [Zenodo – BloodMNIST 128×128](https://zenodo.org/records/10519652)
- Original dataset publication: [Acevedo et al. (2020)](https://doi.org/10.1016/j.dib.2020.105474)
- Dataset collection: [MedMNIST](https://medmnist.com/)

The dataset is not included in this repository and must be downloaded separately.

## 4. Image Preprocessing

The following preprocessing operations were applied:

- Resized images from 128 × 128 to 224 × 224 pixels to match the ResNet-18 input requirements.
- Applied ImageNet mean and standard deviation normalisation.
- Applied horizontal and vertical flipping during training.
- Applied random rotation of up to ±15 degrees during training.
- Used the official training, validation and test splits.

Exploratory data analysis included image dimension inspection, pixel intensity distribution analysis, visualisation of all eight classes and class distribution analysis.

## 5. Model Architecture and Experimental Setup

### ResNet-18

ResNet-18 is a convolutional neural network based on residual learning. The pretrained ImageNet weights were used as the starting point, and the final fully connected layer was replaced with an eight-class classification layer.

### Experimental Configurations

| Configuration | Frozen ResNet-18 | Fine-tuned ResNet-18 |
|---|---|---|
| Pretrained weights | ImageNet | ImageNet |
| Backbone | Frozen | Trainable |
| Final classification layer | Trainable | Trainable |
| Optimiser | AdamW | AdamW |
| Learning rate | 0.001 | 0.0001 |
| Weight decay | 0.0001 | 0.0001 |
| Loss function | Cross-Entropy Loss | Cross-Entropy Loss |
| Epochs | 10 | 10 |
| Batch size | 64 | 64 |

Both configurations were evaluated using three random seeds: 42, 43 and 44.

### Evaluation Metrics

The models were evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score

Macro-averaged metrics were used to give equal importance to all eight blood cell classes.

## 6. Experimental Results

The following table presents the mean test performance across the three random seeds.

| Metric | Frozen ResNet-18 | Fine-tuned ResNet-18 |
|---|---:|---:|
| Accuracy | 89.03% | 98.80% |
| Macro Precision | 89.93% | 98.90% |
| Macro Recall | 87.48% | 98.86% |
| Macro F1-score | 88.44% | 98.88% |

### Key Findings

- Fine-tuning achieved a mean test accuracy of 98.80%.
- The frozen model achieved a mean test accuracy of 89.03%.
- Fine-tuning improved test accuracy by 9.77 percentage points.
- The fine-tuned model achieved higher macro precision, recall and F1-score.
- Results were evaluated across three random seeds to examine consistency.

### Statistical Analysis

Paired bootstrap confidence intervals were calculated using 2,000 resamples of the test images.

The estimated improvement in accuracy was 9.77 percentage points, with a 95% bootstrap confidence interval of [8.85, 10.72].

The macro F1-score improvement was 10.44 percentage points, with a 95% bootstrap confidence interval of [9.43, 11.50].

These results provide evidence of improved performance after fine-tuning on the BloodMNIST test set.

## 7. Explainability Using Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) was used to visualise image regions that influenced the model's predictions.

The explainability analysis includes:

- Grad-CAM visualisations for correct predictions.
- Grad-CAM visualisations for incorrect predictions.
- A random-weight sanity check comparing trained and randomly initialised model explanations.

The mean absolute difference in the heatmaps for the random-weight sanity check was 0.4587.

Grad-CAM provides a visual indication of model attention but does not prove that the model is using clinically meaningful features.

## 8. Project Structure

```text
Part_B_ResNet18_BloodMNIST/
│
├── Part_B_ResNet18_BloodMNIST.ipynb
├── README.md
├── requirements.txt
│
├── figures/
│   ├── aggregated_finetuned_confusion_matrix.png
│   ├── bloodmnist_sample_images.png
│   ├── class_distribution_comparison.png
│   ├── confusion_matrices_comparison.png
│   ├── gradcam_correct_predictions.png
│   ├── gradcam_incorrect_predictions.png
│   ├── gradcam_random_weight_sanity_check.png
│   ├── loss_curves_comparison.png
│   └── pixel_intensity_distribution.png
│
└── results/
    ├── bootstrap_confidence_intervals.csv
    ├── class_distribution.csv
    ├── final_test_performance_summary.csv
    │
    └── histories/
        ├── finetuned_seed_42_history.csv
        ├── finetuned_seed_43_history.csv
        ├── finetuned_seed_44_history.csv
        ├── frozen_seed_42_history.csv
        ├── frozen_seed_43_history.csv
        └── frozen_seed_44_history.csv
```

## 9. Reproducibility and Implementation

The project was implemented using Python and PyTorch in JupyterLab.

### Development Environment

| Component | Version / Configuration |
|---|---|
| Python | 3.11.13 |
| PyTorch | 2.9.1 |
| Torchvision | 0.24.1 |
| CUDA | 12.8 |
| GPU | NVIDIA A6000 |
| Batch size | 64 |
| Training epochs | 10 |
| Random seeds | 42, 43, 44 |

The notebook contains the dataset loading, preprocessing, model implementation, training, evaluation and explainability workflows.

Training histories, evaluation summaries, confusion matrices and Grad-CAM visualisations are provided in the repository.

### Model Checkpoints

The trained model checkpoint files (`.pth`) are not included in this repository because of their size. They are retained separately.

The model checkpoints can be regenerated by executing the corresponding training cells in the notebook.

## 10. Limitations and Clinical Responsibility

Although the fine-tuned model achieved high classification performance on the BloodMNIST test set, several limitations should be considered:

- Experiments were conducted using a single public dataset.
- Generalisation to other hospitals, microscopes, staining procedures and patient populations has not been established.
- Patient-level separation could not be independently verified.
- Grad-CAM visualisations cannot establish that the model relies on clinically meaningful biological features.
- External validation using independent datasets is required before considering clinical applications.

This project is intended for academic and research purposes only. The models are not clinically validated and must not be used as a replacement for professional medical judgement or diagnosis.

## 11. Use of AI Tools

ChatGPT was used as an assistance tool to understand technical concepts, troubleshoot implementation issues, interpret experimental results and improve project documentation.

Experimental results were obtained through execution of the implemented notebook and were not generated by the AI tool.

## 12. References

1. Acevedo, A., et al. (2020). A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. *Data in Brief*. https://doi.org/10.1016/j.dib.2020.105474

2. Yang, J., et al. (2023). MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*. https://doi.org/10.1038/s41597-022-01721-8

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.

5. PyTorch Documentation. https://pytorch.org/docs/

6. Torchvision Documentation. https://pytorch.org/vision/stable/
