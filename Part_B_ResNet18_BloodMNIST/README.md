# Part B: Eight-Class Blood Cell Classification Using ResNet-18

## 1. Project Overview

This project investigates the classification of eight blood cell types from microscopic images using transfer learning with an ImageNet-pretrained ResNet-18 convolutional neural network.

Two configurations are compared:

1. **Frozen ResNet-18:** The pretrained feature extraction layers remain frozen, and only the final classification layer is trained.
2. **Fine-tuned ResNet-18:** All network layers are trainable, allowing the pretrained features to adapt to blood cell microscopy images.

The objective is to investigate whether fine-tuning an ImageNet-pretrained model improves multiclass blood cell classification compared with using a frozen pretrained backbone.

## 2. Research Question and Hypothesis

**Research Question**

To what extent does fine-tuning an ImageNet-pretrained ResNet-18 improve multiclass blood cell classification compared with using a frozen pretrained backbone?

**Research Hypothesis**

We hypothesise that fine-tuning an ImageNet-pretrained ResNet-18 will improve blood cell classification performance compared with keeping its pretrained backbone frozen. Specifically, we expect fine-tuning to produce higher test accuracy and macro F1-score by adapting the pretrained features to the visual characteristics of BloodMNIST images.

We evaluate this hypothesis by measuring the paired differences in test accuracy and macro F1-score between the two configurations across three random seeds, using bootstrap confidence intervals to assess the uncertainty of the improvements.

## 3. Dataset Description

The experiments use the BloodMNIST dataset from the MedMNIST collection.

| Property | Description |
|---|---|
| Dataset | BloodMNIST |
| Original image resolution | 128 × 128 pixels |
| Image channels | RGB |
| Original number of images | 17,092 |
| Retained images after duplicate removal | 17,075 |
| Number of classes | 8 |
| Training images after cleaning | 11,951 |
| Validation images after cleaning | 1,709 |
| Test images after cleaning | 3,415 |

### Blood Cell Classes

The dataset contains eight blood cell categories:

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

### Data Quality and Duplicate Removal

Before training, an exact duplicate audit was performed using image hashes. The audit identified 17 exact duplicate groups, including duplicates within and across the original dataset splits.

Duplicate images were removed using a deterministic retention order: training, validation and then test, retaining the earliest image within each split. One duplicate group contained identical image pixels with conflicting Eosinophil and Neutrophil labels. The training occurrence was retained according to the predefined policy; this does not establish which original label is biologically correct.

The cleaned dataset contains 17,075 images, with 11,951 training, 1,709 validation and 3,415 test images.

The dataset audit also identified five potential near-twin relationships between training and test images. These should be considered when interpreting test performance.

## 4. Image Preprocessing

The following preprocessing operations were applied:

- **Resizing:** Original images were resized from 128 × 128 to 224 × 224 pixels to match the ResNet-18 input dimensions.
- **Data augmentation:** Random horizontal and vertical flips and rotations of up to ±15° were applied only to training images.
- **Tensor conversion:** Images were converted into PyTorch tensors.
- **Normalisation:** RGB channels were normalised using ImageNet mean and standard deviation.

Validation and test images received deterministic preprocessing without random augmentation.

Exploratory data analysis included image dimension inspection, pixel intensity distribution analysis, visualisation of all eight classes and class distribution analysis.

## 5. Model Architecture and Experimental Setup

### ResNet-18

ResNet-18 is a convolutional neural network based on residual learning. ImageNet-pretrained weights were used as the starting point, and the original fully connected classification layer was replaced with a new layer containing eight output neurons.

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

Both configurations were evaluated using three random seeds: 42, 43 and 44. Random seeds were controlled across Python, NumPy and PyTorch. Deterministic cuDNN settings were enabled where supported.

For the frozen configuration, only the final classification layer was trained, while the pretrained backbone and its Batch Normalisation statistics remained frozen. For the fine-tuned configuration, all network layers were trainable.

### Checkpoint Selection and Evaluation Metrics

The checkpoint with the highest validation accuracy was selected for each run. The test set was reserved for final evaluation and was not used for checkpoint selection.

The models were evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score

Macro-averaged metrics were used to give equal importance to all eight blood cell classes.

## 6. Experimental Results and Statistical Analysis

The following table presents the mean test performance across three random seeds. Values are reported as mean ± sample standard deviation.

| Metric | Frozen ResNet-18 | Fine-tuned ResNet-18 |
|---|---:|---:|
| Accuracy | 88.97 ± 0.22% | 98.81 ± 0.02% |
| Macro Precision | 89.93 ± 0.30% | 98.98 ± 0.04% |
| Macro Recall | 87.06 ± 0.88% | 98.84 ± 0.01% |
| Macro F1-score | 88.11 ± 0.60% | 98.91 ± 0.02% |

### Key Findings

- Fine-tuning achieved a mean test accuracy of 98.81%, compared with 88.97% for the frozen model.
- Fine-tuning improved mean test accuracy by 9.84 percentage points.
- Macro F1-score improved by 10.80 percentage points.
- The low standard deviations across three runs indicate consistent performance across the tested random seeds.

These results suggest that adapting the pretrained feature representations to BloodMNIST substantially improved classification performance.

### Statistical Analysis

Paired bootstrap confidence intervals were calculated using 2,000 resamples of the 3,415 test images.

| Metric | Fine-tuned minus frozen | 95% Bootstrap CI |
|---|---:|---:|
| Accuracy improvement | 9.82 percentage points | 8.87–10.80 |
| Macro F1 improvement | 10.79 percentage points | 9.75–11.84 |

Both confidence intervals are entirely above zero, supporting the observed improvement from fine-tuning on the evaluated test set.

### Class-wise Analysis and Confusion Matrix

The confusion matrices showed that most remaining errors in the fine-tuned model involved visually similar blood cell categories.

The most frequent misclassification was neutrophils predicted as immature granulocytes, with 51 errors across the three seeds. This was followed by immature granulocytes predicted as neutrophils, with 21 errors. Other recurring errors included monocytes predicted as immature granulocytes (11 errors) and erythroblasts predicted as immature granulocytes (9 errors).

These findings indicate that distinguishing between neutrophils and immature granulocytes remained a particular challenge despite the high overall classification performance. The error counts are pooled across the three runs and do not represent unique test images.

### Focus-Class Analysis: Immature Granulocytes

Immature granulocytes were selected for additional evaluation because of their potential visual similarity to other granulocyte types, particularly neutrophils.

| Configuration | Mean Recall | 95% Bootstrap CI |
|---|---:|---:|
| Frozen ResNet-18 | 84.80% | 81.98–87.56% |
| Fine-tuned ResNet-18 | 98.39% | 97.58–99.08% |

Fine-tuning improved immature granulocyte recall by 13.59 percentage points. This indicates a substantial improvement in identifying the focus class, although some confusion with neutrophils remained.

## 7. Explainability Using Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) was implemented using the final convolutional layer of ResNet-18 to visualise image regions influencing model predictions.

The explainability analysis includes:

- Grad-CAM visualisations for correctly classified test images.
- Grad-CAM visualisations for incorrectly classified test images.
- A random-weight sanity check comparing heatmaps from the trained model and a randomly initialised ResNet-18 using the same test image.

The mean absolute difference between the trained and random-weight heatmaps in this comparison was 0.3978.

These visualisations provide a qualitative indication of image regions influencing predictions. However, Grad-CAM does not establish that the highlighted regions correspond to biologically meaningful features or prove that the model uses clinically relevant information.

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
│   ├── near_duplicate_image_pairs.png
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

The dataset and trained model checkpoints are not included in the repository. Training histories, evaluation summaries and generated figures are provided.

## 9. Reproducibility and Implementation

The project was implemented using Python and PyTorch in JupyterLab.

### Development Environment

| Component | Version / Configuration |
|---|---|
| Python | 3.11.13 |
| PyTorch | 2.9.1 |
| Torchvision | 0.24.1 |
| CUDA | 12.8 |
| GPU | NVIDIA RTX A6000 |
| Batch size | 64 |
| Training epochs | 10 |
| Random seeds | 42, 43, 44 |

The `requirements.txt` file specifies the package versions used in the experiment.

To install the dependencies, run the following command from the `Part_B_ResNet18_BloodMNIST/` directory in a compatible Python 3.11 environment configured for NVIDIA CUDA:

```bash
python -m pip install -r requirements.txt
```

The notebook contains the dataset loading, duplicate auditing, preprocessing, model implementation, training, evaluation and explainability workflows.

Six model checkpoints (three seeds for each configuration) were generated and used for final evaluation. The checkpoint files are retained separately because of their size and can be regenerated by executing the corresponding training cells.

## 10. Limitations and Clinical Responsibility

Although the fine-tuned model achieved high classification performance on the cleaned BloodMNIST test set, several limitations should be considered:

- **Dataset generalisation:** The experiments used one public benchmark dataset. Performance on images from other hospitals, laboratories, patient populations, microscopes or staining protocols has not been established.
- **Patient-level independence:** Patient and source identifiers were unavailable, so patient-level separation across dataset splits could not be independently verified.
- **Near-duplicate images:** The dataset audit identified five potential near-twin relationships between training and test images. Their potential influence on test performance should be considered.
- **Class similarity:** Remaining confusion between neutrophils and immature granulocytes indicates that some visually similar categories remain challenging.
- **Explainability:** Grad-CAM provides qualitative visualisations but cannot establish that the model relies on clinically meaningful biological features.
- **Limited repeated runs:** Three random seeds provide an indication of consistency, but additional runs would provide stronger evidence of reproducibility.

The project is intended for academic and research purposes only. The models are not clinically validated and must not be used as a replacement for professional medical judgement or diagnosis. Independent external validation and specialist review would be necessary before considering clinical applications.

## 11. References

1. Acevedo, A., et al. (2020). A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. *Data in Brief*. https://doi.org/10.1016/j.dib.2020.105474

2. Yang, J., et al. (2023). MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*. https://doi.org/10.1038/s41597-022-01721-8

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.

5. PyTorch Documentation. https://pytorch.org/docs/

6. Torchvision Documentation. https://pytorch.org/vision/stable/
