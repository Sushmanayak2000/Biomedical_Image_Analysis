# Eight-Class Blood Cell Classification from Microscopy Images Using a Scratch CNN and Pretrained ResNet-18

**Biomedical Image Analysis · M.Sc. Applied Data Science and AI · SRH University Munich · Winter semester 2026/27**

**Authors: Ishika Thakur (Part A) and Sushma (Part B)**
**Repository:** https://github.com/Sushmanayak2000/Biomedical_Image_Analysis  
**Date:** 24 November 2026

## Abstract

We investigated two training choices for classifying eight blood cell types from RGB microscopy images in BloodMNIST. In Part A, we trained a custom CNN from scratch with and without random horizontal and vertical flips across three paired seeds. Mean test macro F1 was 97.13% ± 0.63 without flips and 95.68% ± 1.37 with flips, a mean paired decrease of 1.45 percentage points. In Part B, the fine-tuned ImageNet-pretrained ResNet-18 achieved 98.80% ± 0.18 test accuracy, compared with 89.03% ± 0.27 for the frozen backbone. These results describe performance on BloodMNIST. Performance on independent clinical datasets remains unverified.

## Repository Structure

| File or folder | Purpose |
| --- | --- |
| [`Part A/`](./Part%20A/) | Contains the custom CNN trained from scratch on BloodMNIST, the comparison with and without data augmentation, and the saved evaluation results. |
| [`Part_B_ResNet18_BloodMNIST/`](./Part_B_ResNet18_BloodMNIST/) | Contains the ResNet-18 transfer learning experiments for BloodMNIST classification and their evaluation results. |
| [`lab_results/`](./lab_results/) | Contains the instructor-provided lab notebook and the outputs from the in-class exercises for lab Parts A, B, and C. |
| [`BloodMNIST_SharedPreps.ipynb`](./BloodMNIST_SharedPreps.ipynb) | Records the initial BloodMNIST dataset exploration and preparation that both team members worked on together. |
| [`README.md`](./README.md) | Introduces the project and explains how the repository is organized. |

## 1 · The biomedical problem (15%, with Section 2)

- **Clinical question** — How effectively can CNN-based models classify eight types of blood cells from microscopy images?
- **Task type** — Multiclass image classification. The source image is a 128 × 128 RGB microscopy image of an individual blood cell. The output is one of eight blood cell labels.
- **Why it is hard** — Some blood cell types differ in subtle features such as nucleus shape, staining patterns and cellular texture. Class frequencies also vary, so overall accuracy alone can hide poor performance on individual classes.

## 2 · The data (15%, with Section 1)

| Item | Description |
| --- | --- |
| Source and version | BloodMNIST 128 × 128 from MedMNIST v2. Archive `bloodmnist_128.npz`, Zenodo record 10519652. |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0). |
| Modality | Two-dimensional RGB microscopy images of individual blood cells. |
| Size | 17,092 images at 128 × 128 pixels across eight classes. |
| Labelling | Clinical pathologists at the Hospital Clínic of Barcelona labelled the original images (Acevedo et al., 2020). Inter-annotator agreement was not reported. |
| Classes | Basophil, eosinophil, erythroblast, immature granulocytes, lymphocyte, monocyte, neutrophil and platelet. |
| Part A preprocessing | We kept the original 128 × 128 resolution and computed RGB normalisation statistics using retained training images only. One training condition used random horizontal and vertical flips. Both conditions used identical deterministic preprocessing for validation and test. |
| Part B preprocessing | We resized images to 224 × 224 and used ImageNet normalisation. Training augmentation included horizontal and vertical flips and rotation up to ±15°. |
| Shared with Deep Learning | The Deep Learning project used EuroSAT satellite images. |

Both parts started with the official BloodMNIST splits. Part A removed exact pixel duplicates before preprocessing. Part B used the official split sizes.

| Split | Official images | Retained for Part A | Excluded from Part A |
| --- | ---: | ---: | ---: |
| Training | 11,959 | 11,951 | 8 |
| Validation | 1,712 | 1,709 | 3 |
| Test | 3,421 | 3,415 | 6 |

For Part A, we verified the original shapes, label range and coverage of all eight classes. We audited raw pixels for exact matches and kept the first occurrence in train → validation → test order, using the original row order within each split. One excluded test duplicate had a conflicting label. We kept the retained row's original label and recorded both labels in the split manifest. Validation and test labels had no influence on which training rows we kept.

The retained Part A splits contained zero exact pixel duplicates within or across splits. The archive provides image and label arrays, so patient and source overlap could not be verified. Near-duplicate overlap was not tested.

![One retained training example from each BloodMNIST class](Part%20A/results%20%28part%20A%29/bloodmnist_class_grid.png)

*Figure 1. Representative 128 × 128 RGB images from the retained Part A training split.*

## 3 · Experimental design (15%)

### Part A · Scratch CNN and data augmentation

We tested whether random horizontal and vertical flips improve classification performance for a custom CNN trained from scratch. The CNN trained without augmentation served as the reference condition.

The model has three convolutional blocks with 16, 32 and 64 filters. Each block uses a 3 × 3 convolution, batch normalisation, ReLU and 2 × 2 max pooling. Adaptive average pooling produces 4 × 4 feature maps, which are flattened and passed to a linear layer with eight outputs. The model has 31,896 trainable parameters.

We ran both conditions for seeds 42, 43 and 44. Within each pair, we matched initial weights, training image order per epoch, architecture, loss, optimiser, batch size and epoch budget. We used AdamW with a learning rate of 0.001 and weight decay of 0.0001, cross-entropy loss, a batch size of 64 and 10 epochs. In the augmented condition, we applied horizontal and vertical flips independently, each with a probability of 0.5.

![Examples of original training images and random flip draws](Part%20A/results%20%28part%20A%29/augmentation_preview.png)

*Figure 2. The first column shows the original image. The other columns show independent draws from the augmentation pipeline.*

### Part B · Transfer learning with ResNet-18

We compared an ImageNet-pretrained ResNet-18 with a frozen backbone against a fine-tuned ResNet-18. In the frozen configuration, we trained only the final classification layer. In the fine-tuned configuration, we trained all layers. Both configurations used cross-entropy loss. Part B did not separately evaluate a largest-class or classical machine-learning baseline.

Both configurations used the official BloodMNIST splits, a batch size of 64, 10 epochs per run and seeds 42, 43 and 44. The learning rates were 0.001 for the frozen model and 0.0001 for the fine-tuned model. We evaluated accuracy, macro precision, macro recall and macro F1.

## 4 · Implementation and reproducibility (25%)

### Part A

- **Notebook and environment** — We ran [PartA_cnn.ipynb](Part%20A/PartA_cnn.ipynb) in JupyterLab on Compute Data Lab. We verified CUDA execution and checked that a batch of 128 × 128 RGB images produced eight logits per image. The runs used an NVIDIA RTX A6000 with PyTorch 2.9.1 and CUDA 12.8. Package versions, normalisation statistics and training settings are recorded in the [run configuration](Part%20A/results%20%28part%20A%29/run_config.json).
- **Checkpoint selection** — We selected the checkpoint with the highest validation macro F1 for each run, keeping the earlier epoch in a tie. We froze the training settings and checkpoint choices before final test evaluation.
- **Resume and outputs** — We saved full epoch histories and progress checkpoints for all six runs. The notebook can resume an interrupted run from the last saved epoch and reuse a completed run with matching settings. We also saved the split manifest, validation and test metrics, confusion matrix counts and test predictions with original row IDs. The notebook verified that all 54 expected output files were present and nonempty.

### Part B

- **Notebook and environment** — We implemented the experiment in `Part_B_ResNet18_BloodMNIST.ipynb` using Python and PyTorch. The Part B `requirements.txt` documents the required packages.
- **Seeds and hardware** — Both configurations ran on an NVIDIA A6000 GPU for 10 epochs per seed with a batch size of 64. The seeds were 42, 43 and 44. Total training time was not reported.
- **Outputs** — The notebook generates training histories, validation and test metrics, classification reports, confusion matrices, bootstrap confidence intervals, Grad-CAM visualisations and prediction examples. Checkpoints and histories are saved in the Part B results directory.

## 5 · Evaluation, validation and critical analysis (15%)

### Part A

We evaluated the six frozen checkpoints on the same 3,415 retained test images. We reported accuracy, balanced accuracy, macro F1 and per-class precision, recall and F1. Macro F1 gives each of the eight classes equal weight when averaging class-level F1 scores.

| Training condition | Test accuracy (%) | Test macro F1 (%) | Test balanced accuracy (%) |
| --- | ---: | ---: | ---: |
| Without augmentation | 97.22 ± 0.52 | 97.13 ± 0.63 | 96.88 ± 0.80 |
| With augmentation | 95.99 ± 1.13 | 95.68 ± 1.37 | 95.27 ± 1.52 |

Values are mean ± sample standard deviation across the three seeds.

| Seed | Accuracy change (pp) | Macro F1 change (pp) | Balanced accuracy change (pp) |
| --- | ---: | ---: | ---: |
| 42 | −0.26 | −0.27 | −0.14 |
| 43 | −2.90 | −3.46 | −3.93 |
| 44 | −0.53 | −0.62 | −0.77 |

Each change is the score with augmentation minus the score without augmentation, in percentage points (pp). Flips reduced test macro F1 for all three seeds, with a mean paired change of **−1.45 pp**. Accuracy and balanced accuracy also declined in every pair.

Mean class F1 decreased for seven of the eight classes. The largest drops were for monocytes (−3.36 pp), basophils (−2.53 pp), immature granulocytes (−2.31 pp) and lymphocytes (−2.05 pp). Platelets were the only class with an increase (+0.04 pp).

We kept the shared 10-epoch budget because validation macro F1 did not improve consistently near the end across the six runs. The findings apply to the tested combination of horizontal and vertical flips under this training protocol. Patient and source overlap remain unverifiable, and we did not evaluate performance on an external dataset.

![Test confusion matrices for the six Part A runs](Part%20A/results%20%28part%20A%29/runs/834896128886ba35/test_confusion_matrices.png)

*Figure 3. Columns show seeds 42, 43 and 44. The top row shows runs without augmentation and the bottom row shows runs with augmentation.*

### Part B

| Setting | Mean test accuracy (%) | Reported spread (pp) | Seeds |
| --- | ---: | ---: | --- |
| Frozen ResNet-18 | 89.03 | 0.27 | 42, 43, 44 |
| Fine-tuned ResNet-18 | 98.80 | 0.18 | 42, 43, 44 |

The fine-tuned model also achieved mean test macro precision of 98.90%, macro recall of 98.86% and macro F1 of 98.88%. Frequent classification errors involved neutrophils and immature granulocytes. We also observed confusion involving immature granulocytes, monocytes and lymphocytes.

We selected checkpoints using validation accuracy and evaluated them on the official 3,421-image test split. Inter-observer agreement was unavailable, so we could not establish a human performance reference. Parts A and B used different preprocessing, split handling and checkpoint selection, which limits direct comparison of their scores.

## 6 · Explainability, ethics and clinical responsibility (5%)

### Part A · Explainability, ethics and clinical responsibility

- **Most frequent mistake.** Across the three seeds, the scratch CNN most often misclassified neutrophils as immature granulocytes, doing so in 57 of 1,995 neutrophil predictions (2.86%) without augmentation and 74 of 1,995 (3.71%) with augmentation.
- **Who could be harmed and who is missing.** Patients could be harmed if an incorrect cell label influenced their care, and BloodMNIST lacks patient demographic metadata, so we cannot determine which patient groups are underrepresented.
- **Who decides.** We define the scratch CNN as a research and teaching tool, and qualified laboratory staff and clinicians remain responsible for interpreting blood smears.


### Part B

We used Grad-CAM to inspect image regions associated with ResNet-18 predictions. The heatmaps help identify where the model focused. Further evidence would be needed to establish that these regions correspond to biologically meaningful features.

Both parts used a public research dataset. Misclassification could affect clinical interpretation, so any prospective clinical workflow would require specialist review and independent evaluation across patient populations, imaging systems and staining protocols. We did not validate either model for clinical use.

## 7 · Use of AI tools (5%)

For Part A, ChatGPT helped troubleshoot implementation and interpret evaluation outputs. We checked CUDA execution, the CNN output shape, the duplicate audit and the saved results against actual notebook outputs.

For Part B, ChatGPT helped explain ResNet-18 and PyTorch concepts, troubleshoot errors and structure the report. We inspected tensor shapes, training and validation outputs, checkpoints and evaluation results. We also examined Grad-CAM outputs, including a random-weight sanity check. The reported metrics come from the notebook outputs.

## 8 · Conclusion (documentation and communication, 5%)

In Part A, random horizontal and vertical flips reduced test performance under the shared 10-epoch training protocol. The CNN trained without flips achieved a mean test macro F1 of **97.13%**, compared with **95.68%** for the CNN trained with flips. The decrease occurred in all three seed pairs.

In Part B, fine-tuning the pretrained ResNet-18 improved mean test accuracy from **89.03%** with a frozen backbone to **98.80%**. Each result addresses its own experimental comparison. Independent evaluation would be needed to establish generalisation to other clinical datasets and imaging conditions.

## References

1. Acevedo, A., et al. (2020). A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. *Data in Brief*. https://doi.org/10.1016/j.dib.2020.105474
2. Yang, J., et al. (2023). MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*. https://doi.org/10.1038/s41597-022-01721-8
3. He, K., et al. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*.
4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision*.
5. PyTorch documentation. https://pytorch.org/docs/
6. Torchvision documentation. https://pytorch.org/vision/stable/
7. MedMNIST dataset. BloodMNIST 128 × 128, Zenodo record 10519652. https://doi.org/10.5281/zenodo.10519652
