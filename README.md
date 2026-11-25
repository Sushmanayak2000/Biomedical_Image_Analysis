# Eight-Class Blood Cell Classification from Microscopy Images Using a Scratch CNN and Pretrained ResNet-18

**Biomedical Image Analysis · M.Sc. Applied Data Science and AI · SRH University Munich · Winter semester 2026/27**

**Authors:** Ishika Thakur (Part A) and Sushma (Part B)  
**Repository:** [Biomedical Image Analysis](https://github.com/Sushmanayak2000/Biomedical_Image_Analysis)  
**Date:** 25 November 2026

## Abstract
We investigated approaches to classifying eight blood cell types from RGB microscopy images using the BloodMNIST dataset. In Part A, we trained a custom convolutional neural network (CNN) from scratch for 20 epochs, comparing training with and without random horizontal and vertical flips across three paired seeds. The model achieved a mean test macro F1 of 97.44% ± 0.23 without augmentation and 97.09% ± 0.40 with augmentation,mean paired decrease of 0.35 percentage points. In Part B, we compared an ImageNet-pretrained ResNet-18 with a frozen backbone against a fully fine-tuned model. The fine-tuned model achieved a mean test accuracy of 98.81% ± 0.02, compared with 88.97% ± 0.22 for the frozen model. Overall, fine-tuning achieved the strongest classification performance among the evaluated approaches. These findings apply to the BloodMNIST dataset and do not establish generalisation to independent clinical datasets.

## Repository structure

| File or folder | Purpose |
| --- | --- |
| [`Part A/`](./Part%20A/) | Scratch CNN, paired augmentation experiment and saved evaluation outputs. |
| [`Part A/requirements.txt`](./Part%20A/requirements.txt) | Python dependencies for the Part A notebook. |
| [`Part A/results 4/`](./Part%20A/results%204/) | Final 20-epoch Part A results, plots and configuration files. |
| [`Part_B_ResNet18_BloodMNIST/`](./Part_B_ResNet18_BloodMNIST/) | ResNet-18 transfer learning experiments and evaluation results. |
| [`lab_results/`](./lab_results/) | Instructor-provided lab notebook and outputs from the in-class exercises. |
| [`BloodMNIST_SharedPreps.ipynb`](./BloodMNIST_SharedPreps.ipynb) | Initial dataset exploration and preparation completed jointly by both team members. |

## 1 · The biomedical problem (15%, with Section 2)

- **Clinical question** — How effectively can CNN-based models classify eight types of blood cells from microscopy images?
- **Task** — Multiclass image classification. Each input is an RGB microscopy image of an individual blood cell, and the model predicts one of eight class labels.
- **Why it is hard** — Some cell types differ in subtle features such as nucleus shape, staining patterns and cellular texture. Class frequencies also vary, so overall accuracy can hide errors concentrated in individual classes.
- **Focus class (Immature granulocytes)** — We pay particular attention to immature granulocytes because they can be confused with other granulocyte types, especially neutrophils. We therefore examine their recall separately to understand how reliably each model identifies this class.

## 2 · The data (15%, with Section 1)

| Item | Description |
| --- | --- |
| Source and version | BloodMNIST 128 × 128 from MedMNIST v2. Archive `bloodmnist_128.npz`, Zenodo record 10519652. |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0). |
| Modality | Two-dimensional RGB microscopy images of individual blood cells. |
| Size | 17,092 images across eight classes. |
| Labelling | Clinical pathologists at the Hospital Clínic of Barcelona labelled the original images. Inter-annotator agreement was unavailable. |
| Classes | Basophil, eosinophil, erythroblast, immature granulocytes, lymphocyte, monocyte, neutrophil and platelet. |
| Part A preprocessing | Original 128 × 128 resolution and RGB normalisation statistics computed from retained training images. Validation and test preprocessing was identical in both conditions. |
| Part B preprocessing | Images resized to 224 × 224 with ImageNet normalisation. Training augmentation included horizontal and vertical flips and rotation up to ±15°. |
| Deep Learning project | The separate Deep Learning project used EuroSAT satellite images. |

Both parts started with the official BloodMNIST splits and removed exact pixel duplicates before preprocessing. After duplicate removal, the retained dataset contained 11,951 training images, 1,709 validation images and 3,415 test images for each part.

| Split | Official images | Retained after duplicate removal | Excluded |
| --- | ---: | ---: | ---: |
| Training | 11,959 | 11,951 | 8 |
| Validation | 1,712 | 1,709 | 3 |
| Test | 3,421 | 3,415 | 6 |
| **Total** | **17,092** | **17,075** | **17** |

Part A checked image shapes, label range and class coverage. For pixel-identical images, we kept the first occurrence in train → validation → test order and used the original row order within each split. One excluded test duplicate had a conflicting label. The split manifest records both labels, and the retained row kept its original label. Training row selection was independent of validation and test labels.

The retained Part A splits contained zero exact pixel duplicates within or across splits. The archive provides image and label arrays, so patient and source overlap could not be verified. The template audit reported five near-twins between training and test. Such overlap could make some test images easier to classify. If those findings correspond to five distinct images in the retained Part A test split, the maximum effect on test accuracy would be approximately **0.15 percentage points** (5 of 3,415 images). [Template audit](./lab_results/C5_summary.md)

![One retained training example from each BloodMNIST class](./Part%20A/results%204/bloodmnist_class_grid.png)

*Figure 1. Representative 128 × 128 RGB images from the retained Part A training split.*

Part B used the same retained, duplicate-free training, validation and test splits as Part A. For ResNet-18, images were resized from 128 × 128 to 224 × 224 pixels and normalised using ImageNet mean and standard deviation. Training images were augmented using random horizontal and vertical flips and rotations of up to ±15°. Validation and test images were resized and normalised without random augmentation.

## 3 · Experimental design (15%)

### Part A · Scratch CNN and data augmentation

We tested whether random horizontal and vertical flips improve classification for a custom CNN trained from scratch. The CNN trained without augmentation served as the reference condition.

The model has three convolutional blocks with 16, 32 and 64 filters. Each block uses a 3 × 3 convolution, batch normalisation, ReLU and 2 × 2 max pooling. Adaptive average pooling produces 4 × 4 feature maps, which are flattened and passed to a linear layer with eight outputs. The model has **31,896 trainable parameters**.

We ran both conditions for seeds 42, 43 and 44. Within each pair, we matched initial weights, training image order per epoch, architecture, loss, optimiser, batch size and epoch budget. We used AdamW with a learning rate of 0.001 and weight decay of 0.0001, cross-entropy loss, a batch size of 64 and **20 epochs per run**. In the augmented condition, we applied horizontal and vertical flips independently, each with probability 0.5.

![Original training images and random flip examples](./Part%20A/results%204/augmentation_preview.png)

*Figure 2. The first column shows the original image. The other columns show independent draws from the augmentation pipeline.*

### Part B · Transfer learning with ResNet-18

We investigated whether fine-tuning an ImageNet-pretrained ResNet-18 improves blood cell classification compared with keeping its pretrained feature-extraction layers frozen. Our hypothesis was that fine-tuning would improve classification performance because the network could adapt its learned features to the visual characteristics of blood-cell microscopy images.

We compared two configurations:

- **Frozen ResNet-18:** The pretrained backbone remained frozen, and only the final classification layer was trained to predict the eight BloodMNIST classes.
- **Fine-tuned ResNet-18:** All network layers, including the pretrained backbone and final classification layer, were trainable, allowing the model to adapt its features to the target dataset.

Both configurations used the AdamW optimiser and cross-entropy loss, with a weight decay of 0.0001. The frozen model used a learning rate of 0.001, while the fine-tuned model used a lower learning rate of 0.0001. Each configuration was trained for 10 epochs per run, with a batch size of 64 and three random seeds (42, 43 and 44).

Both configurations used the same duplicate-cleaned BloodMNIST training, validation and test splits. Images were resized to 224 × 224 pixels and normalised using ImageNet mean and standard deviation. Training images were augmented using random horizontal and vertical flips and rotations of up to ±15°. Validation and test images received deterministic preprocessing without random augmentation.

We selected model checkpoints based on validation accuracy and evaluated the selected checkpoints on the held-out test set. We reported accuracy, macro precision, macro recall and macro F1 to assess overall and class-balanced performance.
## 4 · Implementation and reproducibility (25%)

### Part A

- **Notebook and environment** — We ran the Part A notebook in JupyterLab on Compute Data Lab. The runs used an NVIDIA RTX A6000 with PyTorch 2.9.1 and CUDA 12.8. We verified CUDA execution and checked that the CNN produced eight logits per image. Package versions, normalisation statistics and training settings are recorded in the [run configuration](./Part%20A/results%204/run_config.json).
- **Checkpoint selection** — We selected the checkpoint with the highest validation macro F1 for each run, keeping the earlier epoch in a tie. We froze the settings and checkpoint choices before test evaluation.
- **Resume and outputs** — The notebook can resume interrupted runs from the last saved epoch and reuse completed runs with matching settings. It saves epoch histories, checkpoints, the split manifest, evaluation metrics, confusion matrices and test predictions with original row IDs. The final check verified **58 nonempty output files**.

To install the Part A dependencies, run this command from the `Part A/` folder in a Python 3.11 environment configured for NVIDIA CUDA:

~~~bash
python -m pip install -r requirements.txt
~~~

#### Part B

- **Notebook and environment** — We implemented the experiments in `Part_B_ResNet18_BloodMNIST.ipynb` using Python 3.11.13 and PyTorch 2.9.1 with CUDA 12.8. The experiments ran on an NVIDIA RTX A6000 GPU with 47.4 GB of memory.

- **Dependencies** — The Part B `requirements.txt` specifies the package versions used in the experiment, including PyTorch, Torchvision, MedMNIST, NumPy, Pandas and scikit-learn. To install the Part B dependencies, run this command from the `Part_B_ResNet18_BloodMNIST/` folder in a Python 3.11 environment configured for NVIDIA CUDA:

~~~bash
python -m pip install -r requirements.txt
~~~

- **Reproducibility and training configuration** — We controlled random seeds across Python, NumPy and PyTorch, using seeds 42, 43 and 44. Deterministic cuDNN settings were enabled where supported. Both model configurations used the same cleaned dataset splits, with 11,951 training, 1,709 validation and 3,415 test images.

- **Checkpoint selection** — For each run, the model checkpoint with the highest validation accuracy was selected. The test split was reserved for final evaluation and was not used for checkpoint selection.

- **Saved outputs** — The notebook generates training histories, validation and test metrics, classification reports, confusion matrices, bootstrap confidence intervals, Grad-CAM visualisations and prediction examples. Model checkpoints are saved in the `results/checkpoints/` directory.

## 5 · Evaluation, validation and critical analysis (15%)

### Part A

We evaluated the six validation-selected checkpoints on the same **3,415 retained test images**. Macro F1 gives each of the eight classes equal weight. Values below are mean ± sample standard deviation across three seeds.

| Training condition | Test accuracy (%) | Test macro F1 (%) | Test balanced accuracy (%) |
| --- | ---: | ---: | ---: |
| Without augmentation | 97.44 ± 0.17 | 97.44 ± 0.23 | 97.48 ± 0.20 |
| With augmentation | 97.08 ± 0.47 | 97.09 ± 0.40 | 97.07 ± 0.42 |

The paired changes below are the scores with augmentation minus the scores without augmentation, in percentage points (pp).

| Seed | Accuracy change (pp) | Macro F1 change (pp) | Balanced accuracy change (pp) |
| --- | ---: | ---: | ---: |
| 42 | −1.02 | −0.96 | −1.10 |
| 43 | +0.18 | +0.25 | −0.05 |
| 44 | −0.23 | −0.36 | −0.08 |

Test macro F1 improved with flips for one seed and declined for two, with a mean paired change of **−0.35 pp**. Balanced accuracy declined in all three pairs, with a mean paired change of **−0.41 pp**. All six runs selected their best validation checkpoint between epochs 13 and 20. One run selected epoch 20, so the 20-epoch budget still leaves convergence uncertain.

![Learning curves for the six Part A runs](./Part%20A/results%204/runs/404efcfe5a8be877/learning_curves.png)

*Figure 3. Training loss, validation loss and validation macro F1 across the 20 epochs.*

#### Focus class · Immature granulocytes

Mean test F1 declined with flips for six classes, with the largest decreases for monocytes (−1.02 pp) and immature granulocytes (−0.78 pp). We examined immature granulocyte recall separately across the **579** test images in that class.

| Training condition | Mean recall (%) | 95% bootstrap CI |
| --- | ---: | ---: |
| Without augmentation | 93.96 | 92.29–95.51 |
| With augmentation | 94.82 | 93.32–96.26 |
| With minus without (pp) | +0.86 | −0.06 to +1.78 |

Recall improved for two seeds and declined for one. The confidence interval for the paired change includes zero. The bootstrap resamples test images for the three fitted seed pairs.

![Test confusion matrices for the six Part A runs](./Part%20A/results%204/runs/404efcfe5a8be877/test_confusion_matrices.png)

*Figure 4. Columns show seeds 42, 43 and 44. The top row shows runs without augmentation, and the bottom row shows runs with augmentation.*

### Part B · Evaluation, Validation and Critical Analysis

#### Overall performance

We evaluated both ResNet-18 configurations on the same duplicate-cleaned test set of 3,415 images. Results are reported as mean ± standard deviation across three random seeds.

| Model configuration | Accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| Frozen ResNet-18 | 88.97 ± 0.22 | 89.93 ± 0.30 | 87.06 ± 0.88 | 88.11 ± 0.60 |
| Fine-tuned ResNet-18 | 98.81 ± 0.02 | 98.98 ± 0.04 | 98.84 ± 0.01 | 98.91 ± 0.02 |

Fine-tuning improved test accuracy by 9.84 percentage points and macro F1 by 10.80 percentage points. The low standard deviations across the three runs indicate consistent performance. Overall, adapting the pretrained backbone to BloodMNIST substantially improved classification compared with training only the final classification layer.

#### Validation and class-wise analysis

For each run, we selected the checkpoint with the highest validation accuracy and reserved the test set for final evaluation. We used accuracy, macro precision, macro recall and macro F1 to assess overall and class-balanced performance.

The fine-tuned model achieved improved performance across all eight classes. However, the confusion matrices revealed that some visually similar cell types remained challenging. The most frequent error was neutrophils classified as immature granulocytes (51 errors across three seeds), followed by immature granulocytes classified as neutrophils (21 errors). Other recurring errors included monocytes classified as immature granulocytes (11) and erythroblasts classified as immature granulocytes (9).

These errors indicate that the distinction between neutrophils and immature granulocytes remained particularly challenging. The 51 neutrophil-to-immature-granulocyte errors represented approximately 2.56% of pooled neutrophil test predictions across the three runs, while the reverse errors represented approximately 1.21% of pooled immature-granulocyte predictions.

#### Focus-class analysis: Immature granulocytes

We examined immature granulocyte recall separately because of its visual similarity to other granulocyte types.

| Model configuration | Mean recall (%) | 95% bootstrap CI |
| --- | ---: | ---: |
| Frozen ResNet-18 | 84.80 | 81.98–87.56 |
| Fine-tuned ResNet-18 | 98.39 | 97.58–99.08 |

Fine-tuning improved immature granulocyte recall by 13.59 percentage points. This demonstrates a substantial improvement in identifying the focus class, although some confusion with neutrophils remained.

#### Critical analysis and limitations

Despite its strong test performance, the model has several limitations:

- **Dataset generalisation:** BloodMNIST is a benchmark dataset, and performance on it does not guarantee similar results on images from other hospitals, laboratories or microscopy systems. External validation is required.
- **Patient-level independence:** Patient identifiers were unavailable, so we could not verify whether images from the same patient appeared across different dataset splits.
- **Near-duplicate images:** Exact duplicates were removed, but five retained near-twin relationships were identified during the dataset audit. These could potentially influence test performance.
- **Class similarity:** Remaining confusion between neutrophils and immature granulocytes suggests that visually similar cell categories are still challenging to distinguish.
- **Pretraining limitations:** ImageNet features were originally learned from natural images rather than blood-cell microscopy. The improvement after fine-tuning suggests that adapting these features to the target dataset was important.
- **Explainability:** Grad-CAM highlights image regions influencing predictions, but it cannot prove that the model relies on biologically meaningful features.
- **Clinical applicability:** The model has only been evaluated on a benchmark dataset and should not be considered a diagnostic tool. Expert review and independent clinical validation would be necessary before practical use.

Overall, fine-tuning ResNet-18 substantially improved classification performance and immature granulocyte recall. However, remaining class confusion, dataset limitations and the absence of external clinical validation mean that these results demonstrate strong benchmark performance rather than established clinical reliability.

## 6 · Explainability, ethics and clinical responsibility (5%)

### Part A

The most frequent directional error was predicting immature granulocytes for true neutrophils. Across three seeds, it occurred in **54 of 1,995** predictions on true neutrophils (2.71%) without augmentation and **94 of 1,995** (4.71%) with augmentation. Across all classes, **27 test images** were misclassified by all six runs.

![Examples of test images misclassified by all six Part A runs](./Part%20A/results%204/runs/404efcfe5a8be877/test_always_misclassified.png)

*Figure 5. Examples of test images that all six Part A runs classified incorrectly.*

Incorrect cell labels could affect clinical interpretation. BloodMNIST lacks patient demographic metadata, so we cannot determine which patient groups are underrepresented. The scratch CNN is a research and teaching tool. Qualified laboratory staff and clinicians remain responsible for interpreting blood smears.

### Part B

We used Grad-CAM to examine image regions influencing predictions made by the fine-tuned ResNet-18 model. Visualisations were generated for correctly and incorrectly classified test images using the model trained with seed 42. This allowed us to qualitatively inspect the regions associated with the model's predictions and errors.

We also performed a random-weight sanity check by comparing the trained model's heatmap with that of a randomly initialised ResNet-18 model using the same test image. This provided a qualitative comparison of explanations generated with and without learned model parameters. However, Grad-CAM does not establish whether highlighted regions correspond to biologically meaningful features or prove that the model uses clinically relevant information.

Incorrect blood-cell classifications could affect subsequent clinical interpretation. Therefore, the model should be considered a research and educational tool rather than a diagnostic system. Both approaches require further evaluation using independent datasets, different patient populations, imaging systems and staining protocols. Qualified laboratory professionals and clinicians must remain responsible for interpreting blood-cell images.
## 7 · Use of AI tools (5%)

For Part A, ChatGPT helped troubleshoot implementation and interpret evaluation outputs. We checked CUDA execution, the CNN output shape, the duplicate audit and the saved results against actual notebook outputs.

For Part B, ChatGPT helped explain ResNet-18 and PyTorch concepts, troubleshoot errors and structure the report. We inspected tensor shapes, training and validation outputs, checkpoints and evaluation results. We also examined Grad-CAM outputs, including a random-weight sanity check. The reported metrics come from the notebook outputs.

## 8 · Conclusion (documentation and communication, 5%)

In Part A, the scratch CNN trained without augmentation achieved **97.44% mean test accuracy**. Mean test macro F1 was **97.44% without flips** and **97.09% with flips**. Horizontal and vertical flips provided no average macro F1 gain under the shared 20-epoch protocol, and the effect varied across seeds.

In Part B, fine-tuning ImageNet-pretrained ResNet-18 improved mean test accuracy from **89.03%** with a frozen backbone to **98.80%**. The paired Part A comparison isolates the effect of the tested flips on the scratch CNN. The Part B comparison measures the benefit of fine-tuning within its transfer learning setup.

## References

1. Acevedo, A., et al. (2020). A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. *Data in Brief*. https://doi.org/10.1016/j.dib.2020.105474
2. Yang, J., et al. (2023). MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*. https://doi.org/10.1038/s41597-022-01721-8
3. He, K., et al. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*.
4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision*.
5. PyTorch documentation. https://pytorch.org/docs/
6. Torchvision documentation. https://pytorch.org/vision/stable/
7. MedMNIST dataset. BloodMNIST 128 × 128, Zenodo record 10519652. https://doi.org/10.5281/zenodo.10519652
