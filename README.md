# Eight-Class Blood Cell Classification from Microscopy Images Using a Scratch CNN and Pretrained ResNet-18

**Biomedical Image Analysis · M.Sc. Applied Data Science and AI · SRH University Munich · Winter semester 2026/27**
**Author(s):** Ishika Thakur , Sushma  · **Repository:** https://github.com/Sushmanayak2000/Biomedical_Image_Analysis · **Date:** [date]

> How to use this template: copy it into your repository as `report.md` (or into LaTeX or Word with the same sections).
> Replace everything in [square brackets]. The section titles follow the eight grading criteria of Session 1 (weights in brackets).
> The remaining criterion, code understanding and technical defense (15 %), is graded in the exam conversation, not in the report.
> Same structure as the Deep Learning report, plus the biomedical parts: modality, labels, patient-level split, clinical meaning.
> Shared data or paper with Deep Learning? Allowed (decided 15.10.): different questions; data preparation is graded here only; name the shared work in both reports.

---

## Abstract

[Four or five sentences: the clinical question, the data, your classical baseline, your deep-learning model, the result with its spread, what it means and what it does not.]

## 1 · The biomedical problem (15 %, with section 2)


- **Clinical question:** How effectively can CNN-based models, including a CNN trained from scratch and an ImageNet-pretrained ResNet-18, classify eight types of blood cells from microscopy images?
- **Task type:** Multiclass image classification. The input is a 128 × 128 RGB microscopy image of an individual blood cell. The output is a predicted label corresponding to one of eight blood cell types.
- **Why it is hard:** Some blood cell types have visually similar characteristics. Distinguishing them may require identifying subtle differences in nucleus shape, staining patterns, and cellular texture. Differences in the number of images available for each class may also affect model performance, meaning that overall accuracy alone may not adequately represent classification performance across all eight classes.



## 2 · The data (15 %, with section 1)

| Item | Description |
|---|---|
| Source, version, download date | BloodMNIST 128×128 dataset, downloaded directly from Zenodo (Record 10519652). Dataset file: bloodmnist_128.npz. Source: https://zenodo.org/records/10519652, Download date: [12/11/2026].|
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0). The dataset can be shared, modified and used commercially, provided the original creators are credited and changes are acknowledged.|
| Modality | Two-dimensional RGB microscopy images of individual blood cells. |
| Size |17,092 images, each with a resolution of 128 × 128 pixels, belonging to 8 classes. Patient count is not reported in the dataset documentation. |
| Labels: who made them | The blood cell images were labelled by clinical pathologists at the Hospital Clínic of Barcelona (Acevedo et al., 2020).|
| Labels: agreement | Not reported. |
| Classes or targets | Eight blood cell types: Basophil, Eosinophil, Erythroblast, Immature Granulocytes, Lymphocyte, Monocyte, Neutrophil and Platelet. |
| Splits | The official training (11,959 images), validation (1,712 images) and test (3,421 images) splits provided with BloodMNIST were used. Patient-level separation could not be independently verified because patient IDs were unavailable. |
| Preprocessing | Original 128 × 128 RGB images were resized to 224 × 224 pixels and normalised using ImageNet mean and standard deviation. Training images were augmented using horizontal and vertical flips and random rotation (±15°). |
| Quality checks | [Summarize the visual review of 20 images and the duplicate checks. Report what you found.] |
| Shared with Deep Learning? | No. The Deep Learning project used EuroSAT satellite images. |

![Example microscopy images of the eight BloodMNIST classes](results/A1_examples.png)

*Figure 1. Representative 128 × 128 RGB microscopy images from BloodMNIST. The label above each image identifies the cell type.*

## 3 · Experimental design (15 %)
 ### Part B
- **The floor:** The largest-class prediction baseline was not separately evaluated in Part B.
- **Classical baseline:** No classical machine-learning model was implemented in Part B.
- **Deep-learning model:** ResNet-18 with ImageNet-pretrained weights was used for eight-class blood cell classification. Two configurations were compared: a frozen pretrained backbone, training only the final classification layer, and a fine-tuned model, training all network layers. Cross-entropy loss was used.
- **Augmentation:** Training images were augmented using random horizontal flips, vertical flips and rotations of up to 15 degrees to introduce variations in cell orientation. Images were resized to 224 × 224 pixels and normalised using ImageNet mean and standard deviation.
- **Comparisons:** The frozen and fine-tuned configurations were trained using the same official dataset splits, for 10 epochs each, with a batch size of 64 and three random seeds (42, 43 and 44). The learning rates were 0.001 for the frozen model and 0.0001 for the fine-tuned model. Performance was compared using accuracy, macro precision, macro recall and macro F1-score.

## 4 · Implementation and reproducibility (25 %)
### Part B
- **Run it:** The experiments were implemented in the Jupyter Notebook Part_B_ResNet18_BloodMNIST.ipynb using Python and PyTorch. The notebook was executed sequentially, from dataset loading and preprocessing to model training, evaluation and explainability. Required packages and their versions are documented in requirements.txt.
- **Seeds, hardware, training time:** Experiments were conducted using three random seeds (42, 43 and 44) on an NVIDIA A6000 GPU with 47.4 GB of memory. Each configuration (frozen and fine-tuned ResNet-18) was trained for 10 epochs per seed, using a batch size of 64. Total training time is not reported.
- **Where the results come from:** The notebook Part_B_ResNet18_BloodMNIST.ipynb generates the training histories, validation and test metrics, classification reports, confusion matrices, bootstrap confidence intervals, Grad-CAM visualisations and prediction examples. Model checkpoints and experiment histories are saved in the results/ directory.

## 5 · Evaluation, validation and critical analysis (15 %)
### Part B
| Setting | [metric] mean | ± spread | seeds |
|---|---|---|---|
| Frozen ResNet-18|89.03% | 0.27 pp| 42, 43, 44|
| classical baseline |98.80% | 0.18 pp| 42, 43, 44|


- **Metrics that fit the task:** Accuracy, macro precision, macro recall and macro F1-score were used to evaluate the eight-class classification task. Macro-averaged metrics give equal importance to each class.
The fine-tuned ResNet-18 achieved a mean test accuracy of 98.80%, macro precision of 98.90%, macro recall of 98.86% and macro F1-score of 98.88% across three seeds.]
- **Worst cases:** The most frequent classification errors occurred between Neutrophils and Immature Granulocytes, indicating that these visually similar cell types were more difficult to distinguish. Some confusion was also observed between Immature Granulocytes, Monocytes and Lymphocytes.]
- **Ceiling:** Inter-observer agreement or a second-annotator performance score was not available. Therefore, the human-level performance ceiling could not be established.
- **Test set:** The official test set containing 3,421 images was used for final evaluation after selecting the best checkpoints based on validation accuracy. No external dataset was used.
- **Limitations:** The experiments were conducted on a single BloodMNIST dataset with predefined splits. Patient-level separation could not be independently verified because patient identifiers were unavailable. The results do not establish generalisation to other hospitals, microscopy systems, staining protocols or patient populations. The model is intended for educational and research purposes, not clinical diagnosis.
  
## 6 · Explainability, ethics and clinical responsibility (5 %)
### Part B
- **Explainability using Grad-CAM:** Grad-CAM was used to highlight the image regions that influenced the ResNet-18 predictions. It helps us understand where the model focused during classification. However, heatmaps are only visual explanations; they cannot prove that the model learned the correct biological features or made a reliable prediction.
- **Ethics and clinical responsibility:** Incorrect classification could lead to misleading interpretations and potentially affect patient care. Therefore, doctors and laboratory specialists must verify the results by examining cell morphology, image quality and relevant clinical information. The model should not replace professional medical judgement.
- **Privacy and limitations:** This project used a publicly available BloodMNIST dataset for educational and research purposes. The model has not been validated for clinical use. Further testing on independent datasets and different imaging conditions would be required before real-world application.
## 7 · Use of AI tools (5 %)

ChatGPT was used as an assistance tool to understand ResNet-18, explain PyTorch concepts, troubleshoot coding errors and support the interpretation of model evaluation results. It also helped with structuring and improving the report text.

AI-generated code suggestions and explanations were not accepted without verification. The notebook was executed from a clean kernel, cell by cell, to check that the implementation ran correctly. Tensor shapes, training and validation outputs, saved model checkpoints and evaluation results were inspected. The frozen and fine-tuned experiments were repeated using three random seeds (42, 43 and 44). The reported metrics were taken from the actual notebook outputs. Grad-CAM visualisations were also examined, including a random-weight sanity check, to assess the reliability of the explanations.

## 8 · Conclusion (documentation and communication, 5 %)
The experiments showed that fine-tuning the pretrained ResNet-18 significantly improved blood cell classification compared with using a frozen backbone. The fine-tuned model achieved a mean test accuracy of 98.80%, compared with 89.03% for the frozen model across three random seeds. The consistent results suggest that fine-tuning was beneficial for this dataset. However, further evaluation using independent datasets and different imaging conditions is needed to assess generalisation. Future work could include testing additional architectures and improving model explainability.

## References

1. Acevedo, A., et al. (2020). A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. Data in Brief. https://doi.org/10.1016/j.dib.2020.105474

2. Yang, J., et al. (2023). MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data. https://doi.org/10.1038/s41597-022-01721-8

3. He, K., et al. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. IEEE International Conference on Computer Vision.

5. PyTorch Documentation. https://pytorch.org/docs/

6. Torchvision Documentation. https://pytorch.org/vision/stable/

7. MedMNIST Dataset. BloodMNIST 128×128, Zenodo record 10519652. https://doi.org/10.5281/zenodo.10519652
