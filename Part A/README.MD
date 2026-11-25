# BloodMNIST Part A — Custom CNN and Data Augmentation

We tested whether random horizontal and vertical flips improve eight-class blood cell classification on BloodMNIST. We trained the same custom CNN from scratch with and without augmentation using three paired seeds and a 20-epoch training budget. Flips lowered mean test macro F1 by **0.35 percentage points**, from **97.44%** to **97.09%**.

## Dataset and split audit

We used the official BloodMNIST splits at **128 × 128 RGB**. We verified the image shapes, labels, and coverage of all eight classes, then audited raw pixels for exact duplicates within and across splits.

| Split | Official images | Retained images | Excluded |
| --- | ---: | ---: | ---: |
| Train | 11,959 | 11,951 | 8 |
| Validation | 1,712 | 1,709 | 3 |
| Test | 3,421 | 3,415 | 6 |

We kept the first occurrence of each identical image in train → validation → test order, using the original row order within each split. One excluded test duplicate had a conflicting label. The retained image kept its original label, and the split manifest records both labels. Validation and test labels had no influence on which training rows we kept.

The retained splits contained zero exact pixel duplicates. The audit covered exact pixel matches. The archive provides images and labels, so patient and source overlap could not be verified.

### Example images

![One retained training image from each BloodMNIST class](results%204/bloodmnist_class_grid.png)

### Class distribution

![Class distribution across the retained training, validation, and test splits](results%204/class_distribution.png)

## Experiment design

| Component | Setup |
| --- | --- |
| Model | Custom CNN with three convolutional blocks using 16, 32, and 64 filters. Each block uses a 3 × 3 convolution, batch normalization, ReLU, and 2 × 2 max pooling. Adaptive average pooling to 4 × 4 and a linear layer produce eight logits. The model has 31,896 trainable parameters. |
| Paired runs | Seeds 42, 43, and 44. Within each pair, both conditions used the same initial weights, training image order at each epoch, architecture, optimizer, loss, batch size, and epoch budget. |
| Training | 20 epochs per run on CUDA, batch size 64, AdamW with learning rate 0.001 and weight decay 0.0001, and cross-entropy loss. |
| Preprocessing | RGB normalization statistics were computed from retained training images only. Both conditions used the same deterministic preprocessing for validation and test. |
| Augmentation | The augmented training run applied horizontal and vertical flips independently, each with probability 0.5. |
| Checkpoint selection | We selected each run's best checkpoint by validation macro F1, using the earlier epoch to break ties. We froze the experiment settings and checkpoint choices before held-out test evaluation. |

### Augmentation preview

The first column shows the original image. The other columns show independent draws from the training augmentation pipeline.

![Original training images and examples of random flips](results%204/augmentation_preview.png)

## Training and validation results

### Learning curves

The curves show training loss, validation loss, and validation macro F1 over 20 epochs for all six runs. Every run selected its best checkpoint after epoch 10, with best epochs ranging from 13 to 20.

![Training and validation learning curves for all six runs](results%204/runs/404efcfe5a8be877/learning_curves.png)

### Validation confusion matrices

The matrices show which blood cell classes the selected checkpoints confused on the validation split.

![Validation confusion matrices for all six runs](results%204/runs/404efcfe5a8be877/validation_confusion_matrices.png)

## Held-out test results

Scores are mean ± sample standard deviation across the three seeds, expressed as percentages.

| Condition | Accuracy | Macro F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Without augmentation | **97.44 ± 0.17** | **97.44 ± 0.23** | **97.48 ± 0.20** |
| With augmentation | 97.08 ± 0.47 | 97.09 ± 0.40 | 97.07 ± 0.42 |

### Paired changes by seed

Each change is the score with augmentation minus the score without augmentation, in percentage points.

| Seed | Accuracy change | Macro F1 change | Balanced accuracy change |
| --- | ---: | ---: | ---: |
| 42 | −1.02 | −0.96 | −1.10 |
| 43 | +0.18 | +0.25 | −0.05 |
| 44 | −0.23 | −0.36 | −0.08 |

Flips improved test macro F1 for one seed and reduced it for two. The run without augmentation had the higher mean on all three test metrics under this 20-epoch setup.

### Test confusion matrices

The top row shows the runs without augmentation, and the bottom row shows the runs with augmentation. Each column represents one paired seed.

![Held-out test confusion matrices for all six runs](results%204/runs/404efcfe5a8be877/test_confusion_matrices.png)

### Immature granulocyte recall

Across 579 test images and three seeds, mean recall was **93.96%** without augmentation and **94.82%** with flips. The paired change was **+0.86 percentage points** (95% bootstrap CI **−0.06 to +1.78**). The interval reflects uncertainty from resampling test images for the three fitted seed pairs.

Across all classes, 27 test images were misclassified by all six runs.

## Reproduce the experiment

1. Open `BloodMNIST_Part_A.ipynb` in JupyterLab on Compute Data Lab with a CUDA-enabled PyTorch kernel. The notebook uses MedMNIST, PyTorch, torchvision, NumPy, pandas, scikit-learn, Matplotlib, and Pillow. The exact package versions and GPU details are recorded in `run_config.json`.
2. Run the notebook in order from a fresh kernel. The CUDA check runs before training, and the dataset cell downloads BloodMNIST at 128 × 128.
3. Train all six runs and review the validation results. The training cell resumes interrupted runs from the last saved epoch and reuses completed runs when the saved configuration matches.
4. Freeze the experiment settings, run final test evaluation, and verify the saved outputs.

## Repository structure

```text
Part A/
├── README.md
├── BloodMNIST_Part_A.ipynb
└── results 4/
    ├── split_manifest.json
    ├── bloodmnist_class_grid.png
    ├── class_distribution.csv
    ├── class_distribution.png
    ├── augmentation_preview.png
    ├── normalization.json
    ├── run_config.json
    └── runs/
        └── 404efcfe5a8be877/
            ├── frozen_experiment.json
            ├── learning_curves.png
            ├── validation_confusion_matrices.png
            ├── test_confusion_matrices.png
            ├── validation_summary.csv
            ├── test_summary.csv
            ├── test_paired_gains.csv
            ├── test_ig_recall_ci.csv
            ├── seed_42/
            ├── seed_43/
            └── seed_44/
```

Each seed folder contains `without_augmentation/` and `with_augmentation/` subfolders with checkpoints, epoch history, confusion matrix counts, and per-image test predictions.

## Saved outputs

The notebook writes to `~/BloodMNIST_Part_A_outputs/results/` on Compute Data Lab. In this GitHub repository, the exported results are stored in [`results 4/`](results%204/).

| Location | Contents |
| --- | --- |
| `results 4/` | Split manifest, class counts and percentages, example images, augmentation preview, normalization statistics, and run configuration |
| `results 4/runs/404efcfe5a8be877/` | Frozen experiment record, validation and test summaries, per-class metrics, learning curves, confusion matrices, and paired comparisons |
| `results 4/runs/404efcfe5a8be877/seed_<seed>/<condition>/` | Best checkpoint, resume checkpoint, epoch history, validation and test confusion matrix counts, and per-image test predictions with original row IDs |

The notebook verified that all 58 expected output files were present and nonempty.
