# BloodMNIST Part A — Custom CNN and Data Augmentation

We tested whether random horizontal and vertical flips during training improve eight-class blood cell classification on BloodMNIST. We trained the same custom CNN from scratch with and without augmentation. **Random flips lowered test macro F1 in all three seed pairs, by 1.45 percentage points on average.**

## Dataset and split audit

We used the official BloodMNIST splits at **128 × 128 RGB** and verified the image shapes, labels, and coverage of all eight classes. Before preprocessing, we checked the raw pixels for exact duplicates within and across splits.

| Split | Official images | Retained images | Excluded |
| --- | ---: | ---: | ---: |
| Train | 11,959 | 11,951 | 8 |
| Validation | 1,712 | 1,709 | 3 |
| Test | 3,421 | 3,415 | 6 |

We kept the first occurrence of each identical image in train → validation → test order, using the original row order within each split. One excluded test duplicate had a conflicting label. The retained row kept its original label, and the split manifest records both labels. Validation and test labels had no influence on which training rows we kept.

We verified that zero exact pixel duplicates remained within or across the retained splits. The archive provides image and label arrays, so we could not verify patient or source overlap. We did not test for near-duplicate overlap.

![One retained training image from each of the eight BloodMNIST classes](results/bloodmnist_class_grid.png)

## Experiment design

- **Model** — Three convolutional blocks with 16, 32, and 64 filters. Each block uses a 3 × 3 convolution, batch normalization, ReLU, and 2 × 2 max pooling. Adaptive average pooling to 4 × 4 and a linear layer produce eight logits. The CNN has 31,896 trainable parameters.
- **Paired runs** — We used seeds 42, 43, and 44. Within each pair, we matched the initial weights, training image order at each epoch, architecture, optimizer, loss, batch size, and epoch budget.
- **Training** — We ran 10 epochs per model on a CUDA GPU with a batch size of 64, cross-entropy loss, and AdamW. We used a learning rate of 0.001 and weight decay of 0.0001.
- **Preprocessing** — We computed RGB normalization statistics using only retained training images. We used the same deterministic preprocessing for validation and test in both conditions.
- **Augmentation** — In the augmented condition, we applied horizontal and vertical flips independently to training images, each with probability 0.5.
- **Checkpoint selection** — We chose each run's best checkpoint by validation macro F1, keeping the earlier epoch in a tie. We froze the settings and checkpoint choices before final test evaluation.

### Augmentation preview

The first column shows the original image. The remaining columns show independent draws from the augmentation pipeline.

![Original training images and examples of random flips](results/augmentation_preview.png)

### Learning curves

![Training loss, validation loss, and validation macro F1 for all six runs](results/runs/834896128886ba35/learning_curves.png)

## Test results

Scores are mean ± sample standard deviation across three seeds, expressed as percentages.

| Condition | Accuracy | Macro F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Without augmentation | **97.22 ± 0.52** | **97.13 ± 0.63** | **96.88 ± 0.80** |
| With augmentation | 95.99 ± 1.13 | 95.68 ± 1.37 | 95.27 ± 1.52 |

We calculated each paired change by subtracting the score without augmentation from the score with augmentation. All changes below are in percentage points.

| Seed | Accuracy change | Macro F1 change | Balanced accuracy change |
| --- | ---: | ---: | ---: |
| 42 | −0.26 | −0.27 | −0.14 |
| 43 | −2.90 | −3.46 | −3.93 |
| 44 | −0.53 | −0.62 | −0.77 |

Mean test F1 declined for seven of the eight classes. The largest drops were for monocytes (−3.36 points), basophils (−2.53), immature granulocytes (−2.31), and lymphocytes (−2.05). Platelets gained 0.04 points. Under this 10-epoch training setup, the random flips reduced test accuracy, macro F1, and balanced accuracy in all three seed pairs.

### Test confusion matrices

The top row shows runs without augmentation. The bottom row shows runs with augmentation. Each column corresponds to one paired seed.

![Labeled test confusion matrices for all six runs](results/runs/834896128886ba35/test_confusion_matrices.png)

## Reproduce the experiment

1. Open `BloodMNIST_Part_A.ipynb` in JupyterLab on Compute Data Lab with a CUDA-enabled PyTorch kernel.
2. Run the notebook in order from a fresh kernel. The notebook checks CUDA availability and downloads BloodMNIST at 128 × 128. The exact package versions and GPU details are recorded in `results/run_config.json`.
3. Run the six training runs and review the validation results. The training cell resumes interrupted runs from the last saved epoch and reuses completed runs when the configuration matches.
4. Freeze the experiment settings, run the separate final test evaluation cell, and verify the saved outputs.

## Repository structure

```text
repository/
└── Part A/
    ├── README.md
    ├── BloodMNIST_Part_A.ipynb
    └── results/
        ├── split_manifest.json
        ├── bloodmnist_class_grid.png
        ├── class_distribution.csv
        ├── class_distribution.png
        ├── augmentation_preview.png
        ├── normalization.json
        ├── run_config.json
        └── runs/
            └── 834896128886ba35/
                ├── frozen_experiment.json
                ├── learning_curves.png
                ├── validation_confusion_matrices.png
                ├── test_confusion_matrices.png
                ├── validation_summary.csv
                ├── test_summary.csv
                ├── test_paired_gains.csv
                ├── seed_42/
                │   ├── without_augmentation/
                │   └── with_augmentation/
                ├── seed_43/
                │   ├── without_augmentation/
                │   └── with_augmentation/
                └── seed_44/
                    ├── without_augmentation/
                    └── with_augmentation/
```

## Saved outputs

The notebook writes outputs to `~/BloodMNIST_Part_A_outputs/results/` on Compute Data Lab. Extract the `results/` folder from the export ZIP and place it beside this README to render the figures on GitHub.

| Location | Contents |
| --- | --- |
| `results/` | Split manifest, class counts and percentages, image grid, augmentation preview, normalization statistics, and run configuration |
| `results/runs/834896128886ba35/` | Frozen experiment record, validation and test summaries, per-class metrics, learning curves, labeled confusion matrices, and paired comparison tables |
| `results/runs/834896128886ba35/seed_<seed>/<condition>/` | Best and resume checkpoints, epoch history, validation and test confusion matrix counts, and per-image test predictions with original row IDs |

The notebook verified that all 54 expected output files were present and nonempty.
