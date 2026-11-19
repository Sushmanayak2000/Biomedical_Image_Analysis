Question: How effectively can a CNN trained from scratch and an ImageNet-pretrained ResNet-18 classify eight blood cell types in BloodMNIST?
On 3421 test images, the frozen ResNet-18 (balanced) found 238 of 243 lymphocyte images (97.9%; 95 % bootstrap interval 95.9% to 99.6%).
Accuracy 92.9% (interval 92.1% to 93.8%); floor 19.5%; random-forest baseline 93.4%.
Three training seeds: lymphocyte found 96.3% ± 1.9%. Calibration error for lymphocyte: 0.015.
Limits: one data set, test images from the same source (internal validation only); 5 near-twins between training and test; patient not known from the files: cannot check (say so in the report).
