# Dataset Proposal — Assignment 2

## 1. Dataset name / source / version / license

**Dataset name:** Food-101

**Source:** ETH Zürich Computer Vision Laboratory / Food-101 dataset.

**Dataset URL:** https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

**Paper:**  
L. Bossard, M. Guillaumin, and L. Van Gool,  
"Food-101 — Mining Discriminative Components with Random Forests,"  
European Conference on Computer Vision (ECCV), 2014.  
**Paper link:** https://link.springer.com/chapter/10.1007/978-3-319-10599-4_29

**Version:** Food-101 original dataset release.  
The exact dataset loader/version used during implementation will be
recorded in the project configuration for reproducibility.

**License / data rights:**  
The Food-101 images originate from Foodspotting and are not owned by
ETH Zürich. The dataset will be used for academic coursework only.
The original image files will not be redistributed in the project
repository.

Food-101 contains 101 food categories and 101,000 images in total.
Each class contains 750 training images and 250 manually reviewed
test images. The training images were intentionally not fully cleaned
and may contain some noise.

---

## 2. Task, input, output definitions

**Task:** Multi-class image classification.

**Input:**  
An RGB food image.

**Output:**  
One predicted class among 101 food categories.

The model will output a probability distribution over the 101 classes,
and the class with the highest probability will be selected as the
predicted label.

---

## 3. Number of samples & annotation types

Food-101 contains:

- 101 classes
- 101,000 images in total
- 75,750 training images
- 25,250 test images
- 750 training images per class
- 250 test images per class

**Annotation type:**  
Each image has one categorical class label corresponding to one of
the 101 food categories.

There are no bounding-box or pixel-level segmentation annotations
required for this classification task.

---

## 4. Preliminary distribution analysis

Food-101 is class-balanced at the official split level:

- 750 training images per class
- 250 test images per class

The project will perform additional exploratory data analysis on:

1. Number of samples per class.
2. Image width and height.
3. Image aspect ratio.
4. Representative images from each class.
5. Visual differences between food categories.
6. Potentially difficult or visually similar classes.
7. Low-quality or ambiguous training images.

The training data intentionally contains some real-world noise,
including intense colors and occasional incorrect labels.

---

## 5. Data-splitting plan

The official Food-101 test set will remain completely isolated from
training and model selection.

The original training set will be split into training and validation
sets using a stratified split.

Proposed split:

| Split | Number of images |
|---|---:|
| Training | 68,175 |
| Validation | 7,575 |
| Test | 25,250 |

The training and validation sets will be obtained from the original
75,750 training images using a 90/10 stratified split.

The official 25,250-image test set will only be used for final
evaluation.

---

## 6. Split unit (leakage prevention)

The split unit is the **individual image**.

The original Food-101 training/test separation will be preserved.

The official test set will not be used for:

- Model training
- Hyperparameter tuning
- Model selection
- Data augmentation fitting
- Threshold selection

All training and validation preprocessing will be performed without
accessing test images.

A fixed random seed will be recorded to make the train/validation
split reproducible.

---

## 7. Evaluation metric(s)

The primary evaluation metrics are:

### Accuracy

Accuracy measures the proportion of correctly classified images.

### Macro-F1

Macro-F1 calculates the F1-score independently for each class and
then averages the 101 class scores.

Both Accuracy and Macro-F1 will be reported for all main experiments.

Additional analysis will include:

- Confusion matrix
- Per-class F1-score
- Per-class accuracy
- Representative misclassified examples

## 8. Baseline plan

The baseline will be a simple convolutional neural network trained
from scratch.

Proposed baseline structure:

```text
Input image
    ↓
Convolutional blocks
    ↓
Global Average Pooling
    ↓
Fully Connected Layer
    ↓
101-class output
```


## 9. Planned Pretrained Model(s)

We plan to use ResNet-50 pretrained on ImageNet as the main pretrained model.

Two experimental settings are planned:

- **E2 — Frozen backbone:** use the ImageNet-pretrained ResNet-50 backbone as a fixed feature extractor and train only the final classification head for the 101 Food-101 classes.
- **E3 — Fine-tuning:** initialize ResNet-50 with ImageNet-pretrained weights and fine-tune the model on the Food-101 training set.

The comparison between E2 and E3 will be used to study whether adapting the pretrained representation to the Food-101 domain improves classification performance.

The final ResNet-50 configuration, including the number of unfrozen layers and training hyperparameters, will be determined during implementation and documented with the experimental results.


## 10. Compute estimate

The development environment is a laptop equipped with:

- CPU: 12th Gen Intel Core i7-1265U
- RAM: approximately 32 GB
- GPU: Intel Iris Xe Graphics

No NVIDIA CUDA-capable GPU is available on the local machine.

The local machine will be used for dataset preparation, exploratory
data analysis, pipeline development, debugging, and small-scale
experiments.

For full-scale training of ResNet-50 on the Food-101 dataset, a
CUDA-capable GPU may be used if available through an approved cloud
or remote computing environment. The exact hardware and software
configuration used for final experiments will be recorded in the
experiment logs and final report.

Initial development will use a small stratified subset of the
training data to reduce computational cost. After the pipeline has
been verified, experiments will be scaled to the full training set.

Exact training time, memory usage, and inference cost will be measured
during implementation and reported in the final report.


## 11. Subset Selection Rules

No permanent subset of Food-101 will be used for the final experiments.

The proposed final experiments will use the complete dataset split:

- Training: 68,175 images
- Validation: 7,575 images
- Test: 25,250 images

If computational limitations require a smaller subset during early development or debugging, the subset will be used only for development purposes. The subset selection will use a fixed random seed and will be documented.

Final reported results will be obtained using the full proposed training, validation, and test splits.
---


