# Underwater Object Classification with CLIP

Classifying man-made objects found underwater (cutlery, tools, bottles, robots and so on) using the Common Objects Underwater (COU) dataset. The main method is a small trained classifier on top of frozen CLIP features. It is compared with a fine-tuned ResNet18, zero-shot CLIP and a linear layer on CLIP features.

Author: [Urvi Damani] | Course: [Computer Vision and Multimodel Machine Learning] | Date: November 2026

## 1. Task

COU was built for detection and segmentation, so one photo can hold several objects. For classification we cut every annotated object out of its photo and treat each cut-out as one image with one label. All methods use the same cut-outs and the same cross-validation folds.

## 2. Dataset

- **Source:** COU: Common Objects Underwater, Mukherjee, Singh, McWilliams and Sattar (2025), University of Minnesota. DOI: https://doi.org/10.13020/1vwe-2707. Licence: CC BY-SA 4.0.
- **Download:** the YOLO version (`YOLO.zip`) from https://conservancy.umn.edu/handle/11299/270146.
- **Content:** about 10,000 underwater photos from pools, lakes and the ocean, with 24 listed classes.
- **Classes used:** 22. "Unknown Instance" is not a real object, so it is dropped. "Metal Rod" is listed but has no annotations in this download.
- **Crops:** 14,872 crops in total (official split: 10,383 train, 3,056 validation, 1,433 test). Objects smaller than 32 pixels are skipped, and each box is padded by 10%.
- **Class balance:** very uneven, from 1,821 crops (Lure) down to 21 (Plastic Bag). Pipe (35) and Plastic Bag (21) have too few crops to train on, so the trained models use the other **20 classes**. Zero-shot CLIP is also reported on all 22.
- **Single modality:** images only. The text side of CLIP comes from class-name prompts.

See `results/class_counts.png`, `results/sample_crops.png` and `results/crop_shapes.png` for the dataset figures.

### Why 3-fold grouped cross-validation instead of the official split

Frames come from videos, so neighbouring frames are almost identical. In the official split, frames from the same video appear in train and test, which can make test scores too high. The folds here are built with `StratifiedGroupKFold`, using groups of 25 consecutive frames from one source, so near-duplicates stay on the same side. Class proportions are kept as far as possible.

## 3. Methods

| Method | What it is | Input handling |
|---|---|---|
| ResNet18 (benchmark) | ImageNet-pretrained, all layers fine-tuned for 5 epochs | resized to 224 x 224 (squashed) |
| ResNet18, padded (benchmark) | same as above | padded to a square with black borders, then resized |
| CLIP zero-shot (benchmark) | ViT-B/32 (OpenAI weights), compares image features with text features of "an underwater photo of a ..." | padded to a square |
| CLIP + linear layer (benchmark) | one linear layer trained on frozen CLIP image features | padded to a square |
| CLIP adapter (chosen method) | frozen CLIP image features, a small residual adapter (512 -> 128 -> 512), and a cosine classifier whose weights start as the CLIP text embeddings of the class prompts | padded to a square |

All trained models use a class-weighted loss (weights proportional to count^-0.5) to soften the class imbalance. Settings (epochs, learning rate, hidden size) were fixed in advance and were not tuned on the test folds.

## 4. Results

20 classes, 3 folds, mean +- standard deviation across folds.

| Method | Accuracy | Macro-F1 |
|---|---|---|
| CLIP adapter | 0.907 +- 0.010 | 0.896 +- 0.011 |
| CLIP + linear layer | 0.886 +- 0.018 | 0.866 +- 0.018 |
| ResNet18, padded | 0.880 +- 0.042 | 0.869 +- 0.056 |
| ResNet18, squashed | 0.871 +- 0.031 | 0.841 +- 0.028 |
| CLIP zero-shot (underwater prompt) | 0.446 | 0.330 |
| CLIP zero-shot (plain prompt) | 0.379 | 0.310 |

Zero-shot CLIP is not trained, so it has no fold spread. It is scored once over all 20-class crops.

### Padding versus CLIP's default centre crop

CLIP's default preprocessing resizes the short side to 224 and keeps only the centre square, which cuts off the sides of long thin objects. The first runs used that default. After seeing the results, the pipeline was switched to padding. These earlier numbers are from that first run and cannot be reproduced from the saved files, because the padded runs overwrote them. To reproduce them, remove `pad_square` from `clip_zeroshot.py` and rerun the three CLIP scripts.

| Method | Centre crop (accuracy / macro-F1) | Padded (accuracy / macro-F1) |
|---|---|---|
| CLIP zero-shot | 0.361 / 0.267 | 0.446 / 0.330 |
| CLIP + linear layer | 0.846 / 0.824 | 0.886 / 0.866 |
| CLIP adapter | 0.877 / 0.860 | 0.907 / 0.896 |

The choice of padding was made after seeing the centre-crop results, so treat it as a design decision made with the test folds in view.

### Adapter ablation

Run with `src/adapter_ablation.py` (same folds, different random seed per fold, so the last row differs slightly from the main adapter row above):

| Variant | Accuracy | Macro-F1 |
|---|---|---|
| Cosine classifier, random init | 0.911 +- 0.011 | 0.903 +- 0.011 |
| Cosine classifier, text init | 0.910 +- 0.010 | 0.904 +- 0.011 |
| Adapter + random init | 0.906 +- 0.011 | 0.894 +- 0.012 |
| Adapter + text init (full) | 0.910 +- 0.012 | 0.903 +- 0.011 |

All four variants are within about one point of each other, which is smaller than the fold-to-fold spread. In this setup the adapter and the text initialisation add nothing measurable. The gain over the plain linear layer comes from the cosine classifier trained on frozen CLIP features. The cause was not isolated further.

### Significance (McNemar's exact test)

Counts are crops where only the first model is right versus only the second.

| Comparison | Only first right | Only second right | p |
|---|---|---|---|
| ResNet18 vs CLIP adapter | 634 | 1,166 | 1.9e-36 |
| ResNet18 (padded) vs CLIP adapter | 660 | 1,059 | 5.3e-22 |
| CLIP linear vs CLIP adapter | 422 | 731 | 7.2e-20 |
| ResNet18 vs CLIP linear | 861 | 1,084 | 4.7e-7 |
| ResNet18 (padded) vs CLIP linear | 906 | 996 | 0.041 |
| ResNet18 vs ResNet18 (padded) | 875 | 1,008 | 0.0023 |

The test treats every crop as independent, but neighbouring video frames are near-duplicates. The p-values are therefore overconfident. The direction of each difference is more trustworthy than the exact values.

### Hardest classes

Fork, Knife and Spoon are the weakest classes for every trained model (F1 about 0.65 to 0.75 for the adapter). The confusion matrices (`results/resnet18_confusion.png`, `results/clip_adapter_confusion.png`) show them being mixed up with each other and with Screwdriver and Scissors.

## 5. Complexity

| Method | Trainable weights | Training + evaluation time (3 folds) | Peak GPU memory |
|---|---|---|---|
| ResNet18, squashed | 11,186,772 | 655.3 s | 1,001 MB |
| ResNet18, padded | 11,186,772 | 792.5 s | 1,001 MB |
| CLIP adapter | 141,952 | 30.7 s | 84 MB |
| CLIP + linear layer | 10,260 | not timed | not measured |
| CLIP zero-shot | 0 | none | not measured |

The CLIP image encoder (87,849,216 weights; the whole CLIP model has 151,277,313) is frozen. Its features are computed once and saved to `results/clip_img_feats.npy`, so the adapter and linear layer train on cached 512-number vectors. That is why they are fast. **The time to compute those features is not included** in the CLIP rows above and was not measured. The cost of this design is that no image augmentation is possible for the CLIP models.

Hardware: NVIDIA GeForce RTX 4050 Laptop GPU.

## 6. How to run

Developed and run with **Python 3.14.4** on Windows. Other Python versions were not tested.

```
# 1. create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1

# 2. install PyTorch with GPU support first, then the rest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`requirements.txt` was created with `pip freeze`, so the `torch` lines may carry a CUDA suffix. If installation fails, install PyTorch as shown above and remove the `torch` and `torchvision` lines from the file.

Place the downloaded dataset so the folder `data/YOLO/` contains `images/`, `labels/` and `obj.names`. Run every script from the repository root, in this order:

| Step | Command | What it does |
|---|---|---|
| 1 | `python src/prepare_data.py` | cuts each object out of its photo into `data/train`, `data/val`, `data/test` (class folders) and writes `data/manifest.csv` |
| 2 | `python src/make_folds.py` | adds a fold number to each crop (3 folds, grouped by frame block), writes `data/manifest_folds.csv` |
| 3 | `python src/make_figures.py` and `python src/crop_shapes.py` | dataset figures |
| 4 | `python src/train_resnet.py` | ResNet18 with squashed input |
| 5 | `python src/train_resnet_padded.py` | ResNet18 with padded input |
| 6 | `python src/clip_zeroshot.py` | CLIP zero-shot; also saves `results/clip_img_feats.npy` (needed by steps 7 to 9) |
| 7 | `python src/clip_linear.py` | CLIP + linear layer |
| 8 | `python src/clip_adapter.py` | CLIP adapter |
| 9 | `python src/adapter_ablation.py` | adapter ablation |
| 10 | `python src/clip_significancetest.py` | McNemar tests on the saved predictions |
| 11 | `python src/confusion.py` | confusion matrix and top mistakes (edit the file name inside to choose the model) |
| 12 | `python src/complexity.py` | parameter counts |

The first run of the ResNet and CLIP scripts downloads pretrained weights, so an internet connection is needed. All scripts write their outputs to `results/`.

**Note on the original run.** `prepare_data.py` was edited for readability after the crops and folds were produced, and the crops were not regenerated afterwards. The manifest paths were then rewritten to be relative to the repository. The fold assignments and results come from the original run. Rerunning the full pipeline should give the same crops and folds (seeds are fixed), but GPU training can differ in the last digit.

## 7. Repository layout

```
data/            dataset, crops and manifests (git-ignored except the two manifests)
notebooks/       notebooks
results/         predictions, reports, figures, ablation table
src/             all scripts (see section 6)
requirements.txt
README.md
```

## 8. Limitations

- **Single-modality dataset.** COU has images only, so the text input to CLIP is just class-name prompts.
- **Unequal preprocessing.** The CLIP models see padded crops, the ResNet is shown in both squashed and padded forms. The ResNet uses random-crop augmentation, which can crop into the black borders. The CLIP models use cached features with no augmentation.
- **Preprocessing chosen after seeing results.** Padding was adopted after the centre-crop results were known.
- **Near-duplicate frames.** Grouping by 25-frame blocks reduces leakage but cannot remove it. Test crops from the same video are correlated, so the effective test size is smaller than the crop count, and McNemar p-values are overconfident.
- **Rare classes.** Pipe and Plastic Bag are not used in training. Case, Dive Weight and Aqua have only about 20 to 35 crops per fold, so their scores are rough.
- **Background cues.** Some classes come from one place (every Lure is from one lake). A model may use the background. This was not tested.
- **No tuning and one seed.** Hyperparameters were fixed in advance and results come from one seed per method. With 3 folds, small gaps are within the fold-to-fold spread.
- **Class names.** Names such as "LoCo" and "ROV" mean nothing to CLIP's text side, which is why zero-shot scores are low for them. Richer class descriptions were not tested.
- **Not implemented from scratch.** ResNet18 and CLIP are pretrained models used through torchvision and open_clip. The cosine classifier, adapter and training loops are our own code.
- **Timing.** CLIP feature extraction time was not measured.

## 9. References

- Mukherjee, R., Singh, S., McWilliams, J. and Sattar, J. (2025). The Common Objects Underwater (COU) dataset for robust underwater object detection. arXiv:2502.20651.
- Radford, A. et al. (2021). Learning transferable visual models from natural language supervision. ICML.
- He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep residual learning for image recognition. CVPR.
- Dataset citation: Mukherjee, R., Singh, S., McWilliams, J. and Sattar, J. (2025). COU: Common Objects Underwater. Data Repository for the University of Minnesota. https://doi.org/10.13020/1vwe-2707.