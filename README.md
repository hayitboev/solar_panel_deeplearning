# Solar Panel Cleanliness Classification

A team deep learning project that classifies solar panel images as **dirty** (`Kirli`) or **clean** (`Temiz`). The repository contains the image dataset, individual experiment notebooks, and model checkpoints.

## Project overview

The goal is to explore whether image classification models can distinguish clean and dirty solar panels. We used transfer learning and compared several convolutional neural network architectures. This repository documents an experimental workflow, not a deployed inspection system.

### Dataset

Images are organized for PyTorch's `ImageFolder`:

```text
panel_data/
├── Kirli/   # dirty
└── Temiz/   # clean
```

The repository currently contains **1,756 dirty** and **1,876 clean** images (3,632 in total). The notebook by Muhammed Hayitboev uses a seeded random split of **2,542 training**, **544 validation**, and **546 test** images. The source, collection method, and reuse permissions of the images are not documented in this repository; verify them before redistributing or using the dataset beyond this project.

## Team and experiments

| Contributor | Notebook | Scope visible in this repository |
| --- | --- | --- |
| Muhammed Hayitboev | [`MUKHAMMAD_KHAITBOEV_22040301090_PATtechs.ipynb`](MUKHAMMAD_KHAITBOEV_22040301090_PATtechs.ipynb) | Dataset preparation and overall approach; ResNet-50 and DenseNet-121 experiments, evaluation, Grad-CAM visualization, and inference timing |
| Abdullah Akbalık | [`ABDULLAH_AKBALIK_22040301102_PATtechs.ipynb`](ABDULLAH_AKBALIK_22040301102_PATtechs.ipynb) | Individual experiment notebook |
| Asfandiyor Farkhodov | [`ASFANDIYOR_FARKHODOV_22040301121_PATtechs.ipynb`](ASFANDIYOR_FARKHODOV_22040301121_PATtechs.ipynb) | Individual experiment notebook |

The [`models/`](models/) directory contains Git LFS tracked checkpoint files named for ResNet-50, DenseNet-121, VGG-16, Inception-v3, EfficientNet-B0, and MobileNetV3. The presence of a checkpoint does not by itself establish a comparable result for that architecture.

## Muhammed's experiment

The notebook resizes images to 224 × 224, applies ImageNet normalization, fine-tunes pretrained ResNet-50 and DenseNet-121 models with a two-class output layer, and evaluates them with classification reports, confusion matrices, and ROC curves. It also includes Grad-CAM visualization and a simple inference timing check.

| Model | Best validation accuracy reported by notebook | Test accuracy reported by notebook | Test images |
| --- | ---: | ---: | ---: |
| ResNet-50 | 90.62% | 91% (rounded in classification report) | 546 |
| DenseNet-121 | 90.99% | 89% (rounded in classification report) | 546 |

These numbers are **saved notebook outputs**, not independently rerun results. The timing outputs in the notebook were measured on an NVIDIA GeForce RTX 4060 Laptop GPU; they are environment-dependent and are not presented as production benchmarks.

### Evaluation caveat

The current notebook creates train, validation, and test subsets with `random_split` from a **single** `ImageFolder` object. It then changes `dataset.transform` through the validation and test subset references. Because those subsets share the underlying dataset, the training transform is also replaced. The augmentation defined for training therefore does not run as intended. A corrected experiment should use separate dataset instances or a subset wrapper with its own transform, rerun training, and report updated metrics. The random image split also does not establish whether near-duplicate images or panels from the same source occur across splits.

## Run the notebook

1. Clone the repository with Git LFS if you need the checkpoints:

   ```bash
   git lfs install
   git clone https://github.com/hayitboev/solar_panel_deeplearning.git
   cd solar_panel_deeplearning
   ```

2. Create a Python environment and install the packages used by Muhammed's notebook:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install jupyter torch torchvision numpy matplotlib scikit-learn seaborn grad-cam
   jupyter notebook
   ```

   On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.

3. Open the notebook you want to inspect or run. Muhammed's notebook expects `panel_data/` in the repository root. Pretrained torchvision weights may download on the first run. Hardware, package versions, and the transform issue above may affect reproduced results.

## Repository layout

```text
├── panel_data/                   # dirty and clean panel images
├── models/                       # Git LFS tracked model checkpoints
├── ABDULLAH_..._PATtechs.ipynb    # Abdullah's experiment
├── ASFANDIYOR_..._PATtechs.ipynb  # Asfandiyor's experiment
├── MUKHAMMAD_..._PATtechs.ipynb   # Muhammed's experiment
├── .gitattributes                # Git LFS rules
└── LICENSE
```

## License

The repository includes an [MIT license](LICENSE) for its software. The image dataset's provenance and permissions are not specified here; do not assume the software license covers third-party images.
