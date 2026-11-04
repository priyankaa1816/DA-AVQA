# DA-AVQA
### Degradation-Aware Animated Video Quality Assessment

## 1. Project Overview

DA-AVQA is a No-Reference Video Quality Assessment (NR-VQA) approach designed to predict the perceptual quality of animated videos without requiring their original, undistorted versions.

Animated content has distinctive characteristics such as flat-colour regions, sharp edges, stylised textures, and animation-specific temporal patterns. This project explores a multi-branch approach that combines pretrained visual representations with semantic, motion, and animation-related features to improve quality prediction.

## 2. Research Reference and Baseline

This project is motivated by the research paper:

**"A No-Reference Framework for Animated Video Quality Prediction in Consumer Multimedia Applications"**

**Authors:** R. S. Mogasati, S. Ganavdiya, A. K. R. Poreddy, B. Appina, and R. B. Pachori.

**Publication:** IEEE Transactions on Consumer Electronics, 2026.

**DOI:** [10.1109/TCE.2026.3671988](https://doi.org/10.1109/TCE.2026.3671988)

The paper proposes RF-AVQA, a reference-free method that combines structural, textural, spatial, and temporal quality measurements. It serves as an important research reference and published comparison baseline for this project. DA-AVQA explores a different approach based on learned feature representations and multi-branch fusion.

## 3. Proposed Methodology

DA-AVQA uses four complementary feature branches:

- **FAST-VQA:** Extracts video-quality-aware visual features.
- **CLIP:** Captures high-level visual and semantic representations.
- **Motion:** Extracts temporal motion-related statistics.
- **Animation:** Captures animation-specific visual and temporal characteristics using handcrafted features.

The branch features are combined through gated fusion and segment-level attention. A lightweight prediction head estimates a residual quality correction, which is added to the calibrated FAST-VQA score to produce the final quality prediction.

## 4. Dataset

This project uses the **BeDAMQA (Benchmark Dataset for Animated Multimedia Quality Assessment)** dataset. It contains pristine animated videos, distorted versions, and subjective Mean Opinion Score (MOS) annotations.

The dataset includes five degradation categories:

- Blur
- Frame Freeze
- Frame Hang
- H.264 compression
- H.265 compression

**Dataset availability:** BeDAMQA is not included or redistributed in this repository. It must be obtained separately from its authors or official distribution source.

**Dataset reference:**

S. Ganavdiya, B. Appina, and R. B. Pachori, "Descriptor: Benchmark dataset for animated multimedia quality assessment (BeDAMQA)," *IEEE Data Descriptions*, vol. 2, pp. 441–451, 2025.

### Expected Dataset Organization

After obtaining the dataset, arrange it as follows:

```text
DatasetMain/
├── Reference/       # 15 pristine videos (01–15)
├── Blur/
├── FrameFreeze/
├── FrameHang/
├── H264/
├── H265/
└── <MOS spreadsheet>.xlsx
```

The MOS spreadsheet should contain the video filename and its corresponding MOS value. Configure the dataset path in the implementation before running the experiments.

## 5. Repository Structure

```text
DA-AVQA/
├── figures/
│   └── .gitkeep
├── src/
│   ├── run_full_p1.py
│   ├── prepare_data.py
│   ├── extract_features.py
│   ├── train_p1.py
│   ├── trainer.py
│   ├── blocks.py
│   ├── common.py
│   └── report_p1.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Source Code Description

| File | Description |
|---|---|
| `run_full_p1.py` | Coordinates the main experimental pipeline, including cross-validation and locked-test evaluation. |
| `prepare_data.py` | Prepares dataset information, video metadata, and MOS annotations for the experiments. |
| `extract_features.py` | Extracts and prepares the visual, semantic, motion, and animation-related features. |
| `train_p1.py` | Configures and initiates training of the DA-AVQA P1 model. |
| `trainer.py` | Handles model training, validation, optimisation, and early stopping. |
| `blocks.py` | Defines the model components, including feature encoders, gated fusion, residual processing, and segment attention. |
| `common.py` | Provides shared utility functions for data processing and evaluation. |
| `report_p1.py` | Generates performance summaries, evaluation tables, and result visualisations. |

The `figures/` directory is reserved for project-related plots and visualisations.

## 6. Installation

Clone the repository:

```bash
git clone https://github.com/priyankaa1816/DA-AVQA.git
cd DA-AVQA
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 7. Running the Project

Before execution:

1. Obtain and organise the BeDAMQA dataset separately.
2. Configure the dataset paths in the relevant source files.
3. Ensure the required pretrained model weights and feature-extraction resources are available.
4. Install the dependencies listed in `requirements.txt`.

Run the main experiment pipeline using:

```bash
python src/run_full_p1.py
```

Refer to the source scripts for experiment configuration and execution options.

## 8. Evaluation Protocol

The experiments use source-grouped four-fold cross-validation on the development data, with five random seeds, resulting in 20 evaluation runs. All distorted versions of an original animation are kept within the same split to prevent content leakage.

A separate locked test set contains 75 videos from three previously unseen animation sources. Validation data is used for model selection and early stopping, while the locked test set is reserved for final evaluation.

## 9. Implementation Details

The FAST-VQA and CLIP feature extractors are kept frozen, while the lightweight prediction head and fusion components are trained for quality prediction.

The model combines four feature branches through gated fusion and segment-level attention. It uses a residual learning strategy, adding the predicted quality correction to the calibrated FAST-VQA score. Training uses a Huber regression objective.

## 10. Experimental Results

The following table presents the overall results of DA-AVQA against its calibrated FAST-VQA-only baseline.

| Evaluation | Model | SROCC ↑ | PLCC ↑ | RMSE ↓ |
|---|---|---:|---:|---:|
| CV (299 videos) | FAST-only | 0.829 | 0.840 | 0.587 |
| CV (299 videos) | **DA-AVQA** | **0.875** | **0.890** | **0.528** |
| Locked test (75 videos) | FAST-only | 0.878 | 0.882 | 0.593 |
| Locked test (75 videos) | **DA-AVQA** | **0.921** | **0.936** | **0.468** |

## 11. Evaluation Metrics

The following metrics are used to assess prediction performance:

- **SROCC:** Measures agreement in quality ranking.
- **KRCC:** Measures pairwise ranking agreement.
- **PLCC:** Measures linear correlation between predicted and subjective quality scores.
- **RMSE:** Measures prediction error on the MOS scale.

## Project Authors

**A. Priyankaa, M. Kavya, Nithyashree S., R. Pavithra**

Department of Computer Science and Engineering  
Shiv Nadar University Chennai, India
