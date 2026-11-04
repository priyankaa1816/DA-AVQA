# DA-AVQA
### Degradation-Aware Animated Video Quality Assessment

## 1. Project Overview

DA-AVQA is a No-Reference Video Quality Assessment (NR-VQA) approach designed to estimate the perceptual quality of animated videos without requiring their original, undistorted versions.

Animated videos contain distinctive characteristics such as flat-colour regions, sharp edges, stylised textures, and animation-specific temporal patterns. These characteristics motivate quality assessment methods that account for the visual properties of animated content.

This project explores a multi-branch approach that combines pretrained visual representations with semantic, motion, and animation-related features to improve quality prediction.

## 2. Research Reference and Baseline

This project is motivated by the following research paper:

**A No-Reference Framework for Animated Video Quality Prediction in Consumer Multimedia Applications**

**Authors:** Ramya Saisree Mogasati, Sunny Ganavdiya, Ajay Kumar Reddy Poreddy, Balasubramanyam Appina, and Ram Bilas Pachori.

**Publication:** IEEE Transactions on Consumer Electronics.

**DOI:** [10.1109/TCE.2026.3671988](https://doi.org/10.1109/TCE.2026.3671988)

The paper proposes RF-AVQA, a reference-free method that combines local structural and textural features with global spatial and temporal quality measurements. It serves as an important research reference and comparison baseline for this project.

DA-AVQA explores a different approach using pretrained feature representations and a multi-branch architecture.

## 3. Proposed Methodology

DA-AVQA uses four feature branches:

- **FAST-VQA:** Extracts video-quality-aware visual features.
- **CLIP:** Captures high-level visual and semantic representations.
- **Motion:** Extracts temporal motion-related statistics.
- **Animation:** Captures animation-specific visual and temporal characteristics using handcrafted features.

The extracted features are processed through branch-specific encoders and combined using gated fusion. Segment-level attention aggregates information across video segments.

A lightweight prediction head estimates a residual quality correction, which is added to the calibrated FAST-VQA quality score to obtain the final prediction.

## 4. Dataset

This project uses the **BeDAMQA (Benchmark Dataset for Animated Multimedia Quality Assessment)** dataset.

It contains pristine animated videos and their distorted versions, along with subjective Mean Opinion Score (MOS) annotations. The dataset includes the following degradation categories:

- Blur
- Frame Freeze
- Frame Hang
- H.264 compression
- H.265 compression

**Dataset availability:** BeDAMQA is not included or redistributed in this repository. It must be obtained separately from the dataset authors or its official distribution source.

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

The MOS spreadsheet should contain the video filename and its corresponding MOS value. Configure the dataset path in the implementation scripts before running the experiments.

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
| `run_full_p1.py` | Main experiment pipeline that coordinates training, source-grouped cross-validation, and final locked-test evaluation. |
| `prepare_data.py` | Prepares the dataset, reads video metadata and MOS annotations, and organises the inputs required for experiments. |
| `extract_features.py` | Extracts and prepares the visual, semantic, motion, and animation-related features used by the model. |
| `train_p1.py` | Defines the P1 experiment configuration and provides the model training entry point. |
| `trainer.py` | Implements the training and validation process, including optimisation, early stopping, and model checkpointing. |
| `blocks.py` | Defines the model architecture components, including feature encoders, gated fusion, residual processing, and segment attention. |
| `common.py` | Provides shared utility functions for data processing, feature handling, and evaluation. |
| `report_p1.py` | Generates experimental result summaries, performance tables, and plots for evaluation and analysis. |

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

Before running the implementation:

1. Obtain and organise the BeDAMQA dataset separately.
2. Configure the dataset paths in the relevant source files.
3. Ensure the required pretrained model weights and feature-extraction resources are available.
4. Install the dependencies listed in `requirements.txt`.

The main experiment pipeline is:

```bash
python src/run_full_p1.py
```

Refer to the source scripts for the available experiment configurations and execution options.

## 8. Evaluation Metrics

The model is evaluated using:

- **SROCC:** Measures agreement in quality ranking.
- **KRCC:** Measures pairwise ranking agreement.
- **PLCC:** Measures linear correlation between predicted and subjective scores.
- **RMSE:** Measures prediction error on the MOS scale.

The evaluation includes source-grouped cross-validation and a separate locked test set containing previously unseen animation sources.

## 9. Project Repository

**GitHub:** https://github.com/priyankaa1816/DA-AVQA

## Project Authors

**A. Priyankaa, M. Kavya, Nithyashree S., R. Pavithra**

Department of Computer Science and Engineering  
Shiv Nadar University Chennai, India