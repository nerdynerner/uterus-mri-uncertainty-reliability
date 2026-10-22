# Assessing the Reliability of Machine Learning Error Identification Across Uterus MRI Cohorts

Neriah Otu  
Department of Biomedical Engineering, University of Calgary, Calgary, Alberta, Canada

## Overview

Research implementation for a secondary analysis of UT-EndoMRI. The study evaluates whether uncertainty-based review thresholds preserve their performance across acquisition cohorts in automatic uterus segmentation.

## Research objective

Segmentation accuracy measures agreement with reference contours. Uncertainty ranking measures how well a score orders segmentation errors. Fixed-threshold review performance measures the retained workload and residual error when a threshold selected in one cohort is applied elsewhere. Good ranking alone does not establish that a review threshold transfers across cohorts.

The primary endpoint is retained minus overall participant mean-reference 1-Dice. Negative values indicate lower retrospective error among retained cases. This study evaluates technical segmentation performance and does not establish diagnostic accuracy or a clinical acceptance threshold.

## Dataset

[UT-EndoMRI v2](https://doi.org/10.5281/zenodo.15750762) contains pelvic MRI and structure annotations from two acquisition cohorts. The downloaded archive contained 706 readable NIfTI files and 124 source identifiers (51 D1 and 73 D2). The analytical cohort contains 80 cases, comprising 54 D2 and 26 D1 cases.

| Partition | Cohort | Cases |
| --- | --- | ---: |
| Training | D2 | 27 |
| Validation | D2 | 8 |
| Calibration | D2 | 8 |
| Internal test | D2 | 11 |
| External test | D1 | 26 |

The fixed manifests retain de-identified identifiers supplied in the public dataset, relative file paths and integrity hashes. They contain no clinical participant attributes. Thirty-seven source identifiers lacked an admitted T2-family image and uterus-mask grid; four were excluded for cross-identifier image duplication, one for conflicting spatial transforms and two for edge-reaching foreground. Eligibility combined file and geometry checks with representative gross alignment screening. No independent expert reannotation was performed.

## Methods

Image-only preprocessing applies canonical RAS orientation, 1.5 mm in-plane resampling, nonzero intensity clipping at the 1st and 99th percentiles, normalisation and padding. Native-grid reconstruction reverses the image transformation for evaluation.

The MONAI 2D U-Net uses channels 16/32/64/128, two residual units and instance normalisation. Training uses Dice plus cross-entropy loss, AdamW with learning rate 0.001, batch size one and no augmentation. Validation Dice selects checkpoints, with patience 15 and a maximum of 100 epochs.

Five independently initialised models use seeds 61901-61905. Seed 61901 also supplies the deterministic entropy comparator. A separate dropout model uses seed 61911, dropout probability 0.2 and 30 inference passes. Ensemble and dropout uncertainty use mean pairwise mask 1-Dice; the deterministic comparator uses foreground entropy. Native probabilities are thresholded at 0.5.

The calibration 80th-percentile uncertainty thresholds are frozen before test inference, with exact ties retained. Statistical analysis uses 2,000 participant bootstrap resamples, with fixed-policy, two-stage calibration/test and cross-cohort analyses. Secondary endpoints include risk-coverage curves, failure detection, reference sensitivity, fixed review budgets, surface distances, probability calibration and reference disagreement.

## Repository structure

| Path | Contents |
| --- | --- |
| `src/` | Screening, preprocessing, models, training, inference and statistical methods |
| `config/` | Fixed cohort manifest, partitions, eligibility summary and source lineage |
| `results/` | Recorded scores, aggregate analyses, frozen policies and training records |
| `scripts/` | Workspace preparation, result checks and manuscript figure generation |
| `docs/` | Reproduction details, methodological notes and release materials |
| `figures/` | Figures 1-3 and their aggregate plotting tables |

## Computational environment

Recorded versions are Python 3.13.5, PyTorch 2.14.1, MONAI 1.6.1, NumPy 2.5.4, NiBabel 5.4.2 and SciPy 1.18.1. Figure generation uses R 4.5.1 and base graphics. Exact package records are in `environment-record.txt`. Training used Apple MPS; held-out inference used CPU.

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency pins describe the recorded environment. Installation on a fresh machine and numerical equivalence across hardware have not been established.

## Reproducing the analyses

Lightweight checks and aggregate figure generation require no MRI data or model weights.

```sh
python scripts/check_results.py
PYTHONPATH=src python -m unittest discover -s src -p 'test_*.py'
python scripts/make_figures.py --output figure-reproduction
```

These commands verify the saved score-level analysis and redraw figures. They do not reproduce segmentation inference. See [the reproduction workflow](docs/reproduction.md) for data preparation, training, calibration, inference and secondary analysis. Full execution requires the separately obtained dataset, sufficient storage and substantial computation. Trained weights and native prediction arrays are excluded from this repository; selected checkpoint hashes and epochs are recorded.

## Results and outputs

The ensemble retained 8/11 internal cases and 3/26 external cases. Mean-reference Dice was 0.604 internally and 0.330 externally. The fixed-policy effect was -0.107 (95% CI -0.230, 0.000) internally and -0.163 (-0.349, 0.093) externally. Undefined bootstrap draws are counted explicitly when a resample retains no case.

Figures 1-3 show risk-coverage, frozen-policy effects and temperature-scaled probability reliability. Machine-readable aggregate results are in `results/final_analysis.json`. [Methodological notes](docs/methodological-notes.md) explain the implementation lineage and limits of reproduction.

## Data availability

Source images and annotations are available from the [official UT-EndoMRI v2 record](https://doi.org/10.5281/zenodo.15750762) under its non-commercial scientific research terms and citation requirement. This repository contains code, configurations, source identifiers, numerical evaluation scores and aggregate results. It does not redistribute MRI images, annotations, native predictions or trained weights.

Dataset reference: Liang X, Alpuing Radilla LA, Khalaj K, et al. A Multi-Modal Pelvic MRI Dataset for Deep Learning-Based Pelvic Organ Segmentation in Endometriosis. Scientific Data. 2025;12:1292. https://doi.org/10.1038/s41597-025-05623-3

## Citation

Citation metadata is supplied in `CITATION.cff`. Cite Neriah Otu and the software title, repository URL and the commit used. No software DOI has been assigned. 

## License

Original research code is distributed under the MIT License. UT-EndoMRI images and annotations are excluded and remain governed by the dataset provider terms. Numerical study results and dataset identifiers do not change those terms.
