# OPSSAT-AD Reproducibility Package

**Study:** Lightweight Machine Learning for CubeSat Telemetry Anomaly Detection: A Reproducible Comparison of Handcrafted and Raw-Signal Representations on OPSSAT-AD

**Author:** Sanchita Islam  
**Affiliation:** Heritage Institute of Technology, Kolkata, India  
**Contact:** sanchitaislam1@gmail.com

## 1. What this package contains

This repository contains the study-specific analysis scripts used to reproduce the main tabular experiments, feature ablation, feature importance, channel-shift analysis, raw-derived feature screening, and publication figures reported in the manuscript.

The official OPSSAT-AD benchmark code and data-processing notebook are maintained by KP Labs:
https://github.com/kplabs-pl/OPS-SAT-AD

The official dataset release provides `segments.csv`, `dataset.csv`, the dataset-generation notebook, modeling examples, requirements, and license information:
https://doi.org/10.5281/zenodo.15108715

## 2. Important reproducibility note

The tabular experiments are specified directly from the final manuscript and authoritative result tables.

The original raw-CNN/fusion training run was performed during the research workflow, but no trained checkpoint was retained as a portable artifact. Therefore, `src/07_raw_cnn.py` is a clean reimplementation of the raw-sequence experiment described in the manuscript; it should **not** be presented as a byte-for-byte reconstruction of the original training run. The manuscript's reported raw-CNN/fusion test metrics are preserved in the supplied result table.

This distinction is intentional and prevents overclaiming reproducibility.

## 3. Directory structure

```
OPSSAT_AD_reproducibility_package/
├── README.md
├── requirements.txt
├── LICENSE_NOTE.txt
├── src/
│   ├── common.py
│   ├── 01_data_audit.py
│   ├── 02_model_comparison.py
│   ├── 03_feature_ablation.py
│   ├── 04_feature_importance.py
│   ├── 05_channel_generalization.py
│   ├── 06_expanded_features.py
│   ├── 07_raw_cnn.py
│   └── 08_generate_figures.py
├── results/
│   ├── authoritative_model_results.csv
│   ├── authoritative_ablation_results.csv
│   └── authoritative_channel_results.csv
└── figures/
    └── publication figures can be regenerated with 08_generate_figures.py
```

## 4. Data

Place the official `segments.csv` in the project root.

The Zenodo release identifies `segments.csv` as the acquired telemetry and `dataset.csv` as the extracted segment-level feature table.

Do not redistribute the dataset inside this code repository unless the applicable license and repository policy permit it. Prefer instructing users to download it from the official Zenodo record.

## 5. Environment

Python 3.10+ is recommended for this study-specific package.

Install:

```bash
python -m pip install -r requirements.txt
```

Then run the scripts in numerical order.

## 6. Main reported benchmark result

Using the official 18 benchmark features and the supplied train/test split:

Random Forest:
- Accuracy: 0.9735
- Precision: 0.9806
- Recall: 0.8938
- F1: 0.9352
- MCC: 0.9202
- ROC-AUC: 0.9875
- PR-AUC: 0.9675

Confusion matrix:
TN=412, FP=4, FN=12, TP=101.

## 7. Test-set discipline

The official test split must not be used for feature selection or model selection.

The expanded-feature screening uses training-only five-fold cross-validation. The official test set is evaluated only after the representation is fixed.

## 8. Citation

Please cite the OPSSAT-AD benchmark paper and dataset when using this package:

Ruszczak, B., Kotowski, K., Evans, D., & Nalepa, J. The OPS-SAT benchmark for detecting anomalies in satellite telemetry. Scientific Data, 2025. DOI: 10.1038/s41597-025-05035-3.

Ruszczak, B., Kotowski, K., Evans, D., & Nalepa, J. OPSSAT-AD - anomaly detection dataset for satellite telemetry. Zenodo, v2. DOI: 10.5281/zenodo.15108715.
