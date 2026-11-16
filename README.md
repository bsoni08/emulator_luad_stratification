# Emulator for cancer cell state proportions in LUAD bulk RNA-seq

Code for:

> Soni B. et al. *A deep learning emulator for single-cell informed deconvolution and state prediction of cancer cells in clinical cohorts.* Preprint: https://doi.org/10.21203/rs.3.rs-11726977/v1

The emulator is a signature-guided deep neural network that estimates the proportions of four lung adenocarcinoma (LUAD) cancer cell states (Groups 0–3) from bulk RNA-seq. It is trained on synthetic bulk samples built from a LUAD single-cell atlas to reproduce marker-based deconvolution (TOAST), and gives each sample an estimate that does not depend on the other samples analyzed.

## Repository structure

| Folder | Content |
|---|---|
| `01_simulation/` | Simulation of bulk samples from the single-cell atlas and TOAST targets |
| `02_training/` | Emulator training: grid search, 10 independent runs (640 fitting / 160 validation / 200 test samples), ensemble evaluation |
| `03_prediction/` | Prediction of Group 0–3 proportions in clinical bulk RNA-seq cohorts |
| `04_analysis/` | Benchmark, survival analysis and figures |
| `models/` | Trained models and scaler |
| `data/` | Input files (not included, see below) |

The code in `02_training/` is available now; the remaining folders will be filled in shortly.

## Training

```bash
python 02_training/emulator_training_valsplit.py
```

Expected inputs in `data/` are described in `data/README.md`. Models, the scaler, test-set predictions, per-run metrics and figures are written to `models/`.

## Requirements

Python 3 with tensorflow, scikeras, scikit-learn, numpy, pandas, scipy, joblib, matplotlib and seaborn (see `requirements.txt`).

## Data availability

- **Single-cell data:** public; sources are listed in the paper.
- **Clinical trial data** (IMpower110, IMpower130, IMpower150, OAK, POPLAR, BIRCH): not included. Patient-level data cannot be made public because of patient privacy; qualified researchers can request them through Vivli (https://vivli.org).

## Contact

Please open an issue in this repository.
