# D-Hierarchical: Hierarchical Graph Neural Network for Multiclass SDN Intrusion Detection

Official PyTorch / PyTorch Geometric implementation and reproducible evaluation pipeline for **D-Hierarchical** under a strictly inductive (`split-before-graph`) multi-seed protocol.

## Overview

- **Architecture (`DHier`):**
  - `GCTLayerResNorm` (×2): Edge-conditioned Graph Convolutional Transformer layers with residual connections and Layer Normalization.
  - `SubnetAggregation`: Hierarchical `/24` IPv4 subnet pooling and broadcast MLP module.
  - `EdgeAwareLSTMResNorm`: Recurrent neighborhood aggregation over incident flows with residual LayerNorm stabilization.
- **Inductive Protocol (`build_cell`):**
  - Stratified train/validation/test partitioning (`64% / 16% / 20%`) is performed **prior** to graph construction.
  - Feature preprocessing (signed `log1p` + `StandardScaler`) and weighted node feature aggregations are fitted **strictly on training flows**.
  - Unseen test endpoints map to an explicit `UNK` (zero) node representation.
- **Validation-Only Logit Calibration (`calibrate`):**
  - Coordinate-ascent per-class additive logit offsets tuned exclusively on the validation split (`raw` vs. `cal`).
- **Baselines & Hybrid:**
  - `Random Forest` and `LightGBM` evaluated on `edge` (flow-only) and `edge+node` feature views.
  - `Hybrid GNN->LGBM` using 2-fold out-of-fold (OOF) cross-fitted GNN class probabilities inside the training split.

## Repository Structure

- `dhierarchical_evaluation.ipynb`: Self-contained Jupyter / Kaggle notebook reproducing all experiments.
- `dhierarchical_evaluation.py`: Standalone Python script version of the pipeline.
- `requirements.txt`: Required Python packages.
- `results/`: Pre-computed multi-seed CSV outputs (`v2_summary.csv`, `v2_runs.csv`, `v2_perclass.csv`, `v2_tests.csv`).

## Datasets

Place the dataset CSV files inside `./data/` (or attach them under `/kaggle/input` on Kaggle):
1. **InSDN:** `Normal_data.csv`, `OVS.csv`, `metasploitable-2.csv`
2. **CICIDS2017 (`TrafficLabelling` with IP columns):** 8 CSV files from the CICIDS2017 TrafficLabelling directory.
3. **UNSW-NB15:** `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`

## Running the Experiments

```bash
pip install -r requirements.txt
DATA_DIR=./data python dhierarchical_evaluation.py
```

For a fast smoke test (subsampled rows, 1 epoch):
```bash
LEAK_QUICK=1 DATA_DIR=./data python dhierarchical_evaluation.py
```
