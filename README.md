# TimeMixer_ME

[中文说明](./README.zh-CN.md)

TimeMixer_ME is a multi-task time-series modeling repository built on top of the TimeMixer family. It extends the base architecture with:

- `MTMEM` (Multi-Time Scale Memory Enhancement Module)
- `AnyVariateAttention` for explicit cross-variate interaction modeling

The repository currently provides a unified `run.py` entry for:

- long-term forecasting
- imputation
- anomaly detection
- classification

It also includes:

- forecast explainability scripts based on SHAP and attention extraction
- robust train-time augmentation for noisy or partially missing inputs
- benchmark scripts for several weather forecasting settings

## Current Scope

This repository is primarily optimized around `long_term_forecast`.

The other three tasks are available through the same codebase and have working data/model paths, but they are less benchmark-scripted than forecasting. In particular:

- explainability scripts currently target forecasting checkpoints
- robustness augmentation is currently implemented for forecasting training
- the `scripts/long_term_forecast/...` shell scripts are forecasting-only

## Installation

```bash
git clone https://github.com/mimanchi-dongze/TimeMixer-ME.git
cd TimeMixer-ME
pip install -r requirements.txt
```

## Requirements

```text
einops==0.8.1
matplotlib==3.10.1
numpy==2.2.4
pandas==2.0.3
reformer_pytorch==1.4.4
scikit_learn==1.4.2
scipy==1.15.2
torch==2.6.0
shap==0.46.0
```

## Repository Layout

```text
.
|-- data_provider/          # dataset loaders and dataloader factory
|-- exp/                    # task-specific training / evaluation loops
|-- layers/                 # normalization, embedding, decomposition helpers
|-- models/                 # TimeMixer_ME model definition
|-- scripts/
|   |-- ablation/           # robustness ablation scripts
|   |-- explainability/     # SHAP + attention extraction for forecasting
|   `-- long_term_forecast/ # forecasting shell scripts
|-- utils/                  # metrics, tools, visualization helpers
`-- run.py                  # unified CLI entry
```

## Supported Tasks

### 1. Long-Term Forecasting

Supports:

- multivariate-to-multivariate forecasting
- multivariate-to-univariate forecasting
- public benchmark datasets such as `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `PEMS`, `ECL`, `Solar`, `Traffic`
- custom CSV datasets

### 2. Imputation

Supports:

- masked reconstruction training and evaluation
- datasets loaded through the standard forecasting-style CSV path

### 3. Anomaly Detection

Supports:

- reconstruction-based anomaly scoring
- datasets: `PSM`, `MSL`, `SMAP`, `SMD`, `SWAT`

By default, evaluation does **not** use label-aware anomaly post-adjustment. If you need benchmark-compatible post-processing, enable:

```bash
--anomaly_adjustment 1
```

### 4. Classification

Supports padded variable-length classification batches through a dedicated classification loader.

The classification loader currently accepts one of the following under `root_path`:

1. `classification.npz`, `dataset.npz`, or `data.npz`

Supported key layouts:

- `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, `y_test`
- `train_x`, `train_y`, `val_x`, `val_y`, `test_x`, `test_y`

2. split `.npy` files

- `train.npy` + `train_labels.npy`
- `val.npy` + `val_labels.npy` optional
- `test.npy` + `test_labels.npy`

or

- `x_train.npy` + `y_train.npy`
- `x_val.npy` + `y_val.npy` optional
- `x_test.npy` + `y_test.npy`

If no explicit validation split is provided, the loader will split validation samples from the training set automatically.

## Data Format

### Forecasting / Imputation Custom CSV

Expected CSV format:

```text
date,feature_1,feature_2,...,target
2024-01-01 00:00:00,...
2024-01-01 01:00:00,...
...
```

Notes:

- `date` is required for the standard custom loader
- `target` is required when using `features=S` or `features=MS`
- for `features=M`, all columns after `date` are treated as model inputs

Default split for `custom` CSV datasets:

- train: 70%
- val: 10%
- test: 20%

### Built-in Anomaly Datasets

The repository expects the original benchmark file layouts already used by the corresponding loaders:

- `PSM`: `train.csv`, `test.csv`, `test_label.csv`
- `MSL`: `MSL_train.npy`, `MSL_test.npy`, `MSL_test_label.npy`
- `SMAP`: `SMAP_train.npy`, `SMAP_test.npy`, `SMAP_test_label.npy`
- `SMD`: `SMD_train.npy`, `SMD_test.npy`, `SMD_test_label.npy`
- `SWAT`: `swat_train2.csv`, `swat2.csv`

## Quick Start

### Long-Term Forecasting

Train:

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id demo_forecast \
  --model TimeMixer_ME \
  --data ETTm1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7
```

Test:

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model_id demo_forecast \
  --model TimeMixer_ME \
  --data ETTm1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7
```

### Imputation

```bash
python run.py \
  --task_name imputation \
  --is_training 1 \
  --model_id demo_imputation \
  --model TimeMixer_ME \
  --data custom \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --mask_rate 0.25
```

### Anomaly Detection

```bash
python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --model_id demo_anomaly \
  --model TimeMixer_ME \
  --data PSM \
  --root_path ./dataset/PSM/ \
  --data_path ignored.csv \
  --features M \
  --seq_len 100 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --anomaly_ratio 1
```

If you need the older benchmark-style post-adjustment:

```bash
--anomaly_adjustment 1
```

### Classification

Example layout:

```text
./dataset/my_classification/
|-- train.npy
|-- train_labels.npy
|-- test.npy
`-- test_labels.npy
```

Training example:

```bash
python run.py \
  --task_name classification \
  --is_training 1 \
  --model_id demo_cls \
  --model TimeMixer_ME \
  --data custom \
  --root_path ./dataset/my_classification/ \
  --data_path ignored.npy \
  --features M \
  --enc_in 3 \
  --seq_len 128 \
  --label_len 0 \
  --pred_len 0
```

Notes:

- for classification, `data` is still required by the CLI, but the actual split loading is driven by `task_name=classification` and files under `root_path`
- `seq_len` will be overridden internally to the maximum sequence length in the loaded dataset
- if there is no explicit validation split, the loader uses `--classification_val_ratio` and `--classification_split_seed`

## Core Model Ideas

### Multi-Scale Decomposition

The model decomposes sequence representations into seasonal and trend components, then mixes them across scales.

Available decomposition methods:

- `moving_avg`
- `dft_decomp`

### MTMEM

MTMEM combines:

- local depthwise temporal convolution
- short-term self-attention
- a learnable memory bank for long-range context retrieval
- periodicity enhancement
- adaptive fusion weights for short-term and long-term branches

### AnyVariateAttention

AnyVariateAttention supports two regimes:

- `channel_independence=1`: cross-variate attention at each time step
- `channel_independence=0`: temporal attention on the multivariate sequence directly

## Important Arguments

| Argument | Description | Default |
|---|---|---|
| `task_name` | Task type | `long_term_forecast` |
| `model` | Model name | `TimeMixer_ME` |
| `seq_len` | Input sequence length | `96` |
| `label_len` | Decoder warm-up length | `48` |
| `pred_len` | Forecast horizon | `96` |
| `d_model` | Hidden size | `16` |
| `e_layers` | Number of PDM blocks | `2` |
| `d_ff` | Feed-forward hidden size | `32` |
| `dropout` | Dropout ratio | `0.1` |
| `enc_in` | Number of input channels | `7` |
| `c_out` | Number of output channels | `7` |
| `down_sampling_layers` | Number of multiscale levels beyond the base scale | `0` |
| `down_sampling_window` | Down-sampling factor per level | `1` |
| `decomp_method` | Decomposition method | `moving_avg` |
| `channel_independence` | Use channel-independent processing | `1` |
| `num_memories` | MTMEM memory bank size | `32` |
| `causal_levels` | Preferred number of MTMEM attention heads | `4` |
| `classification_val_ratio` | Auto validation split ratio for classification | `0.2` |
| `classification_split_seed` | Random seed for classification split | `42` |
| `anomaly_adjustment` | Enable label-aware anomaly adjustment | `0` |

## Explainability

Explainability scripts currently support `long_term_forecast` checkpoints.

### SHAP Attribution

```bash
python scripts/explainability/shap_forecast_explain.py \
  --checkpoint ./checkpoints/<your_setting>/checkpoint.pth \
  --data custom \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 21 \
  --c_out 21 \
  --background_size 16 \
  --explain_size 16 \
  --pred_index 0 \
  --feature_index 0 \
  --output_dir ./results/shap_explain/weather
```

Outputs:

- `feature_importance.csv`
- `feature_importance_topk.png`
- `shap_values.npy`
- `meta.json`

### Attention Extraction

```bash
python scripts/explainability/attention_forecast_explain.py \
  --checkpoint ./checkpoints/<your_setting>/checkpoint.pth \
  --data custom \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 21 \
  --c_out 21 \
  --channel_independence 1 \
  --output_dir ./results/attention_explain/weather
```

Outputs:

- `attn_layer*_scale*.npy/.png`
- `attn_layer*_mean.npy/.png`
- `attn_global_mean.npy/.png`
- `attention_summary.json`

## Robust Training Augmentation

For forecasting, you can simulate:

- random missing values
- block missing segments
- mixed missing patterns
- Gaussian noise
- channel-level sensor dropout

Example:

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id robust_demo \
  --model TimeMixer_ME \
  --data custom \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --robust_train_aug 1 \
  --aug_missing_rate 0.1 \
  --aug_noise_std 0.03 \
  --aug_missing_fill mean \
  --aug_missing_mode mixed \
  --aug_block_len 12 \
  --aug_channel_dropout_rate 0.05
```

Arguments:

- `robust_train_aug`: enable augmentation
- `aug_missing_rate`: random missing ratio
- `aug_noise_std`: Gaussian noise std
- `aug_missing_fill`: `zero` or `mean`
- `aug_missing_mode`: `random`, `block`, or `mixed`
- `aug_block_len`: missing block length
- `aug_channel_dropout_rate`: full-channel dropout ratio

### Ablation Script

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\ablation\run_robustness_ablation.ps1 `
  -RootPath ./dataset/weather/ `
  -DataPath weather.csv `
  -PredLens 96,192,336,720 `
  -MissingRate 0.1 `
  -NoiseStd 0.03 `
  -BlockLen 12 `
  -ChannelDropoutRate 0.05
```

Dry run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\ablation\run_robustness_ablation.ps1 -DryRun
```

Result summary:

```bash
python scripts/ablation/summarize_robustness_ablation.py \
  --results_root ./results \
  --output_dir ./results/ablation_summary
```

## Forecast Shell Scripts

The `scripts/long_term_forecast/Weather_script/` directory contains several forecasting shell scripts such as:

- `TimeMixer_ME_unify.sh`
- `TimeMixer_ME_aral.sh`
- `TimeMixer_ME_hotan.sh`
- `TimeMixer_ME_kashgar.sh`

Some of these are tied to private or local datasets. Use `TimeMixer_ME_unify.sh` as the public reference template and adjust paths to your own environment.

## Results and Outputs

Training and evaluation artifacts are written to:

- `./checkpoints/<setting>/`
- `./results/<setting>/`
- `./test_results/<setting>/`

Depending on the task, the repository may save:

- `checkpoint.pth`
- `metrics.npy`
- `pred.npy`
- `true.npy`
- task-specific text summaries such as `result_long_term_forecast.txt`

## Notes

- The CLI always requires `--model_id` and `--data`
- `--checkpoints` defaults to `./checkpoints/`
- classification uses a dedicated loader even though `--data` is still required by the parser
- explainability scripts assume the checkpoint config matches the model arguments used for extraction

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
