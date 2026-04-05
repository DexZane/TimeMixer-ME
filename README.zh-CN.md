# TimeMixer_ME

[English README](./README.md)

TimeMixer_ME 是一个基于 TimeMixer 系列模型扩展的多任务时间序列仓库。在原始多尺度混合思路上，这个实现进一步加入了：

- `MTMEM`，即多时间尺度记忆增强模块
- `AnyVariateAttention`，用于显式建模跨变量交互

当前仓库通过统一的 `run.py` 入口支持以下任务：

- 长期时间序列预测
- 缺失值补全
- 异常检测
- 时序分类

仓库还包含：

- 面向预测任务的 SHAP 与注意力可解释性脚本
- 面向预测训练的鲁棒性增强
- 若干天气预测实验脚本

## 当前定位

这个仓库的主要优化目标仍然是 `long_term_forecast`。

其余三个任务现在已经具备可运行的数据与模型路径，但在脚本覆盖度和实验封装程度上不如预测任务完整。尤其是：

- explainability 脚本目前只针对 forecasting checkpoint
- robust augmentation 当前只在 forecasting 训练流程中实现
- `scripts/long_term_forecast/...` 下的脚本仅覆盖预测任务

## 安装

```bash
git clone https://github.com/mimanchi-dongze/TimeMixer-ME.git
cd TimeMixer-ME
pip install -r requirements.txt
```

## 依赖

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

## 目录结构

```text
.
|-- data_provider/          # 数据集加载器与 dataloader 工厂
|-- exp/                    # 各任务训练/验证/测试逻辑
|-- layers/                 # 归一化、嵌入、分解等基础层
|-- models/                 # TimeMixer_ME 主模型
|-- scripts/
|   |-- ablation/           # 鲁棒性消融脚本
|   |-- explainability/     # 预测任务的 SHAP / 注意力提取
|   `-- long_term_forecast/ # 预测脚本
|-- utils/                  # 指标、工具、可视化等
`-- run.py                  # 统一 CLI 入口
```

## 支持的任务

### 1. 长期预测

支持：

- 多变量到多变量预测
- 多变量到单变量预测
- `ETTh1`、`ETTh2`、`ETTm1`、`ETTm2`、`PEMS`、`ECL`、`Solar`、`Traffic` 等公开数据集
- 自定义 CSV 数据集

### 2. 缺失值补全

支持：

- 基于随机掩码的重建训练与评估
- 使用标准 forecasting 风格 CSV 数据路径

### 3. 异常检测

支持：

- 基于重建误差的异常打分
- `PSM`、`MSL`、`SMAP`、`SMD`、`SWAT` 数据集

默认评估 **不启用** 标签感知的异常后处理。如果你需要兼容旧 benchmark 的后处理方式，可以显式开启：

```bash
--anomaly_adjustment 1
```

### 4. 时序分类

分类任务使用专门的数据加载器，支持带 padding 的可变长 batch。

分类数据在 `root_path` 下可采用以下任一格式。

1. `classification.npz`、`dataset.npz` 或 `data.npz`

支持的键名布局：

- `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, `y_test`
- `train_x`, `train_y`, `val_x`, `val_y`, `test_x`, `test_y`

2. 分开的 `.npy` 文件

- `train.npy` + `train_labels.npy`
- `val.npy` + `val_labels.npy`，可选
- `test.npy` + `test_labels.npy`

或者：

- `x_train.npy` + `y_train.npy`
- `x_val.npy` + `y_val.npy`，可选
- `x_test.npy` + `y_test.npy`

如果没有显式的验证集，加载器会自动从训练集中切分验证样本。

## 数据格式说明

### Forecasting / Imputation 的自定义 CSV

推荐格式：

```text
date,feature_1,feature_2,...,target
2024-01-01 00:00:00,...
2024-01-01 01:00:00,...
...
```

说明：

- 标准自定义加载器要求存在 `date` 列
- 当 `features=S` 或 `features=MS` 时，需要指定 `target`
- 当 `features=M` 时，`date` 之后的全部列都会作为输入特征

`custom` CSV 默认按以下比例切分：

- train: 70%
- val: 10%
- test: 20%

### 内置异常检测数据格式

仓库默认沿用各 benchmark 对应的数据文件布局：

- `PSM`: `train.csv`, `test.csv`, `test_label.csv`
- `MSL`: `MSL_train.npy`, `MSL_test.npy`, `MSL_test_label.npy`
- `SMAP`: `SMAP_train.npy`, `SMAP_test.npy`, `SMAP_test_label.npy`
- `SMD`: `SMD_train.npy`, `SMD_test.npy`, `SMD_test_label.npy`
- `SWAT`: `swat_train2.csv`, `swat2.csv`

## 快速开始

### 长期预测

训练：

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

测试：

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

### 缺失值补全

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

### 异常检测

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

如果需要旧版 benchmark 风格后处理：

```bash
--anomaly_adjustment 1
```

### 时序分类

示例目录：

```text
./dataset/my_classification/
|-- train.npy
|-- train_labels.npy
|-- test.npy
`-- test_labels.npy
```

训练示例：

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

说明：

- 分类任务虽然仍然需要传 `--data`，但实际加载由 `task_name=classification` 和 `root_path` 下的文件决定
- `seq_len` 会在运行时被数据集中最长序列长度覆盖
- 如果没有显式验证集，将使用 `--classification_val_ratio` 和 `--classification_split_seed` 自动切分

## 核心模型思路

### 多尺度分解

模型会先对序列表征进行季节项与趋势项分解，然后在不同尺度之间完成混合。

目前支持的分解方法：

- `moving_avg`
- `dft_decomp`

### MTMEM

MTMEM 结合了：

- 局部 depthwise temporal convolution
- 短期自注意力
- 可学习 memory bank 的长程上下文读取
- 周期增强分支
- 短期/长期特征的自适应融合权重

### AnyVariateAttention

AnyVariateAttention 有两种工作模式：

- `channel_independence=1`：在每个时间步上做跨变量注意力
- `channel_independence=0`：直接在多变量序列上做时间维注意力

## 重要参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| `task_name` | 任务类型 | `long_term_forecast` |
| `model` | 模型名 | `TimeMixer_ME` |
| `seq_len` | 输入序列长度 | `96` |
| `label_len` | decoder 预热长度 | `48` |
| `pred_len` | 预测长度 | `96` |
| `d_model` | 隐层维度 | `16` |
| `e_layers` | PDM block 数量 | `2` |
| `d_ff` | 前馈层宽度 | `32` |
| `dropout` | dropout 比例 | `0.1` |
| `enc_in` | 输入通道数 | `7` |
| `c_out` | 输出通道数 | `7` |
| `down_sampling_layers` | 基础尺度之外的多尺度层数 | `0` |
| `down_sampling_window` | 每层下采样倍率 | `1` |
| `decomp_method` | 分解方法 | `moving_avg` |
| `channel_independence` | 是否采用通道独立处理 | `1` |
| `num_memories` | MTMEM memory bank 大小 | `32` |
| `causal_levels` | MTMEM 优先使用的 attention head 数 | `4` |
| `classification_val_ratio` | 分类任务自动验证集比例 | `0.2` |
| `classification_split_seed` | 分类任务切分随机种子 | `42` |
| `anomaly_adjustment` | 是否启用标签感知异常修正 | `0` |

## 可解释性

当前 explainability 脚本仅支持 `long_term_forecast` checkpoint。

### SHAP 归因

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

输出：

- `feature_importance.csv`
- `feature_importance_topk.png`
- `shap_values.npy`
- `meta.json`

### 注意力提取

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

输出：

- `attn_layer*_scale*.npy/.png`
- `attn_layer*_mean.npy/.png`
- `attn_global_mean.npy/.png`
- `attention_summary.json`

## 鲁棒训练增强

当前 forecasting 训练支持模拟：

- 随机缺失
- 连续 block 缺失
- mixed 缺失模式
- 高斯噪声
- channel 级别传感器掉线

示例：

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

相关参数：

- `robust_train_aug`：是否启用增强
- `aug_missing_rate`：随机缺失比例
- `aug_noise_std`：高斯噪声标准差
- `aug_missing_fill`：`zero` 或 `mean`
- `aug_missing_mode`：`random`、`block` 或 `mixed`
- `aug_block_len`：连续缺失块长度
- `aug_channel_dropout_rate`：整通道 dropout 比例

### 消融脚本

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

Dry run：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\ablation\run_robustness_ablation.ps1 -DryRun
```

结果汇总：

```bash
python scripts/ablation/summarize_robustness_ablation.py \
  --results_root ./results \
  --output_dir ./results/ablation_summary
```

## Forecast 脚本

`scripts/long_term_forecast/Weather_script/` 目录下包含若干 forecasting shell 脚本，例如：

- `TimeMixer_ME_unify.sh`
- `TimeMixer_ME_aral.sh`
- `TimeMixer_ME_hotan.sh`
- `TimeMixer_ME_kashgar.sh`

其中一部分依赖私有或本地数据集。若用于公开或自定义实验，建议优先参考 `TimeMixer_ME_unify.sh` 并按本地环境修改路径。

## 输出目录

训练和评估产物默认写入：

- `./checkpoints/<setting>/`
- `./results/<setting>/`
- `./test_results/<setting>/`

按任务不同，常见输出包括：

- `checkpoint.pth`
- `metrics.npy`
- `pred.npy`
- `true.npy`
- 各任务文本结果汇总，如 `result_long_term_forecast.txt`

## 说明

- CLI 层始终要求传 `--model_id` 和 `--data`
- `--checkpoints` 默认值为 `./checkpoints/`
- 分类任务使用独立 loader，虽然 parser 仍要求 `--data`
- explainability 脚本默认要求 checkpoint 配置与当前传入参数匹配

## 许可证

本项目采用 MIT License，详见 [LICENSE](LICENSE)。

