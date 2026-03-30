# TimeMixer_ME

TimeMixer_ME 是一个基于 TimeMixer 架构创新的深度学习模型，专门用于处理复杂的时间序列任务。该模型引入了**多尺度记忆增强机制 (MTMEM)** 与**跨变量注意力机制 (AnyVariateAttention)**，能够有效捕捉长短期时间依赖关系并显式建模多变量之间的复杂耦合交互。

## 📋 主要功能

- **长期时序预测 (Long-term Forecasting)**：支持多变量到多变量、多变量到单变量的精准预测。
- **时序数据补全 (Imputation)**：有效处理缺失值，恢复时间序列的完整性。
- **异常检测 (Anomaly Detection)**：精准识别时间序列中的异常模式与突发事件。
- **时序分类 (Classification)**：提取时序特征进行高准确率的模式分类。

## 🚀 快速开始

### 环境要求

```bash
einops==0.8.1
matplotlib==3.10.1
numpy==2.2.4
pandas==2.0.3
reformer_pytorch==1.4.4
scikit_learn==1.4.2
scipy==1.15.2
torch==2.6.0
```

### 安装

```bash
git clone https://github.com/mimanchi-dongze/TimeMixer-ME.git
cd TimeMixer-ME
pip install -r requirements.txt
```

### 训练模型

示例：
```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model TimeMixer_ME \
    --data ETTm1 \
    --root_path ./data/ETT/ \
    --data_path ETTm1.csv \
    --features M \
    --seq_len 96 \
    --pred_len 96
```

### 模型测试

```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --model TimeMixer_ME \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --pred_len 96
```

> **提示**：大家可以看到 `scripts` 目录下有 `aral`、`hotan` 等脚本，这些是我的私有数据集暂未公开。大家可以使用 [TimeMixer_ME_unify.sh](scripts/long_term_forecast/Weather_script/TimeMixer_ME_unify.sh) 来测试公开数据集脚本。

## 💡 核心特性

1. **多尺度时序分解 (Multi-Scale Series Decomposition)**
   - **自适应季节性分离**：精准剥离时间序列中的周期性成分。
   - **趋势提取与分析**：平滑提取宏观走向，抗噪能力强。
   - **多层级特征融合**：在不同时间分辨率下进行特征交互。

2. **多尺度记忆增强模块 (MTMEM - Multi-Time Scale Memory Enhancement Module)**
   - **短期动态记忆**：结合局部卷积与注意力机制，敏锐捕捉近期突变与局部依赖。
   - **长期静态记忆**：利用全局上下文与可学习的 Memory Bank，提取并存储长期演变模式。
   - **周期性特征提取**：通过一维卷积网络，保留并强化时间维度的周期性波动信号。
   - **自适应融合**：根据不同时间尺度的特征重要性，动态分配权重并进行深度特征增强。

3. **跨变量注意力机制 (AnyVariateAttention)**
   - **跨变量交互建模**：打破传统模型的通道独立性 (Channel Independence) 限制，显式建模不同物理变量（如温度、湿度、风速等）之间的复杂耦合关系。
   - **自适应权重分配**：动态评估各变量对当前预测目标的贡献度。
   - **位置感知特征提取**：结合变量特征与时间位置信息，提升多维数据的表达能力。

## 📊 支持的数据格式

- 单变量时间序列
- 多变量时间序列
- 支持的时间频率：
  - 秒级 (s)
  - 分钟级 (t)
  - 小时级 (h)
  - 天级 (d)
  - 周级 (w)
  - 月级 (m)

## ⚙️ 主要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `task_name` | 任务类型 | `long_term_forecast` |
| `seq_len` | 输入序列长度 | `96` |
| `pred_len` | 预测序列长度 | `96` |
| `d_model` | 模型维度 | `16` |
| `num_memories` | 记忆单元数量 | `32` |
| `causal_levels` | 因果层级数 | `4` |

## 🤝 贡献

欢迎提交问题和改进建议！如需贡献代码：

1. Fork 本仓库
2. 创建您的特性分支
3. 提交您的更改
4. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE)

## Explainability (SHAP)

You can run SHAP-based feature attribution for `long_term_forecast` checkpoints:

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
- `feature_importance.csv`: channel-level mean absolute SHAP values
- `feature_importance_topk.png`: top-k feature importance plot
- `shap_values.npy`: raw SHAP tensor `[batch, seq_len, features]`
- `meta.json`: run metadata

### Attention Explainability

You can also extract cross-variate attention maps (requires checkpoint from the same model config):

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
- `attn_layer*_scale*.npy/.png`: per-layer per-scale attention matrix and heatmap
- `attn_layer*_mean.npy/.png`: mean attention per layer
- `attn_global_mean.npy/.png`: global mean attention across layers/scales
- `attention_summary.json`: top attention feature pairs and summary stats

## Robust Training Augmentation (Missing + Noise)

To improve robustness to missing values and noisy observations, you can enable train-time input augmentation:

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model TimeMixer_ME \
  --data custom \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --robust_train_aug 1 \
  --aug_missing_rate 0.1 \
  --aug_noise_std 0.03 \
  --aug_missing_fill mean \
  --aug_missing_mode mixed \
  --aug_block_len 12 \
  --aug_channel_dropout_rate 0.05
```

Arguments:
- `robust_train_aug`: enable (`1`) / disable (`0`) augmentation (default `0`)
- `aug_missing_rate`: random masking ratio on encoder input during training (default `0.0`)
- `aug_noise_std`: Gaussian noise std added to encoder input during training (default `0.0`)
- `aug_missing_fill`: filling strategy for masked values, `zero` or `mean` (default `zero`)
- `aug_missing_mode`: missing pattern type, `random`, `block`, or `mixed` (default `random`)
- `aug_block_len`: contiguous missing block length when using block/mixed mode (default `8`)
- `aug_channel_dropout_rate`: channel-wise dropout ratio to simulate sensor-level outage (default `0.0`)

### Robustness Ablation Script (PowerShell)

Run one-click ablation for `baseline / random / block / mixed` under the same base config:

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

Dry run (print commands only):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\ablation\run_robustness_ablation.ps1 -DryRun
```

### Robustness Ablation Result Summary

After experiments finish, summarize all `metrics.npy` into CSV/Markdown tables:

```bash
python scripts/ablation/summarize_robustness_ablation.py \
  --results_root ./results \
  --output_dir ./results/ablation_summary
```

Outputs:
- `robustness_ablation_raw.csv`: one row per setting
- `robustness_ablation_mse.csv`: pivot table (`pred_len x mode`) for MSE
- `robustness_ablation_mae.csv`: pivot table (`pred_len x mode`) for MAE
- `robustness_ablation_rmse.csv`: pivot table (`pred_len x mode`) for RMSE
- `robustness_ablation_summary.md`: markdown tables (MSE/MAE/RMSE)
