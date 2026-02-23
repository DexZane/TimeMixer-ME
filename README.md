# TimeMixer_ME

TimeMixer_ME是一个基于TimeMixer创新的深度学习模型，专门用于处理复杂的时间序列任务。该模型采用多尺度记忆增强机制，能够有效处理长期依赖关系和多变量交互。

## 📋 主要功能

- 长期时序预测
- 时序数据补全
- 异常检测
- 时序分类

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
大家可以看到Scripts下有aral、hotan等脚本，那个是我的私有数据集暂未公开，大家可以用[TimeMixer_ME_unify.sh](scripts%2Flong_term_forecast%2FWeather_script%2FTimeMixer_ME_unify.sh)来测试脚本
## 💡 核心特性

1. **多尺度时序分解**
   - 自适应季节性分离
   - 趋势提取与分析
   - 多层级特征融合

2. **记忆增强机制**
   - 短期动态记忆
   - 长期静态记忆
   - 自适应记忆更新

3. **变量注意力机制**
   - 跨变量交互建模
   - 自适应权重分配
   - 位置感知特征提取

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
| task_name | 任务类型 | long_term_forecast |
| seq_len | 输入序列长度 | 96 |
| pred_len | 预测序列长度 | 96 |
| d_model | 模型维度 | 16 |
| num_memories | 记忆单元数量 | 32 |
| causal_levels | 因果层级数 | 4 |

## 🤝 贡献

欢迎提交问题和改进建议！如需贡献代码：

1. Fork本仓库
2. 创建您的特性分支
3. 提交您的更改
4. 创建Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE)
