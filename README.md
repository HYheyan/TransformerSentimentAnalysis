# 文本情感分析系统

这个项目是一个文本情感分析系统，可以根据输入的英文文本判断其情感倾向（正面或负面）。系统使用自定义Transformer模型实现。

## 功能特点

- 专门针对aclImdb数据集优化
- 集成TensorBoard可视化训练过程
- 自动划分训练集和验证集
- 支持GPU加速训练

## 文件结构

```
.
├── models/
│   └── transformer_model.py   # 自定义Transformer模型实现
├── utils/
│   ├── data_processing.py     # 数据处理工具
│   └── model_utils.py         # 模型工具函数
├── sentiment_analysis_main.py # 主程序入口
└── aclImdb/                   # aclImdb数据集目录
```

## 环境依赖

- Python 3.7+
- PyTorch 1.9+
- Transformers
- TensorBoard


## 使用方法

运行主程序：
```bash
python sentiment_analysis_main.py
```

程序将自动加载aclImdb数据集进行训练。

### 使用自定义Transformer模型

自定义Transformer模型完全从头实现，不依赖预训练权重，包含以下组件：

1. 多头注意力机制
2. 位置编码
3. 前馈神经网络
4. 层归一化和残差连接

## 模型训练

系统专门为aclImdb数据集（25000条样本）优化了以下参数：

- 批次大小: 16
- 学习率: 1e-4
- 训练轮数: 50
- 日志显示频率: 每5轮显示一次

训练过程中会将训练指标记录到TensorBoard日志中。

## 查看训练过程

使用TensorBoard查看训练过程：
```bash
tensorboard --logdir=runs
```

## 数据集

本系统专门使用aclImdb数据集：
- 包含25000条电影评论
- 12500条正面评价，12500条负面评价
- 训练集：20000条样本
- 验证集：5000条样本

## 模型保存

训练完成后，系统会自动保存最佳模型：
- Transformer模型：best_transformer_model.pth