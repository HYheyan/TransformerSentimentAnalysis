# 情感分析项目完整教程

本文档旨在帮助您深入理解整个情感分析项目的结构、工作原理以及如何使用它。

## 项目概述

这是一个基于Transformer模型的文本情感分析系统，能够识别英文文本的情感倾向（正面或负面）。项目特别针对aclImdb数据集进行了优化，实现了完整的训练和推理流程。

## 项目结构

```
.
├── models/
│   └── transformer_model.py   # 自定义Transformer模型实现
├── utils/
│   ├── data_processing.py     # 数据处理工具
│   ├── model_utils.py         # 模型工具函数
│   └── simple_tokenizer.py    # 简单分词器实现
├── sentiment_analysis_main.py # 主程序入口
├── aclImdb/                   # aclImdb数据集目录
├── best_transformer_model.pth # 已训练好的模型文件
└── teach.md                  # 本教程文件
```

## 核心组件详解

### 1. 主程序 (sentiment_analysis_main.py)

这是项目的入口点，负责协调整个训练和评估流程：

- 初始化TensorBoard日志记录器
- 加载aclImdb数据集
- 创建词汇表和分词器
- 划分训练集和验证集
- 初始化Transformer模型
- 设置训练参数（优化器、损失函数等）
- 执行模型训练循环
- 保存最佳模型
- 提供示例文本的情感预测

### 2. Transformer模型 (models/transformer_model.py)

包含了完整的Transformer模型实现：

#### PositionalEncoding (位置编码)
- 为输入序列添加位置信息
- 使用正弦和余弦函数生成位置编码

#### MultiHeadAttention (多头注意力)
- 实现标准的缩放点积注意力机制
- 支持掩码操作以忽略填充部分

#### PositionwiseFeedForward (位置前馈网络)
- 两层全连接网络，中间使用ReLU激活函数

#### TransformerEncoderLayer (Transformer编码器层)
- 结合多头注意力和前馈网络
- 包含残差连接和层归一化

#### TransformerEncoder (Transformer编码器)
- 由多个编码器层堆叠而成

#### TransformerTextClassificationModel (Transformer文本分类模型)
- 整合所有组件，用于文本分类任务
- 使用序列的第一个token ([CLS])进行分类

### 3. 数据处理工具 (utils/data_processing.py)

负责加载和预处理aclImdb数据集：

- `preprocess_text`: 清理文本数据（转换为小写，移除特殊字符）
- `load_aclImdb_dataset`: 从文件系统加载训练数据

### 4. 分词器 (utils/simple_tokenizer.py)

实现了简易的分词和编码功能：

#### SimpleVocab (简单词汇表)
- 管理词汇到ID的映射关系
- 包含特殊token（PAD, UNK, CLS, SEP）

#### SimpleTokenizer (简单分词器)
- 将文本转换为token序列
- 将token序列转换为ID序列
- 生成注意力掩码


## 工作流程

### 数据准备
1. 从aclImdb/train/pos和aclImdb/train/neg目录加载正面和负面评论
2. 对文本进行预处理（转小写、去除特殊字符）
3. 构建词汇表，过滤低频词
4. 使用SimpleTokenizer将文本转换为模型可接受的格式

### 模型训练
1. 初始化Transformer模型，设置超参数：
   - 词汇表大小：根据实际词汇表确定
   - 模型维度(d_model)：256
   - 注意力头数(num_heads)：8
   - 编码器层数(num_layers)：4
   - 前馈网络维度(d_ff)：512
   - Dropout率：0.5
   - 分类数：2（正面/负面）

2. 设置训练参数：
   - 批次大小：16
   - 学习率：1e-4
   - 优化器：AdamW
   - 损失函数：交叉熵损失
   - 训练轮数：50轮

3. 训练过程：
   - 每轮迭代所有训练批次
   - 计算损失并反向传播
   - 使用梯度裁剪防止梯度爆炸
   - 记录训练指标到TensorBoard
   - 保存验证集上表现最好的模型

### 模型评估
1. 在验证集上评估模型性能
2. 计算准确率和损失值
3. 使用ReduceLROnPlateau调度器根据验证损失调整学习率

## 如何运行项目

### 环境要求
- Python 3.7+
- PyTorch 1.9+
- TensorBoard

### 安装依赖
```bash
pip install torch torchvision tensorboard
```

### 运行训练
```bash
python sentiment_analysis_main.py
```

### 查看训练过程
```bash
tensorboard --logdir=runs
```

## 模型特点

1. **完全自定义实现**：不依赖预训练模型，从零实现Transformer架构
2. **针对aclImdb优化**：专门针对aclImdb数据集调整了超参数
3. **集成TensorBoard**：实时可视化训练过程
4. **模型保存机制**：自动保存验证集上表现最好的模型

## 超参数说明

| 参数 | 值 | 说明 |
|------|----|------|
| Batch Size | 16 | 每批处理的样本数量 |
| Learning Rate | 1e-4 | 初始学习率 |
| Epochs | 50 | 训练轮数 |
| Model Dimensions | 256 | 词嵌入和模型内部维度 |
| Attention Heads | 8 | 多头注意力的头数 |
| Encoder Layers | 4 | Transformer编码器层数 |
| Feed Forward Dim | 512 | 前馈网络的隐藏层维度 |
| Dropout | 0.5 | Dropout比率 |

## 项目扩展建议

1. **增加更多评估指标**：如精确率、召回率、F1分数等
2. **支持更多分类类别**：扩展到多情感分类（积极、消极、中性等）
3. **改进分词器**：使用更高级的分词技术
4. **增加早停机制**：防止过拟合
5. **支持GPU并行训练**：利用多GPU加速训练过程
6. **增加模型解释性**：可视化注意力权重 。