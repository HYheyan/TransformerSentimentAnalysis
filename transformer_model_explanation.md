# Transformer 模型详解

本文档旨在全面解释 `transformer_model.py` 文件中的内容，包括各个类和函数的作用、实现原理以及它们之间的关系。

## 文件概述

`transformer_model.py` 是一个完整的自定义 Transformer 模型实现，用于文本分类任务。该文件包含了构建 Transformer 模型所需的所有组件，从底层的注意力机制到完整的分类模型。

## 核心类详解

### 1. PositionalEncoding（位置编码）

位置编码是 Transformer 模型的重要组成部分，用于为输入序列添加位置信息，因为 Transformer 本身不具备处理序列顺序的能力。

#### 实现原理
- 使用正弦和余弦函数生成不同位置的编码
- 对于位置 `pos` 和维度 `i`，编码计算如下：
  - `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
  - `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

#### 关键代码
```python
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

### 2. MultiHeadAttention（多头注意力机制）

多头注意力是 Transformer 的核心机制，允许模型在不同表示子空间中关注信息的不同方面。

#### 实现原理
1. 线性变换生成 Q（查询）、K（键）、V（值）矩阵
2. 将这些矩阵分割成多个头
3. 并行计算每个头的注意力
4. 合并所有头的结果并进行线性变换

#### 关键步骤
1. **缩放点积注意力**：
   ```python
   attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
   ```
2. **Softmax 归一化**：
   ```python
   attn_probs = torch.softmax(attn_scores, dim=-1)
   ```
3. **加权求和**：
   ```python
   output = torch.matmul(attn_probs, V)
   ```

### 3. PositionwiseFeedForward（位置前馈网络）

前馈网络对序列中的每个位置独立地应用相同的 MLP（多层感知机）。

#### 实现结构
- 两层全连接网络，中间使用 ReLU 激活函数
- 第一层将维度从 `d_model` 扩展到 `d_ff`
- 第二层将维度从 `d_ff` 压缩回 `d_model`

### 4. TransformerEncoderLayer（Transformer 编码器层）

编码器层是 Transformer 的基本构建块，包含一个多头注意力子层和一个前馈网络子层。

#### 残差连接和层归一化
每个子层都采用残差连接和层归一化：
```python
x = self.norm1(x + self.dropout(attn_output))
x = self.norm2(x + self.dropout(ff_output))
```

### 5. TransformerEncoder（Transformer 编码器）

编码器由多个编码器层堆叠而成：
```python
self.layers = nn.ModuleList([
    TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) 
    for _ in range(num_layers)
])
```

### 6. TransformerTextClassificationModel（Transformer 文本分类模型）

这是完整的分类模型，整合了所有组件用于文本分类任务。

#### 模型结构
1. **词嵌入层**：将词汇索引转换为向量表示
2. **位置编码**：添加位置信息
3. **编码器**：多层 Transformer 编码器
4. **分类头**：使用第一个 token（[CLS]）进行分类

#### 前向传播过程
```python
# 输入嵌入和位置编码
x = self.embedding(x) * math.sqrt(self.d_model)
x = self.positional_encoding(x)
x = self.dropout(x)

# 编码器处理
x = self.encoder(x, mask)

# 使用第一个token进行分类
x = x[:, 0, :]  # 取序列的第一个token
x = self.dropout(x)
x = self.fc(x)
```

### 7. TransformerDataset（Transformer 数据集类）

这是一个 PyTorch Dataset 类，用于处理文本数据并将其转换为模型可用的格式。

#### 功能
- 接收原始文本和标签
- 使用分词器处理文本
- 返回模型训练所需的张量格式

## 辅助函数

### train_transformer_model（训练函数）

用于训练 Transformer 模型的函数，包含以下关键步骤：
1. 设置模型为训练模式
2. 前向传播计算损失
3. 反向传播计算梯度
4. 使用梯度裁剪防止梯度爆炸
5. 更新模型参数
6. 计算准确率

### evaluate_transformer_model（评估函数）

用于评估 Transformer 模型的函数：
1. 设置模型为评估模式
2. 禁用梯度计算以提高效率
3. 计算验证集上的损失和准确率

## 模型特点

### 1. 完全自定义实现
- 不依赖预训练模型
- 从头实现所有组件
- 便于理解和修改

### 2. 模块化设计
- 每个类都有明确的职责
- 组件可重用和替换
- 易于扩展和维护

### 3. 针对文本分类优化
- 使用 [CLS] token 进行分类
- 支持注意力掩码处理变长序列
- 集成 Dropout 防止过拟合

## 使用方式

在主程序中使用模型的典型流程：
```python
# 创建模型实例
model = TransformerTextClassificationModel(
    vocab_size=len(vocab),
    d_model=256,
    num_heads=8,
    num_layers=4,
    d_ff=512,
    dropout=0.5,
    num_classes=2
)

# 训练模型
train_loss, train_acc = train_transformer_model(
    model, train_dataloader, criterion, optimizer, device, epoch, num_epochs
)

# 评估模型
val_loss, val_acc = evaluate_transformer_model(
    model, val_dataloader, criterion, device
)
```

## 总结

`transformer_model.py` 实现了一个完整的、可用于实际项目的 Transformer 文本分类模型。它包含了 Transformer 架构的所有关键组件，并针对文本分类任务进行了优化。通过模块化的设计，代码具有良好的可读性和可扩展性，便于进一步的开发和研究。