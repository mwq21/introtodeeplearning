# Lab 1 实验报告：基于 PyTorch 的 LSTM 音乐生成
- GitHub Repository: [github](https://github.com/mwq21/introtodeeplearning)

- Comet Experiment: [Comet 实验记录链接](https://www.comet.com/mingwei-qiu/6s191-lab1-part2/view/new/panels)

## 项目目录结构
lab1/
│
├── PT_Part1_Intro.ipynb
├── PT_Part2_Music_Generation.ipynb
├── results/
│   ├── training_loss.png
│   └── generated_music.abc
└── REPORT.md

## 1. 实验内容

本实验基于 PyTorch 完成深度学习基础与序列生成任务。

主要包含两部分：

- Part 1：PyTorch 基础操作
- Part 2：基于 LSTM 的字符级音乐生成


## 2. Part 1：PyTorch 基础

完成内容：

- Tensor 创建与基本运算
- 神经网络构建方式：
  - 手动定义参数
  - `nn.Sequential`
  - 继承 `nn.Module` 自定义模型
- PyTorch 自动求导机制（Autograd）
- 基于梯度下降的参数优化

通过实验理解 PyTorch 中模型定义、前向传播、损失计算和反向传播的基本流程。


## 3. Part 2：LSTM 音乐生成

### 3.1 数据处理

实验使用 ABC notation 音乐数据。

首先建立字符词表，将音乐文本中的字符转换为整数索引，使其可以输入神经网络。

数据处理流程：

```
Music Text
    ↓
Character Vocabulary
    ↓
Integer Encoding
    ↓
Training Sequence
```


### 3.2 模型结构


模型结构：

```
Character Index
      ↓
Embedding
      ↓
LSTM
      ↓
Linear
      ↓
Character Probability
```

模型根据已有字符序列预测下一个字符。


### 3.3 训练方法

训练配置：

- Framework: PyTorch
- Model: LSTM
- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Iterations: 3000


训练流程：

```
Input Sequence
      ↓
LSTM Model
      ↓
Prediction
      ↓
Cross Entropy Loss
      ↓
Backward Propagation
      ↓
Parameter Update
```


## 4. 实验结果

### 4.1 Loss 曲线

训练过程中 Loss 逐渐下降，模型能够学习音乐字符序列中的规律。

![Training Loss](results/training_loss.png)


### 4.2 音乐生成结果

训练完成后，模型能够生成新的 ABC 格式音乐序列。

生成结果：

[generated_music.abc](results/generated_music.abc)


## 5. 实验总结

通过本实验：

- 掌握了 PyTorch 中 Tensor 和神经网络模型构建方法；
- 理解了 Autograd 自动求导和梯度下降过程；
- 学习了 LSTM 在序列建模任务中的应用；
- 完成了基于字符级预测的音乐生成任务。
