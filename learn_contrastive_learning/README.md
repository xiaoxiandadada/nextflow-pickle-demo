# 对比学习 (Contrastive Learning) 学习路径

本目录包含更纯粹的对比学习实验代码，旨在帮助你理解不依赖于任何编排工具（如 Nextflow）的算法原理。

## 目录结构

*   `01_simple_infonce.py`: **核心原理**。一个最小化的 NumPy 实现，用于理解 InfoNCE 损失函数如何通过矩阵运算计算正负样本的相似度。
*   `02_simclr_demo.py`: **完整流程**。一个基于 PyTorch 的微型 SimCLR 实现。包含 Dataset 类、数据增广、ResNet/MLP 编码器和完整的训练循环（使用 CIFAR-10 或合成数据）。

## 学习目标

1.  理解 **正样本对** (Positive Pairs) 和 **负样本对** (Negative Pairs) 的构造。
2.  掌握 **InfoNCE Loss** (NT-Xent) 的数学含义与代码实现。
3.  了解 SimCLR 等框架中 **Projector (Projection Head)** 的作用。
