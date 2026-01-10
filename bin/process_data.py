#!/usr/bin/env python3
"""
process_data.py

加载由 `generate_data.py` 生成的向量数据并展示一个对比学习（contrastive learning）的小型示例：
- 构建两视图增广
- 生成正/负对
- 如果安装了 PyTorch，则做一个小的训练循环（CPU）来优化 InfoNCE（NT-Xent）损失
- 如果没有 PyTorch，则使用 numpy 做无训练的示例计算（保证脚本可在无深度学习库的环境运行）

用法:
    python process_data.py dataset.pkl
"""
import os
import sys
import pickle
import math
from pathlib import Path

try:
    import numpy as np
except Exception:
    print("Error: numpy is required. Please install it (e.g. pip install numpy)")
    raise

USE_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    USE_TORCH = True
except Exception:
    # 如果没有 torch，脚本会退回到 numpy-only 演示
    USE_TORCH = False


def augment_vector(v, rng=None):
    """对向量做简单增广：加噪声、随机缩放和 dropout"""
    if rng is None:
        rng = np.random.RandomState()
    v = v.copy()
    # 高斯噪声
    v = v + rng.normal(scale=0.1, size=v.shape)
    # 随机缩放
    scale = rng.uniform(0.9, 1.1)
    v = v * scale
    # 随机dropout
    mask = rng.binomial(1, 0.9, size=v.shape)
    v = v * mask
    return v


def build_pairs_numpy(vectors, rng=None):
    """为每个样本构建两视图并返回数组 (2N, D) 以及对应索引映射"""
    if rng is None:
        rng = np.random.RandomState(0)
    N, D = vectors.shape
    views = []
    for i in range(N):
        x = vectors[i]
        v1 = augment_vector(x, rng)
        v2 = augment_vector(x, rng)
        views.append(v1)
        views.append(v2)
    views = np.stack(views, axis=0).astype(np.float32)
    # 对应的正样本索引：对于 i (0..N-1), positives are (2*i, 2*i+1) and vice versa
    return views


def info_nce_numpy(embeddings, temperature=0.07):
    """简单的 InfoNCE 计算（无训练），输入 embeddings 形状为 (2N, D)
    正样本为相邻奇偶对 (0,1),(2,3),..."""
    # L2 归一化
    e = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    # 相似度矩阵
    sim = np.dot(e, e.T) / temperature
    # mask 去掉自己
    np.fill_diagonal(sim, -1e9)
    N2 = sim.shape[0]
    N = N2 // 2
    losses = []
    for i in range(N):
        a = 2 * i
        b = 2 * i + 1
        # 以 a 为 anchor，b 为正样本
        logits = sim[a]
        # 正样本的 logit
        positive = logits[b]
        # log softmax
        logsum = math.log(sum(math.exp(x) for j, x in enumerate(logits) if j != a))
        loss_a = - (positive - logsum)
        # 同理 b
        logits_b = sim[b]
        positive_b = logits_b[a]
        logsum_b = math.log(sum(math.exp(x) for j, x in enumerate(logits_b) if j != b))
        loss_b = - (positive_b - logsum_b)
        losses.append((loss_a + loss_b) / 2.0)
    return float(np.mean(losses))


if USE_TORCH:
    class SimpleEncoderTorch(nn.Module):
        def __init__(self, input_dim=128, proj_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, proj_dim)
            )

        def forward(self, x):
            return self.net(x)


    def nt_xent_loss(z, temperature=0.07):
        # z: (2N, D)
        z = nn.functional.normalize(z, dim=1)
        N2 = z.shape[0]
        N = N2 // 2
        sim = torch.matmul(z, z.T) / temperature
        mask = torch.eye(N2, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)
        losses = []
        for i in range(N):
            a = 2 * i
            b = 2 * i + 1
            logits_a = sim[a]  # similarities of a with all
            logits_b = sim[b]
            loss_a = - (logits_a[b] - torch.logsumexp(logits_a, dim=0))
            loss_b = - (logits_b[a] - torch.logsumexp(logits_b, dim=0))
            losses.append((loss_a + loss_b) / 2.0)
        return torch.stack(losses).mean()


    def train_torch(views, epochs=5, batch_size=128, lr=1e-3, device='cpu'):
        device = torch.device(device)
        N2, D = views.shape
        assert N2 % 2 == 0
        model = SimpleEncoderTorch(input_dim=D, proj_dim=64).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        data = torch.from_numpy(views)
        for epoch in range(epochs):
            perm = torch.randperm(data.shape[0] // 2)
            # shuffle pairs
            idx_pairs = (perm.unsqueeze(1) * 2).repeat(1, 2) + torch.tensor([0, 1])
            idx_pairs = idx_pairs.view(-1)
            shuffled = data[idx_pairs]
            total_loss = 0.0
            model.train()
            for i in range(0, shuffled.shape[0], batch_size):
                batch = shuffled[i:i+batch_size].to(device)
                z = model(batch)
                loss = nt_xent_loss(z)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * batch.shape[0]
            avg_loss = total_loss / shuffled.shape[0]
            print(f"Epoch {epoch+1}/{epochs} avg_loss={avg_loss:.4f}")
        return model
else:
    # Placeholders to avoid NameError in numpy-only environments
    def train_torch(*args, **kwargs):
        raise RuntimeError("Torch is not available in this environment")


def main():
    if len(sys.argv) < 2:
        print("Usage: process_data.py <pickle_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    print(f"Loading data from {input_file}...")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    vectors = data.get('vectors')
    if vectors is None:
        print("Error: dataset does not contain 'vectors'.")
        sys.exit(1)

    vectors = np.asarray(vectors)
    print(f"Loaded {vectors.shape[0]} samples, dim={vectors.shape[1]}")

    # 构建两视图
    views = build_pairs_numpy(vectors, rng=np.random.RandomState(0))

    result_lines = []
    if USE_TORCH:
        print("Torch detected: running small training loop (CPU).")
        model = train_torch(views, epochs=3, batch_size=256, lr=1e-3, device='cpu')
        # 计算最终 embeddings 并写入示例统计
        with torch.no_grad():
            emb = model(torch.from_numpy(views)).cpu().numpy()
        loss_val = info_nce_numpy(emb)
        result_lines.append(f"Torch training done. Post-train InfoNCE (numpy eval): {loss_val:.6f}")
    else:
        print("Torch not available: running numpy-only demo (no training).")
        # 用随机线性投影作为 encoder 的替代
        rng = np.random.RandomState(1)
        proj = rng.normal(scale=0.1, size=(views.shape[1], 64)).astype(np.float32)
        emb = views.dot(proj)
        loss_val = info_nce_numpy(emb)
        result_lines.append(f"Numpy demo InfoNCE: {loss_val:.6f}")

    out_text = '\n'.join([f"Project: {data.get('project')}"] + result_lines)
    print(out_text)
    with open('result.txt', 'w') as f:
        f.write(out_text + '\n')


if __name__ == '__main__':
    main()
