import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

"""
02_simclr_demo.py

一个极简的 SimCLR (Simple Framework for Contrastive Learning of Visual Representations) 风格实现。
为了无需下载 ImageNet/CIFAR，我们使用合成的随机向量数据。

主要包含组件：
1. Augementation: 一个简单的函数，给向量加噪声。
2. Encoder: 一个简单的 MLP。
3. Projection Head: 另一个 MLP，将特征映射到对比空间 (Metric Space)。
4. Training Loop: 最小化 InfoNCE Loss。
"""

class SimCLR_Model(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=64):
        super().__init__()
        # Encoder (ResNet backbone in real CV tasks)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Generator / Projection Head (z = g(h))
        # 论文指出这里加非线性变换很重要
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z

def nt_xent_loss(z, temperature=0.5):
    """
    z shape: [2 * BatchSize, D]
    假设 batch 排列是 [v1_a, v1_b, v2_a, v2_b, ...]
    其中 (2i, 2i+1) 是正样本对
    """
    N = z.shape[0]
    
    # 1. Cosine Similarity Matrix
    # 归一化特征
    z = nn.functional.normalize(z, dim=1)
    # [N, N] 矩阵
    sim = torch.mm(z, z.T) / temperature
    
    # 2. Mask out self-contrast (diagonal)
    mask_self = torch.eye(N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask_self, -9e15) # 用一个很大的负数代表负无穷
    
    # 3. 构造正样本的 Target Mask
    # 我们知道 0-1 是对，2-3 是对...
    # target[k] 应该是 sim[k] 中正样本的索引
    target = torch.arange(N, device=z.device)
    # 0->1, 1->0, 2->3, 3->2 ...
    # 如果 i 是偶数，正样本是 i+1；如果 i 是奇数，正样本是 i-1
    # 可以用异或技巧: i ^ 1
    target = target ^ 1
    
    # 4. Cross Entropy Loss
    # sim 是 logits, target 是类别标签
    loss = nn.functional.cross_entropy(sim, target)
    return loss

def run_demo():
    print("Running SimCLR Synthetic Demo...")
    # 配置
    BATCH_SIZE = 64
    INPUT_DIM = 128
    EPOCHS = 10
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化模型
    model = SimCLR_Model(input_dim=INPUT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"Training on {DEVICE} for {EPOCHS} epochs.")
    
    for epoch in range(EPOCHS):
        # 1. 生成 batch 数据 (模拟 data loader)
        # 生成 BATCH_SIZE 个“原图”
        x_base = torch.randn(BATCH_SIZE, INPUT_DIM).to(DEVICE)
        
        # 2. 数据增广 (Views)
        # 视图 1: 加一点点噪声
        x_i = x_base + 0.1 * torch.randn_like(x_base)
        # 视图 2: 加另一种噪声
        x_j = x_base + 0.1 * torch.randn_like(x_base)
        
        # 拼接: [x_i[0], x_j[0], x_i[1], x_j[1], ...]
        # 方便后续 loss 计算中 (2k, 2k+1) 是正样本
        x_input = torch.empty(2 * BATCH_SIZE, INPUT_DIM, device=DEVICE)
        x_input[0::2] = x_i
        x_input[1::2] = x_j
        
        # 3. Forward
        z = model(x_input)
        
        # 4. Loss & Backward
        loss = nt_xent_loss(z)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")
            
    print("Done! Model encoded embeddings should now cluster positive pairs together.")

if __name__ == "__main__":
    run_demo()
