"""
01_simple_infonce.py

这是一个最小化的 InfoNCE (Noise Contrastive Estimation) 损失函数演示。
不涉及复杂的神经网络，仅使用 NumPy 展示在给定了特征向量后，Loss 是如何计算的。

原理：
我们有一个 batch 的样本，大小为 N。通过数据增广，每个样本产生 2 个视图。
总共有 2N 个向量。
对于某个视图 z_i，它的正样本是源自同一个原图的 z_j。
其余 2N-2 个向量都是它的负样本。
我们希望 z_i 和 z_j 的余弦相似度最大，与负样本的相似度最小。

数学公式 (针对第 i 个样本):
loss_i = -log( exp(sim(i,j)/T) / sum_{k!=i} exp(sim(i,k)/T) )
"""

import numpy as np
import math

def softmax(x):
    """计算 softmax，减去 max 增加数值稳定性"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def info_nce_manual_step_by_step():
    print("=== Step 1: 模拟 Embeddings ===")
    BatchSize = 2
    Dim = 4
    Temperature = 0.5
    
    # 假设我们有两个原始图片 A, B
    # 生成它们的两个视图 (A1, A2, B1, B2) -> 总共 4 个向量
    # 我们希望 (A1, A2) 相似，(B1, B2) 相似
    # (A1, B*) 不相似
    
    # 随机初始化一些 normalized 向量
    embeddings = np.random.randn(2 * BatchSize, Dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    print(f"Embeddings shape: {embeddings.shape} (2*BatchSize, Dim)")
    print("Vectors:\n", embeddings)

    print("\n=== Step 2: 计算相似度矩阵 ===")
    # Sim(i, j) = z_i . z_j / Temperature
    similarity_matrix = np.dot(embeddings, embeddings.T) / Temperature
    print("Similarity Matrix (scaled):\n", similarity_matrix)

    print("\n=== Step 3: 计算 Loss (以第一个样本 A1 为例) ===")
    # A1 的索引是 0。它的正样本是 A2 (索引 1)。
    # 它的所有样本是 0, 1, 2, 3。
    # 我们需要排除它自己 (index 0)
    
    idx = 0
    pos_idx = 1
    
    # 获取 A1 与其他所有向量的 logits
    logits = similarity_matrix[0]
    
    # 分子：正样本对的 exp(sim)
    numerator = np.exp(logits[pos_idx])
    
    # 分母：所有【除自己以外】的 exp(sim) 之和
    # 在实际实现中，通常使用 mask 矩阵来高效操作
    denominator = 0.0
    for k in range(len(logits)):
        if k != idx:
            denominator += np.exp(logits[k])
            
    loss_0 = -np.log(numerator / denominator)
    print(f"Loss for sample 0 (manual): {loss_0:.4f}")

    print("\n=== Step 4: 矩阵化计算 (更像 PyTorch 的实现) ===")
    # 创建 mask: 对角线为 True (自己)，其余为 False
    N = 2 * BatchSize
    mask = np.eye(N, dtype=bool)
    
    # 将对角线(自己与自己的相似度)设为负无穷，这样在 softmax/exp 中会变为 0
    similarity_matrix[mask] = -1e9
    
    # 对每一行做 LogSoftmax
    # loss = - log_softmax(logits)[positive_index]
    # 但我们手动写一遍
    loss_total = 0
    for i in range(BatchSize):
        # 样本 i (索引 2*i) -> 正样本 (2*i+1)
        a = 2*i
        b = 2*i + 1
        
        # 计算针对 a 的损耗
        # log_prob = logits - log(sum(exp(logits)))
        row_a = similarity_matrix[a]
        log_sum_exp_a = np.log(np.sum(np.exp(row_a))) # 注意这里包含了刚才设为 -inf 的对角线项，exp(-inf)=0，正确
        log_prob_positive = row_a[b] - log_sum_exp_a
        loss_a = -log_prob_positive
        
        # 计算针对 b 的损耗
        row_b = similarity_matrix[b]
        log_sum_exp_b = np.log(np.sum(np.exp(row_b)))
        log_prob_positive_b = row_b[a] - log_sum_exp_b
        loss_b = -log_prob_positive_b
        
        print(f"Pair {i}: Loss({a}->{b})={loss_a:.4f}, Loss({b}->{a})={loss_b:.4f}")
        loss_total += (loss_a + loss_b)

    avg_loss = loss_total / N
    print(f"\nAverage Loss: {avg_loss:.4f}")
    
if __name__ == "__main__":
    info_nce_manual_step_by_step()
