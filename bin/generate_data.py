#!/usr/bin/env python3
"""
生成对比学习示例用的合成向量数据并序列化为 pickle 文件。

输出: dataset.pkl，包含字典 { 'vectors': np.ndarray (N, D), 'labels': list or None }
"""
import os
import pickle
import numpy as np

def generate_synthetic_vectors(n_samples=1000, dim=128, n_clusters=10, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim) * 5.0
    labels = rng.randint(0, n_clusters, size=n_samples)
    vectors = centers[labels] + rng.randn(n_samples, dim) * 0.5
    return vectors.astype(np.float32), labels.tolist()

def main():
    out_file = 'dataset.pkl'
    vectors, labels = generate_synthetic_vectors(n_samples=1000, dim=128, n_clusters=10)
    data = {
        'project': 'Nextflow Pickle Demo - Contrastive',
        'vectors': vectors,
        'labels': labels,
        'meta': {'n_samples': vectors.shape[0], 'dim': vectors.shape[1]}
    }

    print(f"Generating synthetic dataset: samples={vectors.shape[0]}, dim={vectors.shape[1]}")
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved dataset to {os.path.abspath(out_file)}")


if __name__ == '__main__':
    main()
