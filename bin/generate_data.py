#!/usr/bin/env python3
import pickle
import os

# 模拟生成一些复杂的数据结构
data = {
    'project': 'Nextflow Pickle Demo',
    'parameters': {'threshold': 0.5, 'iterations': 100},
    'raw_values': [10, 20, 30, 40, 50]
}

filename = 'dataset.pkl'

print(f"Python: Generating data object: {data}")

# 使用 Pickle 序列化数据到文件
with open(filename, 'wb') as f:
    pickle.dump(data, f)

print(f"Python: Data serialized to {os.path.abspath(filename)}")
