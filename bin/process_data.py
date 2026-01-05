#!/usr/bin/env python3
import pickle
import sys

if len(sys.argv) < 2:
    print("Usage: process_data.py <pickle_file>")
    sys.exit(1)

input_file = sys.argv[1]

print(f"Python: Loading data from {input_file}...")

# 使用 Pickle 反序列化读取数据
with open(input_file, 'rb') as f:
    data = pickle.load(f)

print(f"Python: Data loaded successfully!")
print(f"Python: Content: {data}")

# 进行一些简单的计算
total = sum(data['raw_values'])
avg = total / len(data['raw_values'])

output_content = f"Total: {total}, Average: {avg}\nProject: {data['project']}"
print(f"Python: Processing result -> {output_content}")

# 将结果写入文本文件
with open('result.txt', 'w') as f:
    f.write(output_content)
