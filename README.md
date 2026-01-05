# Nextflow + Pickle Demo

这个仓库演示了如何在 Nextflow 流程中使用 Python Pickle 文件在不同进程（Process）之间传递复杂数据。

## 结构

*   `bin/`: 包含 Python 脚本。Nextflow 会自动将此目录添加到 `$PATH` 中。
    *   `generate_data.py`: 创建字典对象并保存为 `.pkl`。
    *   `process_data.py`: 读取 `.pkl` 并进行计算。
*   `main.nf`: Nextflow 流程定义。

## 运行方法

确保你已经安装了 Nextflow 和 Java。

```bash
nextflow run main.nf
```

## 原理

1.  **GENERATE_PICKLE**: 运行 `generate_data.py`，生成 `dataset.pkl`。Nextflow 识别到 `output: path 'dataset.pkl'`，将其放入 Channel。
2.  **PROCESS_PICKLE**: 从 Channel 接收文件。Nextflow 会将文件暂存（Stage）到该进程的工作目录中。脚本通过 `process_data.py ${pkl}` 读取它。
