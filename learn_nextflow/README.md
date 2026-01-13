# Nextflow 基础学习路径

本目录包含逐步进阶的 Nextflow 脚本，带你从零开始理解这个强大的流程语言。
这里的例子比根目录的完整 Demo 更简单，更适合初学。

## 目录结构

*   `01_hello_world.nf`: **最简单的流程**。展示 `process` 的基本结构和标准输出捕获。
*   `02_channels.nf`: **理解 Channel**。Channel 是 Nextflow 的血脉，负责在进程间搬运数据。
*   `03_parameters.nf`: **参数化流程**。学习如何从命令行传递参数（`--input`）。

## 学习目标

1.  掌握 DSL2 (Nextflow 2.0) 的基本语法 `process` 和 `workflow`。
2.  理解数据是如何通过 `Channel` 自动触发下游任务的。
3.  学会如何运行脚本并查看工作目录 (`work/`) 来调试。

## 运行方法

确保已安装 Nextflow。

```bash
# 运行第一个脚本
nextflow run 01_hello_world.nf
```
