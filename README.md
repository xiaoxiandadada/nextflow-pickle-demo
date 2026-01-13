# Nextflow 对比学习流水线 Demo

欢迎来到这个综合学习仓库！
这个项目的目标是帮助你同时掌握 **Nextflow** (流程编排) 和 **对比学习** (Contrastive Learning) 两个领域的知识。

## 📂 学习路径

我们将内容分为了三个部分，建议按需学习：

### 1. 🟢 基础：学习 Nextflow
如果你对 Nextflow 不熟悉，请先看这里。
进入 `learn_nextflow/` 文件夹。
*   **目标**: 掌握 Channel, Process, Workflow 基础语法。
*   **内容**: 包含 3 个由浅入深的 `.nf` 脚本（Hello World, Channel操作, 参数化）。
*   [前往学习](./learn_nextflow/README.md)

### 2. 🔵 基础：学习对比学习 (Contrastive Learning)
如果你想了解 SimCLR、InfoNCE Loss 背后的数学和代码实现（无 Nextflow 干扰）。
进入 `learn_contrastive_learning/` 文件夹。
*   **目标**: 理解正负样本对、InfoNCE Loss 公式、Projector 的作用。
*   **内容**: 包含纯 Numpy 算法演示和 PyTorch 简易版 SimCLR 实现。
*   [前往学习](./learn_contrastive_learning/README.md)

### 3. 🔴 进阶：综合实战 Demo
当你对两者都有所了解后，回到根目录（当前位置）。
这是一个结合了上述两者的完整流水线：
*   使用 Python 脚本编写对比学习逻辑（在 `bin/` 中）。
*   使用 Nextflow 编排整个“生成数据 -> 训练/处理”流程（`main.nf`）。
*   [查看实战教程](./NEXTFLOW_TUTORIAL.md)

---

## 快速实战 (根目录 Demo)

### 核心功能
1.  **数据生成**: 模拟高维向量聚类数据。
2.  **对比学习**:
    *   **数据增广**: 构造正样本对（Views）。
    *   **模型训练**: 包含一个轻量级的 PyTorch 训练循环演示（也支持纯 Numpy 模式）。
    *   **InfoNCE Loss**: 实现了对比损失函数的计算。
3.  **流程编排**: 使用 Nextflow 自动管理任务依赖和文件传递。

### 运行方法
确保你已经安装了 Nextflow (需 Java环境) 和 Python 3。

1.  安装 Python 依赖（推荐在虚拟环境中）：
    ```bash
    pip install -r requirements.txt
    ```

2.  运行流水线：
    ```bash
    nextflow run main.nf
    ```

## 教程与原理
请移步 [NEXTFLOW_TUTORIAL.md](./NEXTFLOW_TUTORIAL.md) 查看关于项目根目录 Demo 的详细解析。
