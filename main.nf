#!/usr/bin/env nextflow

nextflow.enable.dsl=2

/*
 * Process 1: 生成数据
 * 调用 Python 脚本生成一个 pickle 文件
 */
process GENERATE_PICKLE {
    // 定义输出：声明这个进程会生成一个名为 'dataset.pkl' 的文件
    // Nextflow 会捕获这个文件并放入 channel 中
    output:
    path 'dataset.pkl'

    script:
    """
    generate_data.py
    """
}

/*
 * Process 2: 处理数据
 * 接收上一个进程生成的 pickle 文件作为输入
 */
process PROCESS_PICKLE {
    // 定义输入：接收一个文件路径，我们在脚本中用变量名 ${pkl} 引用它
    input:
    path pkl

    // 定义输出：生成的结果文本
    output:
    path 'result.txt'

    // 打印执行信息到控制台
    debug true

    script:
    """
    # 这里 ${pkl} 是 Nextflow 暂存区中 pickle 文件的实际路径
    process_data.py ${pkl}
    """
}

/*
 * Workflow: 定义流程逻辑
 */
workflow {
    // 1. 运行生成步骤，返回一个 Channel (包含 dataset.pkl)
    pickle_ch = GENERATE_PICKLE()

    // 2. 将生成的 Channel 传递给处理步骤
    PROCESS_PICKLE(pickle_ch)
}
