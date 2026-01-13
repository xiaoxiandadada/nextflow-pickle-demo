#!/usr/bin/env nextflow

nextflow.enable.dsl=2

/*
 * 学习 Channel (通道)
 * Channel 是 Nextflow 中数据流动的管道。
 * Process 就像是管道上的阀门/处理器，它必须等待 Channel 送来数据才会启动。
 */

process CONSUME_DATA {
    input:
    val x

    script:
    """
    echo "Consumed: $x"
    sleep 1 # 模拟耗时任务
    """
}

workflow {
    // 1. Value Channel (单值通道)
    // 这种通道可以被反复读取无限次，通常用于传递配置或参数
    // 但在 input 声明为 val 时表现略有不同，这里主要演示 Queue Channel
    
    // 2. Queue Channel (队列通道)
    // 这是最常用的。它像一个队列，数据取一个少一个。
    // Process 会并行消费队列里的数据。
    
    data_ch = Channel.of(1, 2, 3, 4, 5)

    println "Created channel with 5 items."
    
    // 将 channel 喂给进程
    // 观察终端输出的顺序：它们可能是乱序的！因为是并行执行。
    CONSUME_DATA(data_ch)
    
    // 3. Channel 的操作符 (Operators)
    // Nextflow 提供了类似流式编程的操作符 (map, filter, view...)
    
    println "--- Channel Manipulation ---"
    
    Channel.of(1, 2, 3, 4, 5)
        .filter { it % 2 == 0 }     // 只保留偶数
        .map { it * 10 }            // 乘以 10
        .view { "Result: $it" }     // 打印
}
