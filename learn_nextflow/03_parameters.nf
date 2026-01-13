#!/usr/bin/env nextflow

nextflow.enable.dsl=2

/*
 * 学习参数化 (Parameters)
 * 你可以在命令行通过 --paramName value 覆盖 params.paramName
 */

// 定义默认参数
params.greeting = "Hello"
params.name = "World"

process GREET {
    input:
    val greeting_str
    val name_str
    
    output:
    stdout
    
    script:
    """
    echo "$greeting_str, $name_str!"
    """
}

workflow {
    // 可以直接使用 params.xxx
    println "Current configuration: Greeting=${params.greeting}, Name=${params.name}"
    
    // 将参数传入 Channel
    GREET(params.greeting, params.name).view()
}

/*
 * 尝试运行:
 * nextflow run 03_parameters.nf --name "Nextflow User" --greeting "Welcome"
 */
