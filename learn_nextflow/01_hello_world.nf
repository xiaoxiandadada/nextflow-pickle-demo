#!/usr/bin/env nextflow

// 必须启用 DSL2 语法
nextflow.enable.dsl=2

/*
 * 例子 1: 一个最简单的 Process
 * 它不接收 input，直接执行一段 bash 脚本。
 * 'output: stdout' 表示我们将捕获它打印到屏幕的内容。
 */
process SAY_HELLO {
    output:
    stdout

    script:
    """
    echo "Hello, Nextflow World!"
    """
}

/*
 * 例子 2: 稍微复杂一点的 Process
 * 接收一个字符串输入，将其转换为大写
 */
process TO_UPPER {
    input:
    val x

    output:
    stdout

    script:
    """
    echo "Processing: $x"
    # bash 命令
    echo "$x" | tr 'a-z' 'A-Z'
    """
}

workflow {
    // 1. 运行 SAY_HELLO
    // 它返回的结果可以通过变量 say_hello_out 访问
    say_hello_out = SAY_HELLO()

    // 2. 查看结果 (类似于 print)
    say_hello_out.view()

    // 3. 运行 TO_UPPER
    // 我们可以手动传入一个 Channel
    input_ch = Channel.of('apple', 'banana', 'cherry')
    upper_out = TO_UPPER(input_ch)
    
    // nextflow 自动并行化：你看，这里会并行处理 apple, banana, cherry
    upper_out.view()
}
