# 分词器

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/index>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/index>


快速、最先进的分词器，针对研究和研究进行了优化
生产


[🤗 标记器](https://github.com/huggingface/tokenizers)
 提供了一个
实现当今最常用的分词器，重点是
性能和多功能性。这些标记器也用于
 [🤗 变形金刚](https://github.com/huggingface/transformers)
 。


# 主要特点：

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/index>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/index>


* 使用当今最常用的分词器来训练新词汇并进行分词。
* 由于 Rust 实现，速度极快（训练和标记化）。在服务器 CPU 上标记 1 GB 文本只需不到 20 秒。
* 易于使用，而且用途极其广泛。
* 专为研究和生产而设计。
* 完全对准跟踪。即使使用破坏性标准化，也始终可以获得原始句子中与任何标记相对应的部分。
* 执行所有预处理：截断、填充、添加模型需要的特殊标记。