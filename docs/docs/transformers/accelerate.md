# 使用 🤗 Accelerate 进行分布式训练

> 译者：[BrightLi](https://github.com/brightli)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/accelerate>
>
> 原始地址：<https://huggingface.co/docs/transformers/accelerate>

随着模型变得越来越大，并行计算已经成为一种在有限的硬件上训练更大模型和加速训练速度几个数量级的策略。在 Hugging Face，我们创建了 🤗 [Accelerate](https://huggingface.co/docs/accelerate)库来帮助用户轻松地在任何类型的分布式设置上训练一个 🤗 Transformers 模型，无论是一台机器上的多个 GPU 还是多台机器上的多个 GPU。在本教程中，了解如何自定义你的原生 PyTorch 训练循环以启用分布式环境的训练。

**设置**

首先通过安装 🤗 Accelerate 开始：

```shell
pip install accelerate
```

