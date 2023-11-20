# 概述

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/optimization/opt_overview>
>
> 原始地址：<https://huggingface.co/docs/diffusers/optimization/opt_overview>


生成高质量输出需要大量计算，尤其是在从噪声输出到噪声较小输出的每个迭代步骤中。 🤗 Diffuser 的目标之一是让每个人都能广泛使用这项技术，其中包括实现对消费者和专用硬件的快速推理。


本节将介绍用于优化推理速度和减少内存消耗的提示和技巧，例如半精度权重和切片注意力。您还将学习如何使用以下命令加速 PyTorch 代码
 [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
 或者
 [ONNX 运行时](https://onnxruntime.ai/docs/)
 ，并通过以下方式实现记忆高效的注意力
 [xFormers](https://facebookresearch.github.io/xformers/)
 。还有在 Apple Silicon、Intel 或 Habana 处理器等特定硬件上运行推理的指南。