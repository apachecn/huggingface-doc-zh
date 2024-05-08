# PEFT

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/index>
>
> 原始地址：<https://huggingface.co/docs/peft/index>


🤗 PEFT（Parameter-Efficient Fine-Tuning）是一个库，用于有效地将大型预训练模型适应各种下游应用，而无需对所有模型参数进行微调，因为这是成本过高的。PEFT方法仅微调少量（额外的）模型参数 - 显著降低计算和存储成本 - 同时产生与完全微调模型相媲美的性能。这使得在消费者硬件上训练和存储大型语言模型（LLMs）更加容易。

PEFT已与Transformers、Diffusers和Accelerate库集成，提供了一种更快、更简便的加载、训练和使用大型模型进行推理的方式。



- [开始使用](https://huggingface.apachecn.org/docs/peft/quicktour/)  
 如果您是第一次接触 🤗 PEFT，请从这里开始，了解该库的主要特点以及如何使用 PEFT 方法训练模型。


- [操作指南]((./task_guides/image_classification_lora))
 实用指南展示了如何在不同类型的任务（如图像分类、因果语言建模、自动语音识别等）中应用各种 PEFT 方法。学习如何使用 🤗 PEFT 结合 DeepSpeed 和 Fully Sharded Data Parallel 脚本。

- [概念指南](./conceptual_guides/lora)
 深入理解 LoRA 和各种软提示方法如何帮助减少可训练参数数量，从而使训练更加高效

- [参考](./package_reference/config)
🤗 PEFT 类和方法运作的技术描述。

