# 概述

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/loading_overview>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/loading_overview>


🧨 Diffusers 为生成任务提供了许多管道、模型和调度程序。为了使加载这些组件尽可能简单，我们提供了一个单一且统一的方法 -
 `from_pretrained()`
 - 从拥抱面加载任何这些组件
 [中心](https://huggingface.co/models?library=diffusers&sort=downloads)
 或您的本地计算机。每当您加载管道或模型时，都会自动下载并缓存最新文件，以便您下次可以快速重用它们，而无需重新下载文件。


本节将向您展示有关加载管道、如何加载管道中的不同组件、如何加载检查点变体以及如何加载社区管道所需了解的所有信息。您还将学习如何加载调度程序并比较使用不同调度程序的速度和质量权衡。最后，您将了解如何转换和加载 KerasCV 检查点，以便您可以在 PyTorch 中通过 🧨 Diffusers 使用它们。