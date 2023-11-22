![](https://raw.githubusercontent.com/huggingface/diffusers/77aadfee6a891ab9fcfb780f87c693f7a5beeb8e/docs/source/imgs/diffusers_library.jpg)


# 扩散器

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/index>
>
> 原始地址：<https://huggingface.co/docs/diffusers/index>


🤗 Diffusers 是最先进的预训练扩散模型的首选库，用于生成图像、音频甚至分子的 3D 结构。无论您是在寻找简单的推理解决方案还是想要训练自己的扩散模型，🤗 Diffusers 都是一个支持两者的模块化工具箱。我们的图书馆的设计重点是
 [可用性优于性能]（概念/哲学#可用性优于性能）
 ,
 [简单胜于简单]（概念/哲学#简单胜于简单）
 ， 和
 [可定制性优于抽象]（概念/哲学#tweakable-contributorfriend-over-abstraction）
 。


该库包含三个主要组件：


* 只需几行代码即可使用最先进的扩散管道进行推理。 🤗 Diffusers 中有很多管道，查看管道中的表格
 [概述](api/管道/概述)
 获取可用管道及其解决的任务的完整列表。
* 可互换
 [噪声调度程序](api/调度程序/概述)
 用于平衡生成速度和质量之间的权衡。
* 预训练
 [模型](api/模型)
 它可以用作构建块，并与调度程序相结合，用于创建您自己的端到端扩散系统。


[教程
 

 学习开始生成输出、构建自己的扩散系统和训练扩散模型所需的基本技能。如果您是第一次使用 🤗 扩散器，我们建议您从这里开始！](./tutorials/tutorial_overview)
[操作指南
 

 帮助您加载管道、模型和调度程序的实用指南。您还将学习如何使用管道执行特定任务、控制输出的生成方式、优化推理速度以及不同的训练技术。](./using-diffusers/loading_overview)
[概念指南
 

 了解图书馆为何如此设计，并详细了解使用图书馆的道德准则和安全实施。](./概念/哲学)
[参考
 

 🤗 Diffusers 类和方法如何工作的技术描述。](./api/models/overview)