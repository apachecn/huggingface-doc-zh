# 哲学

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/conceptual/philosophy>
>
> 原始地址：<https://huggingface.co/docs/diffusers/conceptual/philosophy>


🧨 扩散器提供
 **最先进的**
 跨多种模式的预训练扩散模型。
其目的是作为
 **模块化工具箱**
 用于推理和训练。


我们的目标是构建一个经得起时间考验的库，因此非常重视 API 设计。


简而言之，Diffusers 是 PyTorch 的自然扩展。因此，我们大部分的设计选择都是基于
 [PyTorch 的设计原则](https://pytorch.org/docs/stable/community/design.html#pytorch-design-philosophy)
 。让我们回顾一下最重要的：


## 可用性高于性能



* 虽然 Diffusers 有许多内置的性能增强功能（请参阅
 [内存和速度](https://huggingface.co/docs/diffusers/optimization/fp16)
 ），模型总是以最高的精度和最低的优化加载。因此，默认情况下，如果用户没有另外定义，扩散管道始终以 float32 精度在 CPU 上实例化。这确保了跨不同平台和加速器的可用性，并且意味着不需要复杂的安装来运行该库。
* 扩散器旨在成为
 **轻的**
 包，因此所需的依赖项很少，但有许多可以提高性能的软依赖项（例如
 ‘加速’
 ,
 `安全张量`
 ,
 `onnx`
 ， ETC…）。我们努力使库尽可能轻量级，以便可以添加它而不必担心对其他包的依赖。
* 传播者更喜欢简单的、不言自明的代码，而不是简洁的、神奇的代码。这意味着通常不需要 lambda 函数和高级 PyTorch 运算符等简写代码语法。


## 简单胜于简单



正如 PyTorch 所说，
 **显式优于隐式**
 和
 **简单胜于复杂**
 。这种设计理念体现在图书馆的多个部分：


* 我们遵循 PyTorch 的 API，使用以下方法
 [`DiffusionPipeline.to`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.to)
 让用户处理设备管理。
* 提出简洁的错误消息优于默默地纠正错误的输入。 Diffusers 的目的是教导用户，而不是让库尽可能易于使用。
* 复杂的模型与调度程序逻辑是公开的，而不是在内部神奇地处理。调度器/采样器与扩散模型分开，彼此之间的依赖性最小。这迫使用户编写展开的去噪循环。然而，这种分离可以更轻松地进行调试，并使用户能够更好地控制适应去噪过程或切换扩散模型或调度程序。
* 扩散管道的单独训练组件，
 *例如。*
 文本编码器、unet 和变分自动编码器都有自己的模型类。这迫使用户处理不同模型组件之间的交互，并且序列化格式将模型组件分离到不同的文件中。但是，这可以更轻松地进行调试和定制。 DreamBooth 或文本倒置训练
由于扩散器能够分离扩散管道的单个组件，因此非常简单。


## 可调整、对贡献者友好的抽象



对于图书馆的大部分区域，Diffusers 采用了一个重要的设计原则：
 [变形金刚库](https://github.com/huggingface/transformers)
 ，即更喜欢复制粘贴的代码而不是仓促的抽象。这种设计原则非常固执己见，与流行的设计原则形成鲜明对比，例如
 [不要重复自己（DRY）](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)
 。
简而言之，就像 Transformers 对建模文件所做的那样，Diffusers 更喜欢为管道和调度程序保留极低的抽象级别和非常独立的代码。
函数、长代码块甚至类都可以跨多个文件复制，乍一看这看起来像是一个糟糕的、草率的设计选择，使得库无法维护。
 **然而**
 事实证明，这种设计对于 Transformers 来说非常成功，对于社区驱动的开源机器学习库也很有意义，因为：


* 机器学习是一个发展极其迅速的领域，范式、模型架构和算法都在快速变化，因此很难定义持久的代码抽象。
* 机器学习从业者希望能够快速调整现有代码以进行构思和研究，因此更喜欢独立的代码而不是包含许多抽象的代码。
* 开源库依赖于社区贡献，因此必须构建一个易于贡献的库。代码越抽象，依赖关系越多，越难阅读，也越难贡献。由于担心破坏重要功能，贡献者只是停止为非常抽象的库做出贡献。如果对库的贡献不能破坏其他基本代码，那么它不仅对潜在的新贡献者更具吸引力，而且也更容易并行审查和贡献多个部分。


在 Hugging Face，我们将这种设计称为
 **单文件策略**
 这意味着某个类的几乎所有代码都应该编写在一个独立的文件中。要了解更多有关哲学的内容，您可以看看
在
 [这篇博文](https://huggingface.co/blog/transformers-design-philosophy)
 。


在扩散器中，我们对管道和调度程序都遵循这一理念，但仅部分针对扩散模型。我们没有完全遵循扩散模型的这种设计的原因是因为几乎所有的扩散管道，例如
作为
 [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)
 ,
 [稳定扩散](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview#stable-diffusion-pipelines)
 ,
 [unCLIP (DALL·E 2)](https://huggingface.co/docs/diffusers/api/pipelines/unclip)
 和
 [图像](https://imagen.research.google/)
 都依赖于相同的扩散模型，
 [UNet](https://huggingface.co/docs/diffusers/api/models/unet2d-cond)
 。


太好了，现在您应该大致了解为什么🧨 Diffusers 是这样设计的 🤗。
我们尝试在整个图书馆中一致地应用这些设计原则。尽管如此，该理念还是有一些小的例外或一些不幸的设计选择。如果您对设计有任何反馈，我们会❤️听取它
 [直接在 GitHub 上](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feedback.md&title=)
 。


## 细节设计理念



现在，让我们来看看设计理念的具体细节。扩散器本质上由三个主要类别组成：
 [管道](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines)
 ,
 [模型](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models)
 ， 和
 [调度程序](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers)
 。
让我们来看看每个类的更详细的设计决策。


### 


 管道


管道被设计为易于使用（因此不要遵循
 [*简单胜于简单*](#简单胜于简单)
 100%），功能不完整，应该大致被视为如何使用的示例
 [型号](#型号)
 和
 [调度程序](#schedulers)
 供推论。


遵循以下设计原则：


* 管道遵循单文件策略。所有管道都可以在 src/diffusers/pipelines 下的各个目录中找到。一个 pipeline 文件夹对应一份扩散论文/项目/版本。多个管道文件可以收集在一个管道文件夹中，就像这样做的那样
 [`src/diffusers/pipelines/stable-diffusion`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion)
 。如果管道共享相似的功能，则可以利用
 [#从机制复制](https://github.com/huggingface/diffusers/blob/125d783076e5bd9785beb05367a2d2566843a271/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L251)
 。
* 管道全部继承自
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 。
* 每个管道由不同的模型和调度程序组件组成，这些组件记录在
 [`model_index.json`
 文件](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/model_index.json)
 ，可以使用与管道属性相同的名称进行访问，并且可以在管道之间共享
 [`DiffusionPipeline.components`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.components)
 功能。
* 每个管道都应该可以通过
 [`DiffusionPipeline.from_pretrained`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained)
 功能。
* 应使用管道
 **仅有的**
 供推论。
* 管道应该非常可读、不言自明并且易于调整。
* 管道应设计为相互构建，并且易于集成到更高级别的 API 中。
* 管道是
 **不是**
 旨在成为功能完整的用户界面。对于未来完整的用户界面，人们应该看看
 [InvokeAI](https://github.com/invoke-ai/InvokeAI)
 ,
 [扩散器](https://github.com/abhishekrthakur/diffuzers)
 ， 和
 [喇嘛清洁器](https://github.com/Sanster/lama-cleaner)
 。
* 每个管道应该有一种且唯一的一种方式来通过
 `__call__`
 方法。的命名
 `__call__`
 参数应该在所有管道之间共享。
* 管道应以其要解决的任务命名。
* 在几乎所有情况下，新颖的扩散管道应在新的管道文件夹/文件中实现。


### 


 楷模


模型被设计为可配置的工具箱，是
 [PyTorch 的模块类](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
 。他们只部分遵循
 **单文件策略**
 。


遵循以下设计原则：


* 型号对应
 **一种模型架构**
 。
 *例如。*
 这
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)
 类用于所有需要 2D 图像输入并以某些上下文为条件的 UNet 变体。
* 所有型号均可在
 [`src/diffusers/models`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models)
 每个模型架构都应在其文件中定义，例如
 [`unet_2d_condition.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py)
 ,
 [`transformer_2d.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformer_2d.py)
 ， ETC…
* 楷模
 **不要**
 遵循单文件策略并应使用较小的模型构建块，例如
 [`attention.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py)
 ,
 [`resnet.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py)
 ,
 [`embeddings.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py)
 ， ETC…
 **笔记**
 ：这与 Transformers 的建模文件形成鲜明对比，表明模型并未真正遵循单文件策略。
* 模型旨在暴露复杂性，就像 PyTorch 的那样
 `模块`
 类，并给出明确的错误消息。
* 模型全部继承自
 `模型混合`
 和
 `配置混合`
 。
* 当不需要重大代码更改、保持向后兼容性并提供显着的内存或计算增益时，可以优化模型的性能。
* 默认情况下，模型应具有最高精度和最低性能设置。
* 为了集成新的模型检查点（其总体架构可以归类为 Diffusers 中已存在的架构），应调整现有模型架构以使其与新检查点一起工作。仅当模型架构根本不同时才应创建一个新文件。
* 模型的设计应该能够轻松扩展以适应未来的变化。这可以通过限制公共函数参数、配置参数和“预见”未来的变化来实现，
 *例如。*
 通常最好添加
 `字符串`
 “…type”参数可以轻松扩展到未来的新类型，而不是布尔值
 `是_..._类型`
 论据。只需对现有架构进行最小量的更改即可使新模型检查点发挥作用。
* 模型设计是保持代码可读性和简洁性与支持许多模型检查点之间的艰难权衡。对于建模代码的大部分部分，类应适应新的模型检查点，但也有一些例外，最好添加新类以确保代码保持简洁和简洁。
长期可读，例如
 [UNet 块](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_blocks.py)
 和
 [注意力处理器](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
 。


### 


 调度程序


调度程序负责指导推理的去噪过程以及定义训练的噪声时间表。它们被设计为具有可加载配置文件的单独类，并严格遵循
 **单文件策略**
 。


遵循以下设计原则：


* 所有调度程序都可以在
 [`src/diffusers/schedulers`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers)
 。
* 调度程序是
 **不是**
 允许从大型 utils 文件导入，并且应保持非常独立。
* 一个调度程序 Python 文件对应于一种调度程序算法（可能在一篇论文中定义）。
* 如果调度程序共享类似的功能，我们可以利用
 `#复制自`
 机制。
* 调度器全部继承自
 `调度器混合`
 和
 `配置混合`
 。
* 调度程序可以很容易地更换
 [`ConfigMixin.from_config`](https://huggingface.co/docs/diffusers/main/en/api/configuration#diffusers.ConfigMixin.from_config)
 详细解释的方法
 [此处](../using-diffusers/schedulers.md)
 。
* 每个调度器都必须有一个
 `set_num_inference_steps`
 ，和一个
 `步骤`
 功能。
 `set_num_inference_steps(...)`
 必须在每个去噪过程之前调用，
 *IE。*
 前
 `步骤(...)`
 叫做。
* 每个调度程序都会通过一个方法公开要“循环”的时间步长
 `时间步长`
 属性，它是模型将被调用的时间步数组。
* 这
 `步骤(...)`
 函数采用预测模型输出和“当前”样本 (x\_t)，并返回“前一个”、降噪程度稍高的样本 (x\_t-1)。
* 考虑到扩散调度器的复杂性，
 `步骤`
 函数并没有暴露所有的复杂性，并且可能有点像“黑匣子”。
* 在几乎所有情况下，新颖的调度程序都应在新的调度文件中实现。