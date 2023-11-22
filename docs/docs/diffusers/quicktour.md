![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


# 快速游览

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/quicktour>
>
> 原始地址：<https://huggingface.co/docs/diffusers/quicktour>


扩散模型经过训练，可以逐步对随机高斯噪声进行去噪，以生成感兴趣的样本，例如图像或音频。这引发了人们对生成人工智能的巨大兴趣，并且您可能已经在互联网上看到了扩散生成图像的示例。 🧨 Diffusers 是一个旨在让每个人都能广泛使用扩散模型的库。


无论您是开发人员还是日常用户，此快速教程都会向您介绍 🧨 Diffusers，并帮助您快速入门和生成！该库需要了解三个主要组件：


* 这
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 是一个高级端到端类，旨在从预训练的扩散模型快速生成样本以进行推理。
* 流行的预训练
 [模型](./api/models)
 可用作创建扩散系统的构建块的架构和模块。
* 很多不同
 [调度程序](./api/schedulers/overview)
 - 控制如何在训练中添加噪声以及如何在推理过程中生成去噪图像的算法。


快速浏览将向您展示如何使用
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 进行推理，然后引导您了解如何结合模型和调度程序来复制内部发生的情况
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 。


Quicktour 是介绍性 🧨 扩散器的简化版本
 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)
 帮助您快速入门。如果您想了解更多关于 🧨 Diffusers 的目标、设计理念以及其核心 API 的其他详细信息，请查看笔记本！


在开始之前，请确保已安装所有必需的库：



```
# uncomment to install the necessary libraries in Colab
#!pip install --upgrade diffusers accelerate transformers
```


* [🤗 加速](https://huggingface.co/docs/accelerate/index)
 加快推理和训练的模型加载速度。
* [🤗 变形金刚](https://huggingface.co/docs/transformers/index)
 需要运行最流行的扩散模型，例如
 [稳定扩散](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview)
 。


## 扩散管道



这
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 是使用预训练扩散系统进行推理的最简单方法。它是一个包含模型和调度程序的端到端系统。您可以使用
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 对于许多任务来说，开箱即用。请查看下表，了解一些支持的任务，有关支持的任务的完整列表，请查看
 [🧨 扩散器摘要](./api/pipelines/overview#diffusers-summary)
 桌子。


| **Task**  | **Description**  | **Pipeline**  |
| --- | --- | --- |
| 	 Unconditional Image Generation	  | 	 generate an image from Gaussian noise	  | [unconditional\_image\_generation](./using-diffusers/unconditional_image_generation)  |
| 	 Text-Guided Image Generation	  | 	 generate an image given a text prompt	  | [conditional\_image\_generation](./using-diffusers/conditional_image_generation)  |
| 	 Text-Guided Image-to-Image Translation	  | 	 adapt an image guided by a text prompt	  | [img2img](./using-diffusers/img2img)  |
| 	 Text-Guided Image-Inpainting	  | 	 fill the masked part of an image given the image, the mask and a text prompt	  | [inpaint](./using-diffusers/inpaint)  |
| 	 Text-Guided Depth-to-Image Translation	  | 	 adapt parts of an image guided by a text prompt while preserving structure via depth estimation	  | [depth2img](./using-diffusers/depth2img)  |


首先创建一个实例
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 并指定您要下载的管道检查点。
您可以使用
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 对于任何
 [检查点](https://huggingface.co/models?library=diffusers&sort=downloads)
 存储在 Hugging Face Hub 上。
在此快速浏览中，您将加载
 [`stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 文本到图像生成的检查点。


为了
 [稳定扩散](https://huggingface.co/CompVis/stable-diffusion)
 型号请仔细阅读
 [许可证](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
 首先在运行模型之前。 🧨 扩散器实现了
 [`safety_checker`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py)
 以防止攻击性或有害内容，但该模型改进的图像生成功能仍然可以产生潜在的有害内容。


加载模型
 [来自\_pretrained()](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained)
 方法：



```
>>> from diffusers import DiffusionPipeline

>>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
```


这
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 下载并缓存所有建模、标记化和调度组件。您会看到稳定扩散管道由以下部分组成
 [UNet2DConditionModel](/docs/diffusers/v0.23.1/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)
 和
 [PNDMScheduler](/docs/diffusers/v0.23.1/en/api/schedulers/pndm#diffusers.PNDMScheduler)
 除其他事项外：



```
>>> pipeline
StableDiffusionPipeline {
  "\_class\_name": "StableDiffusionPipeline",
  "\_diffusers\_version": "0.21.4",
  ...,
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  ...,
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```


我们强烈建议在 GPU 上运行管道，因为该模型由大约 14 亿个参数组成。
您可以将生成器对象移动到 GPU，就像在 PyTorch 中一样：



```
>>> pipeline.to("cuda")
```


现在您可以将文本提示传递给
 `管道`
 生成图像，然后访问去噪图像。默认情况下，图像输出包装在
 [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class)
 目的。



```
>>> image = pipeline("An image of a squirrel in Picasso style").images[0]
>>> image
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/image_of_squirrel_painting.png)


 通过调用保存图像
 `保存`
 :



```
>>> image.save("image\_of\_squirrel\_painting.png")
```


### 


 本地管道


您还可以在本地使用管道。唯一的区别是您需要先下载权重：



```
!git lfs install
!git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```


然后将保存的权重加载到管道中：



```
>>> pipeline = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5", use_safetensors=True)
```


现在，您可以像上一节中那样运行管道。


### 


 交换调度程序


不同的调度器具有不同的去噪速度和质量权衡。找出哪一种最适合您的最佳方法就是尝试一下！ 🧨 Diffusers 的主要功能之一是允许您轻松地在调度程序之间切换。例如，要替换默认的
 [PNDMScheduler](/docs/diffusers/v0.23.1/en/api/schedulers/pndm#diffusers.PNDMScheduler)
 与
 [EulerDiscreteScheduler](/docs/diffusers/v0.23.1/en/api/schedulers/euler#diffusers.EulerDiscreteScheduler)
 ，加载它
 [from\_config()](/docs/diffusers/v0.23.1/en/api/configuration#diffusers.ConfigMixin.from_config)
 方法：



```
>>> from diffusers import EulerDiscreteScheduler

>>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
>>> pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
```


尝试使用新的调度程序生成图像，看看您是否注意到差异！


在下一节中，您将仔细研究组成模型和调度程序的组件
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 并学习如何使用这些组件生成猫的图像。


## 楷模



大多数模型都会采用噪声样本，并在每个时间步预测
 *残留噪音*
 （其他模型学习直接预测先前的样本或速度或
 [`v-预测`](https://github.com/huggingface/diffusers/blob/5e5ce13e2f89ac45a0066cb3f369462a3cf1d9ef/src/diffusers/schedulers/scheduling_ddim.py#L110)
 ），噪声较小的图像与输入图像之间的差异。您可以混合搭配模型来创建其他扩散系统。


模型是由
 [来自_pretrained()](/docs/diffusers/v0.23.1/en/api/models/overview#diffusers.ModelMixin.from_pretrained)
 方法还会在本地缓存模型权重，因此下次加载模型时速度会更快。对于快速浏览，您将加载
 [UNet2DModel](/docs/diffusers/v0.23.1/en/api/models/unet2d#diffusers.UNet2DModel)
 ，一个基本的无条件图像生成模型，带有在猫图像上训练的检查点：



```
>>> from diffusers import UNet2DModel

>>> repo_id = "google/ddpm-cat-256"
>>> model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)
```


要访问模型参数，请调用
 `模型.config`
 :



```
>>> model.config
```


模型配置是一个🧊冻结🧊字典，这意味着这些参数在模型创建后无法更改。这是有意为之，并确保一开始用于定义模型架构的参数保持不变，而其他参数仍然可以在推理过程中进行调整。


一些最重要的参数是：


* `样本大小`
 ：输入样本的高度和宽度尺寸。
* `in_channels`
 ：输入样本的输入通道数。
* `down_block_types`
 和
 `up_block_types`
 ：用于创建 UNet 架构的下采样和上采样模块的类型。
* `block_out_channels`
 ：下采样块的输出通道数；也以相反的顺序用于上采样块的输入通道的数量。
* `每块层数`
 ：每个 UNet 块中存在的 ResNet 块的数量。


要使用模型进行推理，请使用随机高斯噪声创建图像形状。它应该有一个
 `批量`
 轴，因为模型可以接收多个随机噪声，
 `频道`
 轴对应于输入通道的数量，以及
 `样本大小`
 图像的高度和宽度轴：



```
>>> import torch

>>> torch.manual_seed(0)

>>> noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
>>> noisy_sample.shape
torch.Size([1, 3, 256, 256])
```


为了进行推理，传递噪声图像和
 `时间步长`
 到模型。这
 `时间步长`
 表示输入图像的噪声程度，开始时噪声较多，结尾时噪声较少。这有助于模型确定其在扩散过程中的位置，无论是更接近开始还是结束。使用
 `样本`
 获取模型输出的方法：



```
>>> with torch.no_grad():
...     noisy_residual = model(sample=noisy_sample, timestep=2).sample
```


不过，要生成实际示例，您需要一个调度程序来指导去噪过程。在下一节中，您将学习如何将模型与调度程序结合起来。


## 调度程序



给定模型输出，调度程序管理从噪声样本到噪声较小的样本 - 在本例中，它是
 `嘈杂的残差`
 。


🧨 Diffusers 是用于构建扩散系统的工具箱。虽然
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 是开始使用预构建扩散系统的便捷方法，您还可以单独选择自己的模型和调度程序组件来构建自定义扩散系统。


对于快速浏览，您将实例化
 [DDPMScheduler](/docs/diffusers/v0.23.1/en/api/schedulers/ddpm#diffusers.DDPMScheduler)
 以其
 [from\_config()](/docs/diffusers/v0.23.1/en/api/configuration#diffusers.ConfigMixin.from_config)
 方法：



```
>>> from diffusers import DDPMScheduler

>>> scheduler = DDPMScheduler.from_pretrained(repo_id)
>>> scheduler
DDPMScheduler {
  "\_class\_name": "DDPMScheduler",
  "\_diffusers\_version": "0.21.4",
  "beta\_end": 0.02,
  "beta\_schedule": "linear",
  "beta\_start": 0.0001,
  "clip\_sample": true,
  "clip\_sample\_range": 1.0,
  "dynamic\_thresholding\_ratio": 0.995,
  "num\_train\_timesteps": 1000,
  "prediction\_type": "epsilon",
  "sample\_max\_value": 1.0,
  "steps\_offset": 0,
  "thresholding": false,
  "timestep\_spacing": "leading",
  "trained\_betas": null,
  "variance\_type": "fixed\_small"
}
```


💡 与模型不同，调度程序没有可训练的权重并且是无参数的！


一些最重要的参数是：


* `num_train_timesteps`
 ：去噪过程的长度，或者换句话说，将随机高斯噪声处理为数据样本所需的时间步数。
* `beta_schedule`
 ：用于推理和训练的噪声计划类型。
* `测试版开始`
 和
 `测试版结束`
 ：噪声表的开始和结束噪声值。


要预测噪声稍低的图像，请将以下内容传递给调度程序
 [step()](/docs/diffusers/v0.23.1/en/api/schedulers/ddpm#diffusers.DDPMScheduler.step)
 方法：模型输出，
 `时间步长`
 ，和当前的
 `样本`
 。



```
>>> less_noisy_sample = scheduler.step(model_output=noisy_residual, timestep=2, sample=noisy_sample).prev_sample
>>> less_noisy_sample.shape
torch.Size([1, 3, 256, 256])
```


这
 `less_noisy_sample`
 可以传递给下一个
 `时间步长`
 那里的噪音会更小！现在让我们将它们放在一起并可视化整个去噪过程。


首先，创建一个函数，对去噪图像进行后处理并将其显示为
 `PIL.Image`
 :



```
>>> import PIL.Image
>>> import numpy as np


>>> def display\_sample(sample, i):
...     image_processed = sample.cpu().permute(0, 2, 3, 1)
...     image_processed = (image_processed + 1.0) * 127.5
...     image_processed = image_processed.numpy().astype(np.uint8)

...     image_pil = PIL.Image.fromarray(image_processed[0])
...     display(f"Image at step {i}")
...     display(image_pil)
```


为了加速去噪过程，请将输入和模型移至 GPU：



```
>>> model.to("cuda")
>>> noisy_sample = noisy_sample.to("cuda")
```


现在创建一个去噪循环来预测噪声较小的样本的残差，并使用调度程序计算噪声较小的样本：



```
>>> import tqdm

>>> sample = noisy_sample

>>> for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
...     # 1. predict noise residual
...     with torch.no_grad():
...         residual = model(sample, t).sample

...     # 2. compute less noisy image and set x\_t -> x\_t-1
...     sample = scheduler.step(residual, t, sample).prev_sample

...     # 3. optionally look at image
...     if (i + 1) % 50 == 0:
...         display_sample(sample, i + 1)
```


坐下来看看猫是从噪音中诞生的！ 😻


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/diffusion-quicktour.png)


## 下一步



希望您在本快速教程中使用 🧨 扩散器生成了一些很酷的图像！对于后续步骤，您可以：


* 训练或微调模型以生成您自己的图像
 [培训](./tutorials/basic_training)
 教程。
* 示例参见官方和社区
 [训练或微调脚本](https://github.com/huggingface/diffusers/tree/main/examples#-diffusers-examples)
 适用于各种用例。
* 了解有关加载、访问、更改和比较调度程序的更多信息
 [使用不同的调度程序](./using-diffusers/schedulers)
 指导。
* 探索即时工程、速度和内存优化，以及使用生成更高质量图像的提示和技巧
 [稳定扩散](./stable_diffusion)
 指导。
* 深入了解加速 🧨 扩散器并开启指南
 [GPU 上优化的 PyTorch](./optimization/fp16)
 ，以及运行的推理指南
 [Apple Silicon 上的稳定扩散 (M1/M2)](./optimization/mps)
 和
 [ONNX 运行时](./optimization/onnx)
 。