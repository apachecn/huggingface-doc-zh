# 了解管道、模型和调度程序

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/write_own_pipeline>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


🧨 Diffusers 被设计为一个用户友好且灵活的工具箱，用于构建适合您的用例的扩散系统。该工具箱的核心是模型和调度程序。虽然
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 为了方便起见，将这些组件捆绑在一起，您还可以解绑管道并分别使用模型和调度程序来创建新的扩散系统。


在本教程中，您将学习如何使用模型和调度程序来组装用于推理的扩散系统，从基本管道开始，然后逐步发展到稳定扩散管道。


## 解构基本管道



管道是一种运行模型进行推理的快速而简单的方法，只需不超过四行代码即可生成图像：



```
>>> from diffusers import DDPMPipeline

>>> ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")
>>> image = ddpm(num_inference_steps=25).images[0]
>>> image
```


![从 DDMPPipeline 创建的猫的图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ddpm-cat.png)


 这非常简单，但是管道是如何做到的呢？让我们分解一下管道，看看幕后发生了什么。


在上面的例子中，管道包含一个
 [UNet2DModel](/docs/diffusers/v0.23.0/en/api/models/unet2d#diffusers.UNet2DModel)
 模型和一个
 [DDPMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.DDPMScheduler)
 。该管道通过获取所需输出大小的随机噪声并将其多次通过模型来对图像进行去噪。在每个时间步，模型预测
 *残留噪音*
 调度程序使用它来预测噪声较小的图像。管道重复此过程，直到到达指定数量的推理步骤的末尾。


要分别使用模型和调度程序重新创建管道，让我们编写自己的去噪过程。


1.加载模型和调度器：



```
>>> from diffusers import DDPMScheduler, UNet2DModel

>>> scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
>>> model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")
```


2. 设置运行去噪过程的时间步数：



```
>>> scheduler.set_timesteps(50)
```


3. 设置调度程序时间步长会创建一个包含均匀间隔元素的张量，在本示例中为 50。每个元素对应于模型对图像进行去噪的时间步长。当您稍后创建去噪循环时，您将迭代该张量以对图像进行去噪：



```
>>> scheduler.timesteps
tensor([980, 960, 940, 920, 900, 880, 860, 840, 820, 800, 780, 760, 740, 720,
    700, 680, 660, 640, 620, 600, 580, 560, 540, 520, 500, 480, 460, 440,
    420, 400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180, 160,
    140, 120, 100,  80,  60,  40,  20,   0])
```


4. 创建一些与所需输出形状相同的随机噪声：



```
>>> import torch

>>> sample_size = model.config.sample_size
>>> noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")
```


5. 现在编写一个循环来迭代时间步。在每个时间步，模型都会执行
 [UNet2DModel.forward()](/docs/diffusers/v0.23.0/en/api/models/unet2d#diffusers.UNet2DModel.forward)
 通过并返回噪声残差。调度程序的
 [step()](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.DDPMScheduler.step)
 方法采用噪声残差、时间步长和输入，并预测前一个时间步长的图像。该输出成为去噪循环中模型的下一个输入，并且它将重复直到到达降噪循环的末尾。
 `时间步长`
 大批。



```
>>> input = noise

>>> for t in scheduler.timesteps:
...     with torch.no_grad():
...         noisy_residual = model(input, t).sample
...     previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
...     input = previous_noisy_sample
```


这就是整个去噪过程，您可以使用相同的模式来编写任何扩散系统。


6. 最后一步是将去噪输出转换为图像：



```
>>> from PIL import Image
>>> import numpy as np

>>> image = (input / 2 + 0.5).clamp(0, 1).squeeze()
>>> image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
>>> image = Image.fromarray(image)
>>> image
```


在下一部分中，您将测试您的技能并分解更复杂的稳定扩散管道。步骤或多或少是相同的。您将初始化必要的组件，并设置时间步数来创建
 `时间步长`
 大批。这
 `时间步长`
 数组用于去噪循环，对于该数组中的每个元素，模型预测噪声较小的图像。去噪循环迭代
 `时间步长`
 的，并且在每个时间步长，它输出一个噪声残差，调度程序使用它来预测前一个时间步长的噪声较小的图像。重复此过程，直到到达末尾
 `时间步长`
 大批。


我们来尝试一下吧！


## 解构稳定扩散管道



稳定扩散是文本到图像
 *潜在扩散*
 模型。它被称为潜在扩散模型，因为它使用图像的低维表示而不是实际的像素空间，这使得它的内存效率更高。编码器将图像压缩为更小的表示，解码器将压缩的表示转换回图像。对于文本到图像模型，您需要一个分词器和一个编码器来生成文本嵌入。从前面的示例中，您已经知道您需要一个 UNet 模型和一个调度程序。


正如您所看到的，这已经比仅包含 UNet 模型的 DDPM 管道更复杂。稳定扩散模型具有三个独立的预训练模型。


💡 阅读
 [稳定扩散如何工作？](https://huggingface.co/blog/stable_diffusion#how-does-stable-diffusion-work)
 有关 VAE、UNet 和文本编码器模型如何工作的更多详细信息，请参阅博客。


既然您知道稳定扩散管道需要什么，请使用以下命令加载所有这些组件
 [来自_pretrained()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin.from_pretrained)
 方法。你可以在预训练中找到它们
 [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 检查点，每个组件都存储在单独的子文件夹中：



```
>>> from PIL import Image
>>> import torch
>>> from transformers import CLIPTextModel, CLIPTokenizer
>>> from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

>>> vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
>>> tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
>>> text_encoder = CLIPTextModel.from_pretrained(
...     "CompVis/stable-diffusion-v1-4", subfolder="text\_encoder", use_safetensors=True
... )
>>> unet = UNet2DConditionModel.from_pretrained(
...     "CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True
... )
```


而不是默认的
 [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler)
 ，将其交换为
 [UniPCMultistepScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)
 看看插入不同的调度程序是多么容易：



```
>>> from diffusers import UniPCMultistepScheduler

>>> scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
```


为了加速推理，请将模型移至 GPU，因为与调度程序不同，它们具有可训练的权重：



```
>>> torch_device = "cuda"
>>> vae.to(torch_device)
>>> text_encoder.to(torch_device)
>>> unet.to(torch_device)
```


### 


 创建文本嵌入


下一步是对文本进行标记以生成嵌入。该文本用于调节 UNet 模型并将扩散过程引导至类似于输入提示的方向。


💡 的
 `指导规模`
 参数决定生成图像时应赋予提示多少权重。


如果您想生成其他内容，请随意选择您喜欢的任何提示！



```
>>> prompt = ["a photograph of an astronaut riding a horse"]
>>> height = 512  # default height of Stable Diffusion
>>> width = 512  # default width of Stable Diffusion
>>> num_inference_steps = 25  # Number of denoising steps
>>> guidance_scale = 7.5  # Scale for classifier-free guidance
>>> generator = torch.manual_seed(0)  # Seed generator to create the initial latent noise
>>> batch_size = len(prompt)
```


对文本进行标记并根据提示生成嵌入：



```
>>> text_input = tokenizer(
...     prompt, padding="max\_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
... )

>>> with torch.no_grad():
...     text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
```


您还需要生成
 *无条件文本嵌入*
 这是填充令牌的嵌入。这些需要具有相同的形状（
 `批量大小`
 和
 `seq_length`
 ) 作为条件
 `文本嵌入`
 :



```
>>> max_length = text_input.input_ids.shape[-1]
>>> uncond_input = tokenizer([""] * batch_size, padding="max\_length", max_length=max_length, return_tensors="pt")
>>> uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
```


让我们将条件嵌入和无条件嵌入连接到一个批处理中，以避免进行两次前向传递：



```
>>> text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
```


### 


 创建随机噪声


接下来，生成一些初始随机噪声作为扩散过程的起点。这是图像的潜在表示，它将逐渐去噪。此时，
 ‘潜伏’
 图像比最终图像尺寸小，但这没关系，因为模型稍后会将其转换为最终的 512x512 图像尺寸。


💡 高度和宽度除以 8，因为
 `瓦伊`
 模型有 3 个下采样层。您可以通过运行以下命令进行检查：



```
2 ** (len(vae.config.block_out_channels) - 1) == 8
```



```
>>> latents = torch.randn(
...     (batch_size, unet.config.in_channels, height // 8, width // 8),
...     generator=generator,
... )
>>> latents = latents.to(torch_device)
```


### 


 对图像进行去噪


首先使用初始噪声分布缩放输入，
 *西格玛*
 ，噪声标度值，这是改进的调度程序所需的，例如
 [UniPCMultistepScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)
 :



```
>>> latents = latents * scheduler.init_noise_sigma
```


最后一步是创建降噪循环，逐步将纯噪声转换为
 `潜伏`
 到您的提示所描述的图像。请记住，去噪循环需要做三件事：


1. 设置去噪期间使用的调度程序的时间步长。
2. 迭代时间步。
3. 在每个时间步，调用UNet模型来预测噪声残差，并将其传递给调度器来计算先前的噪声样本。



```
>>> from tqdm.auto import tqdm

>>> scheduler.set_timesteps(num_inference_steps)

>>> for t in tqdm(scheduler.timesteps):
...     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
...     latent_model_input = torch.cat([latents] * 2)

...     latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

...     # predict the noise residual
...     with torch.no_grad():
...         noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

...     # perform guidance
...     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
...     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

...     # compute the previous noisy sample x\_t -> x\_t-1
...     latents = scheduler.step(noise_pred, t, latents).prev_sample
```


### 


 解码图像


最后一步是使用
 `瓦伊`
 将潜在表示解码为图像并获得解码输出
 `样本`
 :



```
# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample
```


最后将图像转换为
 `PIL.Image`
 查看您生成的图像！



```
>>> image = (image / 2 + 0.5).clamp(0, 1).squeeze()
>>> image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
>>> images = (image * 255).round().astype("uint8")
>>> image = Image.fromarray(image)
>>> image
```


![](https://huggingface.co/blog/assets/98_stable_diffusion/stable_diffusion_k_lms.png)


## 下一步



从基本管道到复杂管道，您已经看到编写自己的扩散系统真正需要的只是一个降噪循环。该循环应设置调度程序的时间步长，对其进行迭代，并交替调用 UNet 模型来预测噪声残差，并将其传递给调度程序以计算先前的噪声样本。


这确实是 🧨 Diffusers 的设计目的：让使用模型和调度程序直观、轻松地编写自己的扩散系统。


对于您的后续步骤，请随时：


* 了解如何
 [构建并贡献管道](../using-diffusers/contribute_pipeline)
 到🧨 扩散器。我们迫不及待地想看看您会想出什么！
* 探索
 [现有管道](../api/pipelines/overview)
 在库中，看看是否可以分别使用模型和调度程序从头开始解构和构建管道。