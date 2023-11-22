# 无条件图像生成

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/unconditional_image_generation>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/unconditional_image_generation>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


无条件图像生成是一项相对简单的任务。该模型仅生成图像 - 没有任何其他上下文（如文本或图像） - 类似于其所训练的训练数据。


这
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 是使用预先训练的扩散系统进行推理的最简单方法。


首先创建一个实例
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 并指定您要下载的管道检查点。
您可以使用任何 🧨 扩散器
 [检查点](https://huggingface.co/models?library=diffusers&sort=downloads)
 从中心（您将使用的检查点生成蝴蝶图像）。


💡 想训练自己的无条件图像生成模型吗？看看训练情况
 [指南](../training/unconditional_training)
 了解如何生成自己的图像。


在本指南中，您将使用
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 用于无条件图像生成
 [DDPM](https://arxiv.org/abs/2006.11239)
 :



```
from diffusers import DiffusionPipeline

generator = DiffusionPipeline.from_pretrained("anton-l/ddpm-butterflies-128", use_safetensors=True)
```


这
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 下载并缓存所有建模、标记化和调度组件。
由于该模型由大约 14 亿个参数组成，因此我们强烈建议在 GPU 上运行它。
您可以将生成器对象移动到 GPU，就像在 PyTorch 中一样：



```
generator.to("cuda")
```


现在您可以使用
 `发电机`
 生成图像：



```
image = generator().images[0]
image
```


输出默认包装成
 [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class)
 目的。


您可以通过调用以下命令保存图像：



```
image.save("generated\_image.png")
```


尝试下面的空间，并随意使用推理步骤参数，看看它如何影响图像质量！