# 通过确定性生成提高图像质量

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/reusing_seeds>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/reusing_seeds>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


提高生成图像质量的常见方法是
 *确定性批次生成*
 ，生成一批图像并选择一张图像进行改进，并在第二轮推理中提供更详细的提示。关键是传递一个列表
 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html#generator)
 到批量图像生成的管道，并将每个
 `发电机`
 到种子，以便您可以将其重新用于图像。


让我们使用
 [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 例如，并生成以下提示的多个版本：



```
prompt = "Labrador in the style of Vermeer"
```


实例化管道
 [DiffusionPipeline.from\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained)
 并将其放置在 GPU 上（如果可用）：



```
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
pipe = pipe.to("cuda")
```


现在，定义四种不同的
 `发电机`
 s 并分配每个
 `发电机`
 一粒种子（
 `0`
 到
 `3`
 ）这样你就可以重复使用
 `发电机`
 稍后针对特定图像：



```
generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(4)]
```


生成图像并查看：



```
images = pipe(prompt, generator=generator, num_images_per_prompt=4).images
make_image_grid(images, rows=2, cols=2)
```


![img](https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/reusabe_seeds.jpg)


在此示例中，您将改进第一张图像 - 但实际上，您可以使用任何您想要的图像（甚至是带有双眼的图像！）。第一张图片使用了
 `发电机`
 有种子
 `0`
 ，这样你就可以重用它
 `发电机`
 进行第二轮推理。要提高图像质量，请在提示中添加一些附加文本：



```
prompt = [prompt + t for t in [", highly realistic", ", artsy", ", trending", ", colorful"]]
generator = [torch.Generator(device="cuda").manual_seed(0) for i in range(4)]
```


创建四个带有种子的生成器
 `0`
 ，并生成另一批图像，所有这些图像都应该看起来像上一轮的第一张图像！



```
images = pipe(prompt, generator=generator).images
make_image_grid(images, rows=2, cols=2)
```


![img](https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/reusabe_seeds_2.jpg)