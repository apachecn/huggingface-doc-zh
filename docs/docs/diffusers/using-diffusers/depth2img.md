# 文本引导深度图像生成

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/depth2img>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/depth2img>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


这
 [StableDiffusionDepth2ImgPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/depth2img#diffusers.StableDiffusionDepth2ImgPipeline)
 允许您传递文本提示和初始图像来调节新图像的生成。此外，您还可以通过
 `深度图`
 以保留图像结构。如果不
 `深度图`
 提供后，管道通过集成的自动预测深度
 [深度估计模型](https://github.com/isl-org/MiDaS)
 。


首先创建一个实例
 [StableDiffusionDepth2ImgPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/depth2img#diffusers.StableDiffusionDepth2ImgPipeline)
 :



```
import torch
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers.utils import load_image, make_image_grid

pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")
```


现在将您的提示传递到管道。您还可以通过
 `否定提示`
 防止某些单词指导图像的生成方式：



```
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
init_image = load_image(url)
prompt = "two tigers"
negative_prompt = "bad, deformed, ugly, bad anatomy"
image = pipeline(prompt=prompt, image=init_image, negative_prompt=negative_prompt, strength=0.7).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```


| 	 Input	  | 	 Output	  |
| --- | --- |
|  |  |