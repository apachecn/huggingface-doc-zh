# 稳定扩散XL

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/sdxl>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/sdxl>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


[稳定扩散XL](https://huggingface.co/papers/2307.01952)
 (SDXL) 是一种强大的文本到图像生成模型，它以三种关键方式迭代以前的稳定扩散模型：


1. UNet 增大 3 倍，SDXL 将第二个文本编码器 (OpenCLIP ViT-bigG/14) 与原始文本编码器相结合，显着增加参数数量
2. 引入尺寸和裁剪调节，以防止训练数据被丢弃，并更好地控制生成图像的裁剪方式
3.引入两阶段模型过程；这
 *根据*
 模型（也可以作为独立模型运行）生成图像作为输入
 *精炼机*
 添加额外高品质细节的模型


本指南将向您展示如何使用 SDXL 进行文本到图像、图像到图像和修复。


在开始之前，请确保已安装以下库：



```
# uncomment to install the necessary libraries in Colab
#!pip install diffusers transformers accelerate safetensors omegaconf invisible-watermark>=0.2.0
```


我们建议安装
 [隐形水印](https://pypi.org/project/invisible-watermark/)
 库来帮助识别生成的图像。如果安装了invisible-watermark库，则默认使用它。要禁用水印：



```
pipeline = StableDiffusionXLPipeline.from_pretrained(..., add_watermarker=False)
```


## 加载模型检查点



模型权重可能存储在 Hub 上或本地的单独子文件夹中，在这种情况下，您应该使用
 [来自\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained)
 方法：



```
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to("cuda")
```


您还可以使用
 [来自\_single\_file()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.from_single_file)
 加载以单个文件格式存储的模型检查点的方法（
 `.ckpt`
 或者
 `.safetensors`
 ) 从中心或本地：



```
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd\_xl\_base\_1.0.safetensors", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd\_xl\_refiner\_1.0.safetensors", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to("cuda")
```


## 文本转图像



对于文本到图像，请传递文本提示。默认情况下，SDXL 生成 1024x1024 图像以获得最佳效果。您可以尝试设置
 `高度`
 和
 `宽度`
 参数设置为 768x768 或 512x512，但低于 512x512 的任何值都不太可能工作。



```
from diffusers import AutoPipelineForText2Image
import torch

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline(prompt=prompt).images[0]
```


![生成的丛林中宇航员图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png)


## 图像到图像



对于图像到图像，SDXL 特别适用于 768x768 和 1024x1024 之间的图像尺寸。传递初始图像和文本提示以调节图像：



```
from diffusers import AutoPipelineForImg2Img
from diffusers.utils import load_image

# use from\_pipe to avoid consuming additional memory when loading a checkpoint
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-img2img.png"

init_image = load_image(url).convert("RGB")
prompt = "a dog catching a frisbee in the jungle"
image = pipeline(prompt, image=init_image, strength=0.8, guidance_scale=10.5).images[0]
```


![生成的一只狗在丛林中抓飞盘的图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-img2img.png)


## 修复



对于修复，您需要原始图像和要在原始图像中替换的内容的蒙版。创建一个提示来描述您想要用什么来替换遮罩区域。



```
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

# use from\_pipe to avoid consuming additional memory when loading a checkpoint
pipeline = AutoPipelineForInpainting.from_pipe(pipeline_text2image).to("cuda")

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
mask_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint-mask.png"

init_image = load_image(img_url).convert("RGB")
mask_image = load_image(mask_url).convert("RGB")

prompt = "A deep sea diver floating"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.85, guidance_scale=12.5).images[0]
```


![丛林中深海潜水员的生成图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint.png)


## 提高图像质量



SDXL 包括一个
 [精炼模型](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)
 专门用于对低噪声阶段图像进行去噪，以从基础模型生成更高质量的图像。精炼机有两种使用方法：


1. 结合使用基础模型和细化器模型来生成细化图像
2. 使用基础模型生成图像，随后使用细化模型为图像添加更多细节（这就是 SDXL 最初的训练方式）


### 


 基础+精炼机模型


当您同时使用基础模型和精化模型来生成图像时，这称为 (
 [*专家降噪器集合*](https://research.nvidia.com/labs/dir/eDiff-I/)
 ）。与将基本模型的输出传递给精化器模型相比，专家降噪器方法的集合需要更少的整体降噪步骤，因此它的运行速度应该要快得多。但是，您将无法检查基本模型的输出，因为它仍然包含大量噪声。


作为专家降噪器的集合，基础模型在高噪声扩散阶段充当专家，细化器模型在低噪声扩散阶段充当专家。加载基础模型和优化器模型：



```
from diffusers import DiffusionPipeline
import torch

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")
```


要使用此方法，您需要定义每个模型运行各自阶段的时间步数。对于基本模型，这是由
 [`denoising_end`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline.__call__.denoising_end)
 参数，对于精炼机模型，它由
 [`denoising_start`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline.__call__.denoising_start)
 范围。


这
 `去噪结束`
 和
 `去噪_开始`
 参数应该是 0 到 1 之间的浮点数。这些参数表示为调度程序定义的离散时间步长的比例。如果您也使用
 ‘力量’
 参数，它将被忽略，因为去噪步骤的数量是由模型训练的离散时间步长和声明的分数截止值决定的。


让我们设置一下
 `去噪结束=0.8`
 所以基础模型执行前 80% 的去噪
 **高噪音**
 时间步长和设置
 `去噪_开始=0.8`
 因此，精炼模型执行最后 20% 的去噪
 **低噪声**
 时间步长。基本模型输出应该是
 **潜**
 空间而不是 PIL 图像。



```
prompt = "A majestic lion jumping from a big stone at night"

image = base(
    prompt=prompt,
    num_inference_steps=40,
    denoising_end=0.8,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    num_inference_steps=40,
    denoising_start=0.8,
    image=image,
).images[0]
```


![生成的夜间岩石上的狮子图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lion_base.png)

 基础模型


![以更高的质量生成夜间岩石上的狮子图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lion_refined.png)

 专家降噪器集合


精炼机模型还可用于修复
 [StableDiffusionXLInpaintPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline)
 :



```
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image

base = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting\_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting\_examples/overture-creations-5sI6fQgYIuo\_mask.png"

init_image = load_image(img_url).convert("RGB")
mask_image = load_image(mask_url).convert("RGB")

prompt = "A majestic tiger sitting on a bench"
num_inference_steps = 75
high_noise_frac = 0.7

image = base(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    denoising_start=high_noise_frac,
).images[0]
```


这种专家降噪方法的集合适用于所有可用的调度程序！


### 


 从基础到精炼模型


SDXL 通过在图像到图像设置中使用细化器模型向基本模型的完全去噪图像添加额外的高质量细节，从而提高图像质量。


加载基础模型和精炼模型：



```
from diffusers import DiffusionPipeline
import torch

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")
```


从基础模型生成图像，并将模型输出设置为
 **潜**
 空间：



```
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

image = base(prompt=prompt, output_type="latent").images[0]
```


将生成的图像传递给细化器模型：



```
image = refiner(prompt=prompt, image=image[None, :]).images[0]
```


![生成的宇航员在火星上骑着绿马的图像](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/init_image.png)

 基础模型


![在火星上骑着绿马的宇航员的更高质量图像](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/refined_image.png)

 基础模型+精炼模型


对于修复，将精炼器模型加载到
 [StableDiffusionXLInpaintPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline)
 ， 去除
 `去噪结束`
 和
 `去噪_开始`
 参数，并为细化器选择较少数量的推理步骤。


## 微调节



SDXL 训练涉及几种额外的调节技术，这些技术被称为
 *微调节*
 。其中包括原始图像大小、目标图像大小和裁剪参数。微调节可在推理时用于创建高质量的居中图像。


借助无分类器的指导，您可以使用微调节和负微调节参数。它们可以在
 [StableDiffusionXLPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline)
 ,
 [StableDiffusionXLImg2ImgPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline)
 ,
 [StableDiffusionXLInpaintPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline)
 ， 和
 [StableDiffusionXLControlNetPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetPipeline)
 。



### 


 尺寸调节


尺寸调节有两种类型：


* [`original_size`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline.__call__.original_size)
 调节来自训练批次中的放大图像（因为丢弃占总训练数据近 40% 的较小图像会很浪费）。通过这种方式，SDXL 了解到高分辨率图像中不应出现放大伪影。在推理过程中，您可以使用
 `原始大小`
 表示原始图像的分辨率。使用默认值
 `(1024, 1024)`
 生成类似于数据集中的 1024x1024 图像的更高质量的图像。如果您选择使用较低的分辨率，例如
 `(256, 256)`
 ，模型仍然生成 1024x1024 图像，但它们看起来像数据集中的低分辨率图像（更简单的图案，模糊）。
* [`target_size`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline.__call__.target_size)
 调节来自微调 SDXL 以支持不同的图像长宽比。在推理过程中，如果使用默认值
 `(1024, 1024)`
 ，您将获得类似于数据集中方形图像组成的图像。我们建议使用相同的值
 `目标大小`
 和
 `原始大小`
 ，但请随意尝试其他选项！


🤗 Diffusers 还允许您指定有关图像大小的负面条件，以引导生成远离某些图像分辨率：



```
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(
    prompt=prompt,
    negative_original_size=(512, 512),
    negative_target_size=(1024, 1024),
).images[0]
```


![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/negative_conditions.png)

 负片图像的图像分辨率为 (128, 128)、(256, 256) 和 (512, 512)。
 

###


 作物调理


以前的稳定扩散模型生成的图像有时可能会被裁剪。这是因为图像实际上在训练期间被裁剪，以便批次中的所有图像具有相同的大小。通过调整裁剪坐标，SDXL
 *学习*
 没有裁剪 - 坐标
 `(0, 0)`
 - 通常与居中主体和完整面部相关（这是 🤗 漫射器中的默认值）。如果您想生成偏心的构图，您可以尝试不同的坐标！



```
from diffusers import StableDiffusionXLPipeline
import torch


pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline(prompt=prompt, crops_coords_top_left=(256,0)).images[0]
```


![生成的丛林中宇航员图像，稍作裁剪](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-cropped.png)


 您还可以指定负裁剪坐标以引导生成远离某些裁剪参数：



```
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(
    prompt=prompt,
    negative_original_size=(512, 512),
    negative_crops_coords_top_left=(0, 0),
    negative_target_size=(1024, 1024),
).images[0]
```


## 为每个文本编码器使用不同的提示



SDXL 使用两个文本编码器，因此可以向每个文本编码器传递不同的提示，这可以
 [提高质量](https://github.com/huggingface/diffusers/issues/4004#issuecomment-1627764201)
 。将您的原始提示传递给
 `提示`
 第二个提示
 `提示_2`
 （使用
 `否定提示`
 和
 `否定提示_2`
 如果您使用否定提示）：



```
from diffusers import StableDiffusionXLPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

# prompt is passed to OAI CLIP-ViT/L-14
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# prompt\_2 is passed to OpenCLIP-ViT/bigG-14
prompt_2 = "Van Gogh painting"
image = pipeline(prompt=prompt, prompt_2=prompt_2).images[0]
```


![以梵高绘画风格生成的丛林中宇航员图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-double-prompt.png)


 双文本编码器还支持需要单独加载的文本反转嵌入，如 [SDXL 文本反转](textual\_inversion\_inference#stable-diffusion-xl] 部分中所述。


## 优化



SDXL 是一个大型模型，您可能需要优化内存才能使其在您的硬件上运行。以下是一些节省内存和加快推理速度的技巧。


1. 将模型卸载到 CPU
 [启用\_model\_cpu\_offload()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/gligen#diffusers.StableDiffusionGLIGENTextImagePipeline.enable_model_cpu_offload)
 对于内存不足错误：



```
- base.to("cuda")
- refiner.to("cuda")
+ base.enable\_model\_cpu\_offload
+ refiner.enable\_model\_cpu\_offload
```


2. 使用
 `torch.编译`
 大约 20% 的加速（你需要
 `torch>2.0`
 ）：



```
+ base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
+ refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
```


3. 启用
 [xFormers](/优化/xformers)
 运行 SDXL 如果
 `torch<2.0`
 :



```
+ base.enable\_xformers\_memory\_efficient\_attention()
+ refiner.enable\_xformers\_memory\_efficient\_attention()
```


## 其他资源



如果您有兴趣尝试最小版本
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)
 在 SDXL 中使用，看看
 [minSDXL](https://github.com/cloneofsimo/minSDXL)
 用 PyTorch 编写的实现，与 🤗 Diffusers 直接兼容。