# [](#latent-consistency-model)潜在一致性模型（Latent Consistency Model）

> 译者：[疾风兔X](https://github.com/jifnegtu)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/inference_with_lcm>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/inference_with_lcm>



![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)

![在 Studio 实验室中打开](https://studiolab.sagemaker.aws/studiolab.svg)


潜在一致性模型 （LCM） 通常只需 2-4 个步骤即可生成高质量的图像，从而可以在几乎实时的设置中使用扩散模型。

从[官方网站](https://latent-consistency-models.github.io/)：

> LCM 只需 4,000 个训练步骤（~32 个 A100 GPU 小时）即可从任何预训练的稳定扩散 （SD） 中提取出来，只需 2~4 个步骤甚至一个步骤即可生成高质量的 768 x 768 分辨率图像，从而显著加快文本到图像的生成速度。我们使用 LCM 在短短 4,000 次训练迭代中提炼出 Dreamshaper-V7 版本的 SD。

有关 LCM 的更多技术概述，请参阅[白皮书](https://huggingface.co/papers/2310.04378)。

LCM 蒸馏模型可用于 [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)、[stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 和 [SSD-1B](https://huggingface.co/segmind/SSD-1B) 模型。所有检查点都可以在此[集合](https://huggingface.co/collections/latent-consistency/latent-consistency-models-weights-654ce61a95edd6dffccef6a8)中找到。

本指南介绍如何使用 LCM 对以下

+   文生图
+   图生图
+   结合 style LoRA
+   ControlNet/T2I-Adapter

## [](#text-to-image)文生图（Text-to-image）

你将 [StableDiffusionXLPipeline](/docs/diffusers/v0.27.2/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline) 流水线（pipeline）与 [LCMScheduler](/docs/diffusers/v0.27.2/en/api/schedulers/lcm#diffusers.LCMScheduler) 一起使用，然后加载 LCM-LoRA。与 LCM-LoRA 和调度器一起，该流水线（pipeline）可实现快速推理工作流程，克服扩散模型的缓慢迭代特性。

```py
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler
import torch

unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

generator = torch.manual_seed(0)
image = pipe(
    prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=8.0
).images[0]
```
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdxl_t2i.png)


注意，我们在生成过程中只使用了4步，这比标准的SDXL要少得多。

一些需要注意的细节：

+   为了执行无分类器引导（classifier-free guidance），通常会在管线内部将批次大小翻倍。然而，LCM（latent conditioning model，潜在条件模型）则是通过引导嵌入来应用引导，因此在这种情况下不必将批次大小翻倍。这导致推理时间更快，但缺点是负面提示对去噪过程没有任何影响。
+   UNet使用\[3.，13.\]指导量表范围进行训练。UNet是在[3.，13.]的引导尺度范围内进行训练的。因此，这是引导尺度(`guidance_scale`)理想范围。然而，在大多数情况下，使用1.0的值来禁用引导尺度(`guidance_scale`)也是有效的。

## [](#image-to-image)图生图（Image-to-image）

LCM 也可以应用于图像到图像任务。在此示例中，我们将使用 [LCM\_Dreamshaper\_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) 模型，但相同的步骤也可以应用于其他 LCM 模型。

```py
import torch
from diffusers import AutoPipelineForImage2Image, UNet2DConditionModel, LCMScheduler
from diffusers.utils import make_image_grid, load_image

unet = UNet2DConditionModel.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    subfolder="unet",
    torch_dtype=torch.float16,
)

pipe = AutoPipelineForImage2Image.from_pretrained(
    "Lykon/dreamshaper-7",
    unet=unet,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)
prompt = "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
generator = torch.manual_seed(0)
image = pipe(
    prompt,
    image=init_image,
    num_inference_steps=4,
    guidance_scale=7.5,
    strength=0.5,
    generator=generator
).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdv1-5_i2i.png)

您可以根据提示和提供的图像获得不同的结果。为了获得最佳结果，我们建议对`num_inference_steps` 、`strength` 和 `guidance_scale` 参数尝试不同的值，然后选择最佳值。

## [](#combine-with-style-loras)与风格 LoRA 结合（Combine with style LoRAs）

LCM 可以与其他样式的 LoRA 一起使用，只需很少的步骤 （4-8） 即可生成样式图像。在以下示例中，我们将使用[剪纸 LoRA。](TheLastBen/Papercut_SDXL)

```py
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler
import torch

unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

generator = torch.manual_seed(0)
image = pipe(
    prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=8.0
).images[0]
```
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdx_lora_mix.png)

## [](#controlnett2i-adapter)ControlNet/T2I-Adapter

让我们看看如何使用 ControlNet/T2I-Adapter 和 LCM 进行推理。

### [](#controlnet)ControlNet

在此示例中，我们将使用 [LCM\_Dreamshaper\_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) 模型结合Canny ControlNet ，但相同的步骤也可以应用于其他 LCM 模型。

```py
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from diffusers.utils import load_image, make_image_grid

image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
).resize((512, 512))

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

# set scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

generator = torch.manual_seed(0)
image = pipe(
    "the mona lisa",
    image=canny_image,
    num_inference_steps=4,
    generator=generator,
).images[0]
make_image_grid([canny_image, image], rows=1, cols=2)
```
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdv1-5_controlnet.png)

此示例中的推理参数可能不适用于所有示例，因此我们建议对`num_inference_steps`、`guidance_scale`、`controlnet_conditioning_scale`和`cross_attention_kwargs`参数尝试不同的值，然后选择最佳值。

### [](#t2i-adapter)T2I-Adapter

此示例说明如何将`lcm-sdxl`与 [Canny T2I 适配器](TencentARC/t2i-adapter-canny-sdxl-1.0)一起使用。

```py
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionXLAdapterPipeline, UNet2DConditionModel, T2IAdapter, LCMScheduler
from diffusers.utils import load_image, make_image_grid

# Prepare image
# Detect the canny map in low resolution to avoid high-frequency details
image = load_image(
    "https://huggingface.co/Adapter/t2iadapter/resolve/main/figs_SDXLV1.0/org_canny.jpg"
).resize((384, 384))

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image).resize((1024, 1216))

# load adapter
adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16, varient="fp16").to("cuda")

unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    unet=unet,
    adapter=adapter,
    torch_dtype=torch.float16,
    variant="fp16", 
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

prompt = "Mystical fairy in real, magic, 4k picture, high quality"
negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"

generator = torch.manual_seed(0)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=canny_image,
    num_inference_steps=4,
    guidance_scale=5,
    adapter_conditioning_scale=0.8, 
    adapter_conditioning_factor=1,
    generator=generator,
).images[0]
grid = make_image_grid([canny_image, image], rows=1, cols=2)
```
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdxl_t2iadapter.png)
