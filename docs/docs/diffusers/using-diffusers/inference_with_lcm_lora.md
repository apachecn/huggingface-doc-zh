# [](#performing-inference-with-lcm-lora)使用 LCM-LoRA 执行推理

> 译者：[疾风兔X](https://github.com/jifnegtu)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/inference_with_lcm_lora>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/inference_with_lcm_lora>




![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)

![在 Studio 实验室中打开](https://studiolab.sagemaker.aws/studiolab.svg)



潜在一致性模型 （LCM） 通常只需 2-4 个步骤即可生成高质量的图像，从而可以在几乎实时的设置中使用扩散模型。

从[官方网站](https://latent-consistency-models.github.io/)：

> LCM 只需 4,000 个训练步骤（~32 个 A100 GPU 小时）即可从任何预训练的稳定扩散 （SD） 中蒸馏出来，只需 2~4 个步骤甚至一个步骤即可生成高质量的 768 x 768 分辨率图像，从而显著加快文本到图像的生成速度。我们使用 LCM 在短短 4,000 次训练迭代中提炼出 Dreamshaper-V7 版本的 SD。

有关 LCM 的更多技术概述，请参阅[白皮书](https://huggingface.co/papers/2310.04378)。

然而，为了进行潜在一致性蒸馏（distillation），每个模型都需要单独进行蒸馏。LCM-LoRA的核心思想是仅训练几个适配器层，在这个案例中，适配器是LoRA。这样，我们就不需要训练整个模型，并保持可训练参数的数量可管理。 resulting LoRAs 然后可以应用于任何经过微调的模型版本，而无需单独蒸馏它们。此外，LoRAs还可以应用于图像到图像、ControlNet/T2I-Adapter、修复、AnimateDiff等。LCM-LoRA还可以与其他LoRAs结合使用，以非常少的步骤（4-8步）生成具有风格的图像。

LCM-LoRA 可用于 [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)、[stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 和 [SSD-1B](https://huggingface.co/segmind/SSD-1B) 型号。所有检查点都可以在此[集合](https://huggingface.co/collections/latent-consistency/latent-consistency-models-loras-654cdd24e111e16f0865fba6)中找到。

有关 LCM-LoRA 的更多详细信息，请参阅[技术报告](https://huggingface.co/papers/2311.05556)。

本指南介绍如何使用 LCM-LoRA 对

+   文生图（text-to-image）
+   图生图（image-to-image）
+   结合风格化的 LoRA
+   ControlNet/T2I-Adapter
+   修复（inpainting）
+   动画变化 （AnimateDiff）

在阅读本指南之前，我们将了解使用 LCM-LoRA 执行推理的一般工作流程。 LCM-LoRA 与其他稳定扩散 LoRA 类似，因此它们可以与任何支持 LoRA 的 [DiffusionPipeline](/docs/diffusers/v0.27.2/en/api/pipelines/overview#diffusers.DiffusionPipeline) 一起使用。

+   加载特定于任务的管道和模型。
+   将计划程序设置为 [LCMScheduler](/docs/diffusers/v0.27.2/en/api/schedulers/lcm#diffusers.LCMScheduler)。
+   加载模型的 LCM-LoRA 权重。
+   减小 between 并设置 between \[4， 8\]。`guidance_scale``[1.0, 2.0]``num_inference_steps`
+   使用常用参数对管道执行推理。

让我们看看如何使用 LCM-LoRA 对不同的任务进行推理。

首先，确保您安装了 [peft](https://github.com/huggingface/peft)，以获得更好的 LoRA 支持。

```py
pip install -U peft
```

## [](#text-to-image)文生图（Text-to-image）

您将 [StableDiffusionXLPipeline](/docs/diffusers/v0.27.2/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline) 与调度程序 [LCMScheduler](/docs/diffusers/v0.27.2/en/api/schedulers/lcm#diffusers.LCMScheduler) 一起使用，然后加载 LCM-LoRA。与 LCM-LoRA 和调度程序一起，该pipeline可实现快速推理工作流程，克服扩散模型的缓慢迭代特性。

```py
import torch
from diffusers import DiffusionPipeline, LCMScheduler

pipe = DiffusionPipeline.from\_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch\_dtype=torch.float16
).to("cuda")

\# set scheduler
pipe.scheduler = LCMScheduler.from\_config(pipe.scheduler.config)

\# load LCM-LoRA
pipe.load\_lora\_weights("latent-consistency/lcm-lora-sdxl")

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

generator = torch.manual\_seed(42)
image = pipe(
    prompt=prompt, num\_inference\_steps=4, generator=generator, guidance\_scale=1.0
).images\[0\]
```
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdxl_t2i.png)

请注意，我们仅使用 4 个步骤进行生成，这比通常用于标准 SDXL 的步骤要少得多。

>您可能已经注意到，我们设置了 ，它禁用了 classifer-free-guidance。这是因为 LCM-LoRA 是在指导下训练的，因此在这种情况下，批处理大小不必加倍。这导致了更快的推理时间，缺点是负面提示对去噪过程没有任何影响。`guidance_scale=1.0`
>您还可以将指导与 LCM-LoRA 一起使用，但由于训练的性质，模型对值非常敏感，高值可能会导致生成的图像中出现伪影。在我们的实验中，我们发现最佳值在 \[1.0， 2.0\] 的范围内。`guidance_scale`

### [](#inference-with-a-fine-tuned-model)使用微调模型进行推理

如上所述，LCM-LoRA 可以应用于模型的任何微调版本，而无需单独蒸馏它们。让我们看看如何使用微调模型进行推理。在此示例中，我们将使用 [animagine-xl](https://huggingface.co/Linaqruf/animagine-xl) 模型，它是用于生成动漫的 SDXL 模型的微调版本。

```py
from diffusers import DiffusionPipeline, LCMScheduler

pipe = DiffusionPipeline.from\_pretrained(
    "Linaqruf/animagine-xl",
    variant="fp16",
    torch\_dtype=torch.float16
).to("cuda")

\# set scheduler
pipe.scheduler = LCMScheduler.from\_config(pipe.scheduler.config)

\# load LCM-LoRA
pipe.load\_lora\_weights("latent-consistency/lcm-lora-sdxl")

prompt = "face focus, cute, masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck"

generator = torch.manual\_seed(0)
image = pipe(
    prompt=prompt, num\_inference\_steps=4, generator=generator, guidance\_scale=1.0
).images\[0\]
```
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdxl_t2i_finetuned.png)

## [](#image-to-image)图生图（Image-to-image）

LCM-LoRA 也可以应用于图像到图像任务。让我们看看如何使用 LCM 进行图像到图像的生成。在此示例中，我们将使用 [dreamshaper-7](https://huggingface.co/Lykon/dreamshaper-7) 模型和 LCM-LoRA 进行 .`stable-diffusion-v1-5`

```py
import torch
from diffusers import AutoPipelineForImage2Image, LCMScheduler
from diffusers.utils import make\_image\_grid, load\_image

pipe = AutoPipelineForImage2Image.from\_pretrained(
    "Lykon/dreamshaper-7",
    torch\_dtype=torch.float16,
    variant="fp16",
).to("cuda")

\# set scheduler
pipe.scheduler = LCMScheduler.from\_config(pipe.scheduler.config)

\# load LCM-LoRA
pipe.load\_lora\_weights("latent-consistency/lcm-lora-sdv1-5")

\# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init\_image = load\_image(url)
prompt = "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"

\# pass prompt and image to pipeline
generator = torch.manual\_seed(0)
image = pipe(
    prompt,
    image=init\_image,
    num\_inference\_steps=4,
    guidance\_scale=1,
    strength=0.6,
    generator=generator
).images\[0\]
make\_image\_grid(\[init\_image, image\], rows=1, cols=2)
```
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdv1-5_i2i.png)

>您可以根据提示和提供的图像获得不同的结果。为了获得最佳结果，我们建议对`num_inference_steps`、`strength`和`guidance_scale`参数尝试的不同值，然后选择最佳值。

## [](#combine-with-styled-loras)与样式化的 LoRA 结合使用

LCM-LoRA 可以与其他 LoRA 结合使用，只需很少的步骤即可生成样式化图像 （4-8）。在以下示例中，我们将 LCM-LoRA 与[papercut LoRA](TheLastBen/Papercut_SDXL) 一起使用。 要了解有关如何组合 LoRA 的更多信息，请参阅[本指南](https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference#combine-multiple-adapters)。

```py
import torch
from diffusers import DiffusionPipeline, LCMScheduler

pipe = DiffusionPipeline.from\_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch\_dtype=torch.float16
).to("cuda")

\# set scheduler
pipe.scheduler = LCMScheduler.from\_config(pipe.scheduler.config)

\# load LoRAs
pipe.load\_lora\_weights("latent-consistency/lcm-lora-sdxl", adapter\_name="lcm")
pipe.load\_lora\_weights("TheLastBen/Papercut\_SDXL", weight\_name="papercut.safetensors", adapter\_name="papercut")

\# Combine LoRAs
pipe.set\_adapters(\["lcm", "papercut"\], adapter\_weights=\[1.0, 0.8\])

prompt = "papercut, a cute fox"
generator = torch.manual\_seed(0)
image = pipe(prompt, num\_inference\_steps=4, guidance\_scale=1, generator=generator).images\[0\]
image
```
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdx_lora_mix.png)

## [](#controlnett2i-adapter)ControlNet/T2I-Adapter

让我们看看如何使用 ControlNet/T2I-Adapter 和 LCM-LoRA 进行推理。

### [](#controlnet)ControlNet

在此示例中，我们将使用 SD-v1-5 模型和 SD-v1-5 的 LCM-LoRA 以及canny  ControlNet。

```py
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from diffusers.utils import load\_image

image = load\_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input\_image\_vermeer.png"
).resize((512, 512))

image = np.array(image)

low\_threshold = 100
high\_threshold = 200

image = cv2.Canny(image, low\_threshold, high\_threshold)
image = image\[:, :, None\]
image = np.concatenate(\[image, image, image\], axis=2)
canny\_image = Image.fromarray(image)

controlnet = ControlNetModel.from\_pretrained("lllyasviel/sd-controlnet-canny", torch\_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from\_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch\_dtype=torch.float16,
    safety\_checker=None,
    variant="fp16"
).to("cuda")

\# set scheduler
pipe.scheduler = LCMScheduler.from\_config(pipe.scheduler.config)

\# load LCM-LoRA
pipe.load\_lora\_weights("latent-consistency/lcm-lora-sdv1-5")

generator = torch.manual\_seed(0)
image = pipe(
    "the mona lisa",
    image=canny\_image,
    num\_inference\_steps=4,
    guidance\_scale=1.5,
    controlnet\_conditioning\_scale=0.8,
    cross\_attention\_kwargs={"scale": 1},
    generator=generator,
).images\[0\]
make\_image\_grid(\[canny\_image, image\], rows=1, cols=2)
```
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdv1-5_controlnet.png)

>此示例中的推理参数可能不适用于所有示例，因此我们建议您尝试不同的 'num\_inference\_steps'、'guidance\_scale'、'controlnet\_conditioning\_scale' 和 'cross\_attention\_kwargs' 参数值，并选择最佳值。

### [](#t2i-adapter)T2I-Adapter

此示例说明如何将 LCM-LoRA 与 [Canny T2I-Adapter](TencentARC/t2i-adapter-canny-sdxl-1.0)和 SDXL 一起使用。

```py
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, LCMScheduler
from diffusers.utils import load\_image, make\_image\_grid

\# Prepare image
\# Detect the canny map in low resolution to avoid high-frequency details
image = load\_image(
    "https://huggingface.co/Adapter/t2iadapter/resolve/main/figs\_SDXLV1.0/org\_canny.jpg"
).resize((384, 384))

image = np.array(image)

low\_threshold = 100
high\_threshold = 200

image = cv2.Canny(image, low\_threshold, high\_threshold)
image = image\[:, :, None\]
image = np.concatenate(\[image, image, image\], axis=2)
canny\_image = Image.fromarray(image).resize((1024, 1024))

\# load adapter
adapter = T2IAdapter.from\_pretrained("TencentARC/t2i-adapter-canny-sdxl-1.0", torch\_dtype=torch.float16, varient="fp16").to("cuda")

pipe = StableDiffusionXLAdapterPipeline.from\_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    adapter=adapter,
    torch\_dtype=torch.float16,
    variant="fp16", 
).to("cuda")

\# set scheduler
pipe.scheduler = LCMScheduler.from\_config(pipe.scheduler.config)

\# load LCM-LoRA
pipe.load\_lora\_weights("latent-consistency/lcm-lora-sdxl")

prompt = "Mystical fairy in real, magic, 4k picture, high quality"
negative\_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"

generator = torch.manual\_seed(0)
image = pipe(
    prompt=prompt,
    negative\_prompt=negative\_prompt,
    image=canny\_image,
    num\_inference\_steps=4,
    guidance\_scale=1.5, 
    adapter\_conditioning\_scale=0.8, 
    adapter\_conditioning\_factor=1,
    generator=generator,
).images\[0\]
make\_image\_grid(\[canny\_image, image\], rows=1, cols=2)
```
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdxl_t2iadapter.png)

## [](#inpainting)修复（Inpainting）

LCM-LoRA也可用于修复。

```py
import torch
from diffusers import AutoPipelineForInpainting, LCMScheduler
from diffusers.utils import load\_image, make\_image\_grid

pipe = AutoPipelineForInpainting.from\_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch\_dtype=torch.float16,
    variant="fp16",
).to("cuda")

\# set scheduler
pipe.scheduler = LCMScheduler.from\_config(pipe.scheduler.config)

\# load LCM-LoRA
pipe.load\_lora\_weights("latent-consistency/lcm-lora-sdv1-5")

\# load base and mask image
init\_image = load\_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask\_image = load\_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint\_mask.png")

\# generator = torch.Generator("cuda").manual\_seed(92)
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
generator = torch.manual\_seed(0)
image = pipe(
    prompt=prompt,
    image=init\_image,
    mask\_image=mask\_image,
    generator=generator,
    num\_inference\_steps=4,
    guidance\_scale=4, 
).images\[0\]
make\_image\_grid(\[init\_image, mask\_image, image\], rows=1, cols=3)
```
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdv1-5_inpainting.png)

## [](#animatediff)动画变化（AnimateDiff）

`AnimateDiff`允许您使用 Stable Diffusion 模型对图像进行动画处理。为了获得良好的结果，我们需要生成多个帧（16-24帧），而使用标准标清模型执行此操作可能会非常慢。 LCM-LoRA 可用于显着加快该过程，因为您只需要为每帧执行 4-8 个步骤。让我们看看如何使用 LCM-LoRA 和 AnimateDiff 执行动画。

```py
import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler, LCMScheduler
from diffusers.utils import export\_to\_gif

adapter = MotionAdapter.from\_pretrained("diffusers/animatediff-motion-adapter-v1-5")
pipe = AnimateDiffPipeline.from\_pretrained(
    "frankjoshua/toonyou\_beta6",
    motion\_adapter=adapter,
).to("cuda")

\# set scheduler
pipe.scheduler = LCMScheduler.from\_config(pipe.scheduler.config)

\# load LCM-LoRA
pipe.load\_lora\_weights("latent-consistency/lcm-lora-sdv1-5", adapter\_name="lcm")
pipe.load\_lora\_weights("guoyww/animatediff-motion-lora-zoom-in", weight\_name="diffusion\_pytorch\_model.safetensors", adapter\_name="motion-lora")

pipe.set\_adapters(\["lcm", "motion-lora"\], adapter\_weights=\[0.55, 1.2\])

prompt = "best quality, masterpiece, 1girl, looking at viewer, blurry background, upper body, contemporary, dress"
generator = torch.manual\_seed(0)
frames = pipe(
    prompt=prompt,
    num\_inference\_steps=5,
    guidance\_scale=1.25,
    cross\_attention\_kwargs={"scale": 1},
    num\_frames=24,
    generator=generator
).frames\[0\]
export\_to\_gif(frames, "animation.gif")
```
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdv1-5_animatediff.gif)