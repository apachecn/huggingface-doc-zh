
# [](#trajectory-consistency-distillation-lora)轨迹一致性蒸馏-LoRA（Trajectory Consistency Distillation-LoRA）

> 译者：[疾风兔X](https://github.com/jifnegtu)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/inference_with_tcd_lora>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/inference_with_tcd_lora>



![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)

![在 Studio 实验室中打开](https://studiolab.sagemaker.aws/studiolab.svg)



轨迹一致性蒸馏 （TCD） 使模型能够以更少的步骤生成更高质量、更详细的图像。此外，由于在蒸馏过程中有效缓解了误差，即使在大推理步骤的条件下，TCD也表现出卓越的性能。

TCD的主要优点是：

+   超越教师模型：TCD在小规模和大规模推理步骤中展现出更优的生成质量，并且性能超过了具有稳定扩散 XL （SDXL）的[DPM-Solver++（2S）](../../api/schedulers/multistep_dpm_solver)。在 TCD 培训期间，不包括额外的鉴别器或 LPIPS 监督。
    
+   灵活的推理步数调整：TCD采样的推理步骤可以自由调整，而不会对图像质量产生不利影响。
    
+   自由调整细节层次：在推理过程中，仅需通过单一的超参数 *gamma* ，即可调节图像的细节程度。
    

有关TCD的更多技术细节，请参阅[论文](https://arxiv.org/abs/2402.19159)或[官方项目页面](https://mhh0318.github.io/tcd/)）。

对于像 SDXL 这样的大型模型，TCD 采用了 [LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) 进行训练，以减少内存使用量。这一点尤为实用，因为只要基础模型相同，您就可以在不同的微调模型之间复用LoRA，无需进一步训练。

本指南将展示如何使用TCD-LoRA进行多种任务的推理操作，比如文本到图像转换和图像修复（inpainting），以及如何简便地将TCD-LoRA与其他适配器结合使用。请从下表中选择一个受支持的基础模型及其对应的TCD-LoRA检查点，开始你的操作。



| 基础模型 | TCD-LoRA 检查点 |
| --- | --- |
| [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | [TCD-SD15](https://huggingface.co/h1t/TCD-SD15-LoRA) |
| [stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) | [TCD-SD21-base](https://huggingface.co/h1t/TCD-SD21-base-LoRA) |
| [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | [TCD-SDXL](https://huggingface.co/h1t/TCD-SDXL-LoRA) |

确保您安装了 [PEFT](https://github.com/huggingface/peft)，以获得更好的 LoRA 支持。

>pip install -U peft

## [](#general-tasks)一般任务（General tasks）

在本指南中，我们使用 [StableDiffusionXLPipeline](/docs/diffusers/v0.27.2/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline) 和 [TCDScheduler](/docs/diffusers/v0.27.2/en/api/schedulers/tcd#diffusers.TCDScheduler)。使用 [load\_lora\_weights（）](/docs/diffusers/v0.27.2/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.load_lora_weights) 方法加载兼容 SDXL 的 TCD-LoRA 权重。

TCD-LoRA推理要记住的一些提示是：

+   保持`num_inference_steps`在 4 到 50 之间
+   设置 `eta`（用于控制每个步骤的随机性）介于 0 和 1 之间。在增加推理步骤数时，您应该使用更高的`eta`值，但缺点是更大的`eta`值在 [TCDScheduler](/docs/diffusers/v0.27.2/en/api/schedulers/tcd#diffusers.TCDScheduler) 中会导致图像更模糊。建议使用 0.3 的值以产生良好的结果。

文生图


```py
import torch
from diffusers import StableDiffusionXLPipeline, TCDScheduler

device = "cuda"
base\_model\_id = "stabilityai/stable-diffusion-xl-base-1.0"
tcd\_lora\_id = "h1t/TCD-SDXL-LoRA"

pipe = StableDiffusionXLPipeline.from\_pretrained(base\_model\_id, torch\_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = TCDScheduler.from\_config(pipe.scheduler.config)

pipe.load\_lora\_weights(tcd\_lora\_id)
pipe.fuse\_lora()

prompt = "Painting of the orange cat Otto von Garfield, Count of Bismarck-Schönhausen, Duke of Lauenburg, Minister-President of Prussia. Depicted wearing a Prussian Pickelhaube and eating his favorite meal - lasagna."

image = pipe(
    prompt=prompt,
    num\_inference\_steps=4,
    guidance\_scale=0,
    eta=0.3, 
    generator=torch.Generator(device=device).manual\_seed(0),
).images\[0\]
```
![](https://github.com/jabir-zheng/TCD/raw/main/assets/demo_image.png)

修复
```py
import torch
from diffusers import AutoPipelineForInpainting, TCDScheduler
from diffusers.utils import load\_image, make\_image\_grid

device = "cuda"
base\_model\_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
tcd\_lora\_id = "h1t/TCD-SDXL-LoRA"

pipe = AutoPipelineForInpainting.from\_pretrained(base\_model\_id, torch\_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = TCDScheduler.from\_config(pipe.scheduler.config)

pipe.load\_lora\_weights(tcd\_lora\_id)
pipe.fuse\_lora()

img\_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting\_examples/overture-creations-5sI6fQgYIuo.png"
mask\_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting\_examples/overture-creations-5sI6fQgYIuo\_mask.png"

init\_image = load\_image(img\_url).resize((1024, 1024))
mask\_image = load\_image(mask\_url).resize((1024, 1024))

prompt = "a tiger sitting on a park bench"

image = pipe(
  prompt=prompt,
  image=init\_image,
  mask\_image=mask\_image,
  num\_inference\_steps=8,
  guidance\_scale=0,
  eta=0.3,
  strength=0.99,  \# make sure to use \`strength\` below 1.0
  generator=torch.Generator(device=device).manual\_seed(0),
).images\[0\]

grid\_image = make\_image\_grid(\[init\_image, mask\_image, image\], rows=1, cols=3)
```
![](https://github.com/jabir-zheng/TCD/raw/main/assets/inpainting_tcd.png)

## [](#community-models)社区模型

TCD-LoRA 还可以与许多社区微调模型和插件配合使用。例如，加载 [animagine-xl-3.0](https://huggingface.co/cagliostrolab/animagine-xl-3.0) 检查点，它是 SDXL 的社区微调版本，用于生成动漫图像。


```py
import torch
from diffusers import StableDiffusionXLPipeline, TCDScheduler

device = "cuda"
base\_model\_id = "cagliostrolab/animagine-xl-3.0"
tcd\_lora\_id = "h1t/TCD-SDXL-LoRA"

pipe = StableDiffusionXLPipeline.from\_pretrained(base\_model\_id, torch\_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = TCDScheduler.from\_config(pipe.scheduler.config)

pipe.load\_lora\_weights(tcd\_lora\_id)
pipe.fuse\_lora()

prompt = "A man, clad in a meticulously tailored military uniform, stands with unwavering resolve. The uniform boasts intricate details, and his eyes gleam with determination. Strands of vibrant, windswept hair peek out from beneath the brim of his cap."

image = pipe(
    prompt=prompt,
    num\_inference\_steps=8,
    guidance\_scale=0,
    eta=0.3, 
    generator=torch.Generator(device=device).manual\_seed(0),
).images\[0\]
```
![](https://github.com/jabir-zheng/TCD/raw/main/assets/animagine_xl.png)

TCD-LoRA 还支持其他在不同风格上训练的 LoRA。例如，让我们加载 [TheLastBen/Papercut\_SDXL](https://huggingface.co/TheLastBen/Papercut_SDXL) LoRA，并使用 [set\_adapters（）](/docs/diffusers/v0.27.2/en/api/loaders/unet#diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters) 方法将其与 TCD-LoRA 融合。

>查看[合并 LoRA](merge_loras) 指南，了解有关高效合并方法的更多信息。

```py
import torch
from diffusers import StableDiffusionXLPipeline
from scheduling\_tcd import TCDScheduler 

device = "cuda"
base\_model\_id = "stabilityai/stable-diffusion-xl-base-1.0"
tcd\_lora\_id = "h1t/TCD-SDXL-LoRA"
styled\_lora\_id = "TheLastBen/Papercut\_SDXL"

pipe = StableDiffusionXLPipeline.from\_pretrained(base\_model\_id, torch\_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = TCDScheduler.from\_config(pipe.scheduler.config)

pipe.load\_lora\_weights(tcd\_lora\_id, adapter\_name="tcd")
pipe.load\_lora\_weights(styled\_lora\_id, adapter\_name="style")
pipe.set\_adapters(\["tcd", "style"\], adapter\_weights=\[1.0, 1.0\])

prompt = "papercut of a winter mountain, snow"

image = pipe(
    prompt=prompt,
    num\_inference\_steps=4,
    guidance\_scale=0,
    eta=0.3, 
    generator=torch.Generator(device=device).manual\_seed(0),
).images\[0\]
```
![](https://github.com/jabir-zheng/TCD/raw/main/assets/styled_lora.png)

## [](#adapters)适配器（Adapters）

TCD-LoRA 用途广泛，可以与其他适配器类型（如 ControlNets、IP-Adapter 和 AnimateDiff）结合使用。

### ControlNet
#### [](#depth-controlnet)Depth ControlNet
#### [](#canny-controlnet)Canny ControlNet
```py
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils import load\_image, make\_image\_grid
from scheduling\_tcd import TCDScheduler 

device = "cuda"
base\_model\_id = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet\_id = "diffusers/controlnet-canny-sdxl-1.0"
tcd\_lora\_id = "h1t/TCD-SDXL-LoRA"

controlnet = ControlNetModel.from\_pretrained(
    controlnet\_id,
    torch\_dtype=torch.float16,
    variant="fp16",
).to(device)
pipe = StableDiffusionXLControlNetPipeline.from\_pretrained(
    base\_model\_id,
    controlnet=controlnet,
    torch\_dtype=torch.float16,
    variant="fp16",
).to(device)
pipe.enable\_model\_cpu\_offload()

pipe.scheduler = TCDScheduler.from\_config(pipe.scheduler.config)

pipe.load\_lora\_weights(tcd\_lora\_id)
pipe.fuse\_lora()

prompt = "ultrarealistic shot of a furry blue bird"

canny\_image = load\_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd\_controlnet/bird\_canny.png")

controlnet\_conditioning\_scale = 0.5  \# recommended for good generalization

image = pipe(
    prompt, 
    image=canny\_image, 
    num\_inference\_steps=4, 
    guidance\_scale=0,
    eta=0.3,
    controlnet\_conditioning\_scale=controlnet\_conditioning\_scale,
    generator=torch.Generator(device=device).manual\_seed(0),
).images\[0\]

grid\_image = make\_image\_grid(\[canny\_image, image\], rows=1, cols=2)
```
![](https://github.com/jabir-zheng/TCD/raw/main/assets/controlnet_canny_tcd.png)

>此示例中的推理参数可能不适用于所有示例，因此我们建议您尝试不同的 'num\_inference\_steps'、'guidance\_scale'、'controlnet\_conditioning\_scale' 和 'cross\_attention\_kwargs' 参数值，并选择最佳值。

```py
import torch
import numpy as np
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils import load\_image, make\_image\_grid
from scheduling\_tcd import TCDScheduler 

device = "cuda"
depth\_estimator = DPTForDepthEstimation.from\_pretrained("Intel/dpt-hybrid-midas").to(device)
feature\_extractor = DPTFeatureExtractor.from\_pretrained("Intel/dpt-hybrid-midas")

def get\_depth\_map(image):
    image = feature\_extractor(images=image, return\_tensors="pt").pixel\_values.to(device)
    with torch.no\_grad(), torch.autocast(device):
        depth\_map = depth\_estimator(image).predicted\_depth

    depth\_map = torch.nn.functional.interpolate(
        depth\_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align\_corners=False,
    )
    depth\_min = torch.amin(depth\_map, dim=\[1, 2, 3\], keepdim=True)
    depth\_max = torch.amax(depth\_map, dim=\[1, 2, 3\], keepdim=True)
    depth\_map = (depth\_map - depth\_min) / (depth\_max - depth\_min)
    image = torch.cat(\[depth\_map\] \* 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()\[0\]
    image = Image.fromarray((image \* 255.0).clip(0, 255).astype(np.uint8))
    return image

base\_model\_id = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet\_id = "diffusers/controlnet-depth-sdxl-1.0"
tcd\_lora\_id = "h1t/TCD-SDXL-LoRA"

controlnet = ControlNetModel.from\_pretrained(
    controlnet\_id,
    torch\_dtype=torch.float16,
    variant="fp16",
).to(device)
pipe = StableDiffusionXLControlNetPipeline.from\_pretrained(
    base\_model\_id,
    controlnet=controlnet,
    torch\_dtype=torch.float16,
    variant="fp16",
).to(device)
pipe.enable\_model\_cpu\_offload()

pipe.scheduler = TCDScheduler.from\_config(pipe.scheduler.config)

pipe.load\_lora\_weights(tcd\_lora\_id)
pipe.fuse\_lora()

prompt = "stormtrooper lecture, photorealistic"

image = load\_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
depth\_image = get\_depth\_map(image)

controlnet\_conditioning\_scale = 0.5  \# recommended for good generalization

image = pipe(
    prompt, 
    image=depth\_image, 
    num\_inference\_steps=4, 
    guidance\_scale=0,
    eta=0.3,
    controlnet\_conditioning\_scale=controlnet\_conditioning\_scale,
    generator=torch.Generator(device=device).manual\_seed(0),
).images\[0\]
grid\_image = make\_image\_grid(\[depth\_image, image\], rows=1, cols=2)
```
![](https://github.com/jabir-zheng/TCD/raw/main/assets/controlnet_depth_tcd.png)

### IP-Adapter

此示例说明如何将 TCD-LoRA 与 [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter/tree/main)和 SDXL 一起使用。

```py
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load\_image, make\_image\_grid

from ip\_adapter import IPAdapterXL
from scheduling\_tcd import TCDScheduler 

device = "cuda"
base\_model\_path = "stabilityai/stable-diffusion-xl-base-1.0"
image\_encoder\_path = "sdxl\_models/image\_encoder"
ip\_ckpt = "sdxl\_models/ip-adapter\_sdxl.bin"
tcd\_lora\_id = "h1t/TCD-SDXL-LoRA"

pipe = StableDiffusionXLPipeline.from\_pretrained(
    base\_model\_path, 
    torch\_dtype=torch.float16, 
    variant="fp16"
)
pipe.scheduler = TCDScheduler.from\_config(pipe.scheduler.config)

pipe.load\_lora\_weights(tcd\_lora\_id)
pipe.fuse\_lora()

ip\_model = IPAdapterXL(pipe, image\_encoder\_path, ip\_ckpt, device)

ref\_image = load\_image("https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/assets/images/woman.png").resize((512, 512))

prompt = "best quality, high quality, wearing sunglasses"

image = ip\_model.generate(
    pil\_image=ref\_image, 
    prompt=prompt,
    scale=0.5,
    num\_samples=1, 
    num\_inference\_steps=4, 
    guidance\_scale=0,
    eta=0.3, 
    seed=0,
)\[0\]

grid\_image = make\_image\_grid(\[ref\_image, image\], rows=1, cols=2)
```
![](https://github.com/jabir-zheng/TCD/raw/main/assets/ip_adapter.png)


### AnimateDiff 
`AnimateDiff`允许使用稳定扩散模型对图像进行动画处理。TCD-LoRA可以在不降低图像质量的情况下大大加快这一过程。TCD-LoRA 和 AnimateDiff 的动画质量具有更清晰的结果。

```py
import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from scheduling\_tcd import TCDScheduler
from diffusers.utils import export\_to\_gif

adapter = MotionAdapter.from\_pretrained("guoyww/animatediff-motion-adapter-v1-5")
pipe = AnimateDiffPipeline.from\_pretrained(
    "frankjoshua/toonyou\_beta6",
    motion\_adapter=adapter,
).to("cuda")

\# set TCDScheduler
pipe.scheduler = TCDScheduler.from\_config(pipe.scheduler.config)

\# load TCD LoRA
pipe.load\_lora\_weights("h1t/TCD-SD15-LoRA", adapter\_name="tcd")
pipe.load\_lora\_weights("guoyww/animatediff-motion-lora-zoom-in", weight\_name="diffusion\_pytorch\_model.safetensors", adapter\_name="motion-lora")

pipe.set\_adapters(\["tcd", "motion-lora"\], adapter\_weights=\[1.0, 1.2\])

prompt = "best quality, masterpiece, 1girl, looking at viewer, blurry background, upper body, contemporary, dress"
generator = torch.manual\_seed(0)
frames = pipe(
    prompt=prompt,
    num\_inference\_steps=5,
    guidance\_scale=0,
    cross\_attention\_kwargs={"scale": 1},
    num\_frames=24,
    eta=0.3,
    generator=generator
).frames\[0\]
export\_to\_gif(frames, "animation.gif")
```
![](https://github.com/jabir-zheng/TCD/raw/main/assets/animation_example.gif)