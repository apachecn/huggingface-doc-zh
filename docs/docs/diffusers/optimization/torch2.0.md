# 火炬2.0

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/optimization/torch2.0>
>
> 原始地址：<https://huggingface.co/docs/diffusers/optimization/torch2.0>


🤗 Diffusers 支持最新的优化
 [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/)
 其中包括：


1. 内存高效的注意力实现，缩放点积注意力，不需要任何额外的依赖项，例如 xFormers。
2. [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
 ，一个即时 (JIT) 编译器，可在编译各个模型时提供额外的性能提升。


这两项优化都需要 PyTorch 2.0 或更高版本以及 🤗 Diffusers > 0.13.0。



```
pip install --upgrade torch diffusers
```


## 缩放点积注意力



[`torch.nn.function.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.function.scaled_dot_product_attention)
 (SDPA) 是一种经过优化且内存高效的注意力机制（类似于 xFormers），可根据模型输入和 GPU 类型自动启用多种其他优化。如果您使用 PyTorch 2.0 和最新版本的 🤗 Diffusers，则默认启用 SDPA，因此您无需在代码中添加任何内容。


但是，如果您想显式启用它，您可以设置
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 使用
 [AttnProcessor2\_0](/docs/diffusers/v0.23.0/en/api/attnprocessor#diffusers.models.attention_processor.AttnProcessor2_0)
 ：



```
  import torch
  from diffusers import DiffusionPipeline
+ from diffusers.models.attention\_processor import AttnProcessor2\_0

  pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
+ pipe.unet.set\_attn\_processor(AttnProcessor2\_0())

  prompt = "a photo of an astronaut riding a horse on mars"
  image = pipe(prompt).images[0]
```


SDPA 的速度和内存效率应与
 `xFormers`
 ;检查
 [基准](#benchmark)
 更多细节。


在某些情况下 - 例如使管道更具确定性或将其转换为其他格式 - 使用普通注意力处理器可能会有所帮助，
 [AttnProcessor](/docs/diffusers/v0.23.0/en/api/attnprocessor#diffusers.models.attention_processor.AttnProcessor)
 。要恢复到
 [AttnProcessor](/docs/diffusers/v0.23.0/en/api/attnprocessor#diffusers.models.attention_processor.AttnProcessor)
 ，调用
 [set\_default\_attn\_processor()](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel.set_default_attn_processor)
 管道上的功能：



```
  import torch
  from diffusers import DiffusionPipeline
  from diffusers.models.attention_processor import AttnProcessor

  pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
+ pipe.unet.set\_default\_attn\_processor()

  prompt = "a photo of an astronaut riding a horse on mars"
  image = pipe(prompt).images[0]
```


## 火炬编译



这
 `火炬.编译`
 函数通常可以为您的 PyTorch 代码提供额外的加速。在 🤗 Diffusers 中，通常最好用以下方式包裹 UNet
 `火炬.编译`
 因为它完成了管道中的大部分繁重工作。



```
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
images = pipe(prompt, num_inference_steps=steps, num_images_per_prompt=batch_size).images[0]
```


根据 GPU 类型，
 `火炬.编译`
 可以提供一个
 *额外加速*
 的
 **5-300x**
 在 SDPA 之上！如果您使用的是更新的 GPU 架构，例如 Ampere（A100、3090）、Ada（4090）和 Hopper（H100），
 `火炬.编译`
 能够从这些 GPU 中榨取更多性能。


编译需要一些时间才能完成，因此它最适合您准备一次管道，然后多次执行相同类型的推理操作的情况。例如，在不同的图像大小上调用已编译的管道会再次触发编译，这可能会很昂贵。


有关更多信息和不同选项
 `火炬.编译`
 ，请参阅
 [`torch_compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
 教程。


## 基准



我们对 PyTorch 2.0 的高效注意力实现进行了全面的基准测试
 `火炬.编译`
 针对我们最常用的五个管道的不同 GPU 和批量大小。代码以🤗 Diffusers v0.17.0.dev0为基准进行优化
 `火炬.编译`
 用法（参见
 [此处](https://github.com/huggingface/diffusers/pull/3313)
 更多细节）。


展开下面的下拉列表以查找用于对每个管道进行基准测试的代码：



### 


 稳定的扩散文本到图像



```
from diffusers import DiffusionPipeline
import torch

path = "runwayml/stable-diffusion-v1-5"

run_compile = True  # Set True / False

pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")
pipe.unet.to(memory_format=torch.channels_last)

if run_compile:
    print("Run torch compile")
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

prompt = "ghibli style, a fantasy landscape with castles"

for _ in range(3):
    images = pipe(prompt=prompt).images
```


### 


 图像到图像的稳定扩散



```
from diffusers import StableDiffusionImg2ImgPipeline
import requests
import torch
from PIL import Image
from io import BytesIO

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((512, 512))

path = "runwayml/stable-diffusion-v1-5"

run_compile = True  # Set True / False

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(path, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")
pipe.unet.to(memory_format=torch.channels_last)

if run_compile:
    print("Run torch compile")
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

prompt = "ghibli style, a fantasy landscape with castles"

for _ in range(3):
    image = pipe(prompt=prompt, image=init_image).images[0]
```


### 


 稳定扩散修复



```
from diffusers import StableDiffusionInpaintPipeline
import requests
import torch
from PIL import Image
from io import BytesIO

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

def download\_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting\_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting\_examples/overture-creations-5sI6fQgYIuo\_mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

path = "runwayml/stable-diffusion-inpainting"

run_compile = True  # Set True / False

pipe = StableDiffusionInpaintPipeline.from_pretrained(path, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")
pipe.unet.to(memory_format=torch.channels_last)

if run_compile:
    print("Run torch compile")
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

prompt = "ghibli style, a fantasy landscape with castles"

for _ in range(3):
    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
```


### 


 控制网



```
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import requests
import torch
from PIL import Image
from io import BytesIO

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((512, 512))

path = "runwayml/stable-diffusion-v1-5"

run_compile = True  # Set True / False
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    path, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

pipe = pipe.to("cuda")
pipe.unet.to(memory_format=torch.channels_last)
pipe.controlnet.to(memory_format=torch.channels_last)

if run_compile:
    print("Run torch compile")
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    pipe.controlnet = torch.compile(pipe.controlnet, mode="reduce-overhead", fullgraph=True)

prompt = "ghibli style, a fantasy landscape with castles"

for _ in range(3):
    image = pipe(prompt=prompt, image=init_image).images[0]
```


### 


 DeepFloyd IF 文本转图像 + 放大



```
from diffusers import DiffusionPipeline
import torch

run_compile = True  # Set True / False

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", variant="fp16", text_encoder=None, torch_dtype=torch.float16, use_safetensors=True)
pipe.to("cuda")
pipe_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-M-v1.0", variant="fp16", text_encoder=None, torch_dtype=torch.float16, use_safetensors=True)
pipe_2.to("cuda")
pipe_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16, use_safetensors=True)
pipe_3.to("cuda")


pipe.unet.to(memory_format=torch.channels_last)
pipe_2.unet.to(memory_format=torch.channels_last)
pipe_3.unet.to(memory_format=torch.channels_last)

if run_compile:
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    pipe_2.unet = torch.compile(pipe_2.unet, mode="reduce-overhead", fullgraph=True)
    pipe_3.unet = torch.compile(pipe_3.unet, mode="reduce-overhead", fullgraph=True)

prompt = "the blue hulk"

prompt_embeds = torch.randn((1, 2, 4096), dtype=torch.float16)
neg_prompt_embeds = torch.randn((1, 2, 4096), dtype=torch.float16)

for _ in range(3):
    image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=neg_prompt_embeds, output_type="pt").images
    image_2 = pipe_2(image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=neg_prompt_embeds, output_type="pt").images
    image_3 = pipe_3(prompt=prompt, image=image, noise_level=100).images
```


下图突出显示了
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 使用 PyTorch 2.0 跨五个 GPU 系列
 `火炬.编译`
 已启用。下图的基准是在
 *迭代次数/秒*
 。


![t2i_speedup](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/pt2_benchmarks/t2i_speedup.png)


为了让您更好地了解这种加速如何适用于其他管道，请考虑以下内容
使用 PyTorch 2.0 的 A100 的图表和
 `火炬.编译`
 ：


![a100_numbers](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/pt2_benchmarks/a100_numbers.png)


在下表中，我们报告了我们的发现
 *迭代次数/秒*
 。


### 


 A100（批量：1）


| **Pipeline**  | **torch 2.0 -	 	 no compile**  | **torch nightly -	 	 no compile**  | **torch 2.0 -	 	 compile**  | **torch nightly -	 	 compile**  |
| --- | --- | --- | --- | --- |
| 	 SD - txt2img	  | 	 21.66	  | 	 23.13	  | 	 44.03	  | 	 49.74	  |
| 	 SD - img2img	  | 	 21.81	  | 	 22.40	  | 	 43.92	  | 	 46.32	  |
| 	 SD - inpaint	  | 	 22.24	  | 	 23.23	  | 	 43.76	  | 	 49.25	  |
| 	 SD - controlnet	  | 	 15.02	  | 	 15.82	  | 	 32.13	  | 	 36.08	  |
| 	 IF	  | 	 20.21 /	 	 13.84 /	 	 24.00	  | 	 20.12 /	 	 13.70 /	 	 24.03	  | 	 ❌	  | 	 97.34 /	 	 27.23 /	 	 111.66	  |
| 	 SDXL - txt2img	  | 	 8.64	  | 	 9.9	  | 	 -	  | 	 -	  |


### 


 A100（批量大小：4）


| **Pipeline**  | **torch 2.0 -	 	 no compile**  | **torch nightly -	 	 no compile**  | **torch 2.0 -	 	 compile**  | **torch nightly -	 	 compile**  |
| --- | --- | --- | --- | --- |
| 	 SD - txt2img	  | 	 11.6	  | 	 13.12	  | 	 14.62	  | 	 17.27	  |
| 	 SD - img2img	  | 	 11.47	  | 	 13.06	  | 	 14.66	  | 	 17.25	  |
| 	 SD - inpaint	  | 	 11.67	  | 	 13.31	  | 	 14.88	  | 	 17.48	  |
| 	 SD - controlnet	  | 	 8.28	  | 	 9.38	  | 	 10.51	  | 	 12.41	  |
| 	 IF	  | 	 25.02	  | 	 18.04	  | 	 ❌	  | 	 48.47	  |
| 	 SDXL - txt2img	  | 	 2.44	  | 	 2.74	  | 	 -	  | 	 -	  |


### 


 A100（批量：16）


| **Pipeline**  | **torch 2.0 -	 	 no compile**  | **torch nightly -	 	 no compile**  | **torch 2.0 -	 	 compile**  | **torch nightly -	 	 compile**  |
| --- | --- | --- | --- | --- |
| 	 SD - txt2img	  | 	 3.04	  | 	 3.6	  | 	 3.83	  | 	 4.68	  |
| 	 SD - img2img	  | 	 2.98	  | 	 3.58	  | 	 3.83	  | 	 4.67	  |
| 	 SD - inpaint	  | 	 3.04	  | 	 3.66	  | 	 3.9	  | 	 4.76	  |
| 	 SD - controlnet	  | 	 2.15	  | 	 2.58	  | 	 2.74	  | 	 3.35	  |
| 	 IF	  | 	 8.78	  | 	 9.82	  | 	 ❌	  | 	 16.77	  |
| 	 SDXL - txt2img	  | 	 0.64	  | 	 0.72	  | 	 -	  | 	 -	  |


### 


 V100（批量大小：1）


| **Pipeline**  | **torch 2.0 -	 	 no compile**  | **torch nightly -	 	 no compile**  | **torch 2.0 -	 	 compile**  | **torch nightly -	 	 compile**  |
| --- | --- | --- | --- | --- |
| 	 SD - txt2img	  | 	 18.99	  | 	 19.14	  | 	 20.95	  | 	 22.17	  |
| 	 SD - img2img	  | 	 18.56	  | 	 19.18	  | 	 20.95	  | 	 22.11	  |
| 	 SD - inpaint	  | 	 19.14	  | 	 19.06	  | 	 21.08	  | 	 22.20	  |
| 	 SD - controlnet	  | 	 13.48	  | 	 13.93	  | 	 15.18	  | 	 15.88	  |
| 	 IF	  | 	 20.01 /	 	 9.08 /	 	 23.34	  | 	 19.79 /	 	 8.98 /	 	 24.10	  | 	 ❌	  | 	 55.75 /	 	 11.57 /	 	 57.67	  |


### 


 V100（批量大小：4）


| **Pipeline**  | **torch 2.0 -	 	 no compile**  | **torch nightly -	 	 no compile**  | **torch 2.0 -	 	 compile**  | **torch nightly -	 	 compile**  |
| --- | --- | --- | --- | --- |
| 	 SD - txt2img	  | 	 5.96	  | 	 5.89	  | 	 6.83	  | 	 6.86	  |
| 	 SD - img2img	  | 	 5.90	  | 	 5.91	  | 	 6.81	  | 	 6.82	  |
| 	 SD - inpaint	  | 	 5.99	  | 	 6.03	  | 	 6.93	  | 	 6.95	  |
| 	 SD - controlnet	  | 	 4.26	  | 	 4.29	  | 	 4.92	  | 	 4.93	  |
| 	 IF	  | 	 15.41	  | 	 14.76	  | 	 ❌	  | 	 22.95	  |


### 


 V100（批量大小：16）


| **Pipeline**  | **torch 2.0 -	 	 no compile**  | **torch nightly -	 	 no compile**  | **torch 2.0 -	 	 compile**  | **torch nightly -	 	 compile**  |
| --- | --- | --- | --- | --- |
| 	 SD - txt2img	  | 	 1.66	  | 	 1.66	  | 	 1.92	  | 	 1.90	  |
| 	 SD - img2img	  | 	 1.65	  | 	 1.65	  | 	 1.91	  | 	 1.89	  |
| 	 SD - inpaint	  | 	 1.69	  | 	 1.69	  | 	 1.95	  | 	 1.93	  |
| 	 SD - controlnet	  | 	 1.19	  | 	 1.19	  | 	 OOM after warmup	  | 	 1.36	  |
| 	 IF	  | 	 5.43	  | 	 5.29	  | 	 ❌	  | 	 7.06	  |


### 


 T4（批量：1）


| **Pipeline**  | **torch 2.0 -	 	 no compile**  | **torch nightly -	 	 no compile**  | **torch 2.0 -	 	 compile**  | **torch nightly -	 	 compile**  |
| --- | --- | --- | --- | --- |
| 	 SD - txt2img	  | 	 6.9	  | 	 6.95	  | 	 7.3	  | 	 7.56	  |
| 	 SD - img2img	  | 	 6.84	  | 	 6.99	  | 	 7.04	  | 	 7.55	  |
| 	 SD - inpaint	  | 	 6.91	  | 	 6.7	  | 	 7.01	  | 	 7.37	  |
| 	 SD - controlnet	  | 	 4.89	  | 	 4.86	  | 	 5.35	  | 	 5.48	  |
| 	 IF	  | 	 17.42 /	 	 2.47 /	 	 18.52	  | 	 16.96 /	 	 2.45 /	 	 18.69	  | 	 ❌	  | 	 24.63 /	 	 2.47 /	 	 23.39	  |
| 	 SDXL - txt2img	  | 	 1.15	  | 	 1.16	  | 	 -	  | 	 -	  |


### 


 T4（批量大小：4）


| **Pipeline**  | **torch 2.0 -	 	 no compile**  | **torch nightly -	 	 no compile**  | **torch 2.0 -	 	 compile**  | **torch nightly -	 	 compile**  |
| --- | --- | --- | --- | --- |
| 	 SD - txt2img	  | 	 1.79	  | 	 1.79	  | 	 2.03	  | 	 1.99	  |
| 	 SD - img2img	  | 	 1.77	  | 	 1.77	  | 	 2.05	  | 	 2.04	  |
| 	 SD - inpaint	  | 	 1.81	  | 	 1.82	  | 	 2.09	  | 	 2.09	  |
| 	 SD - controlnet	  | 	 1.34	  | 	 1.27	  | 	 1.47	  | 	 1.46	  |
| 	 IF	  | 	 5.79	  | 	 5.61	  | 	 ❌	  | 	 7.39	  |
| 	 SDXL - txt2img	  | 	 0.288	  | 	 0.289	  | 	 -	  | 	 -	  |


### 


 T4（批量：16）


| **Pipeline**  | **torch 2.0 -	 	 no compile**  | **torch nightly -	 	 no compile**  | **torch 2.0 -	 	 compile**  | **torch nightly -	 	 compile**  |
| --- | --- | --- | --- | --- |
| 	 SD - txt2img	  | 	 2.34s	  | 	 2.30s	  | 	 OOM after 2nd iteration	  | 	 1.99s	  |
| 	 SD - img2img	  | 	 2.35s	  | 	 2.31s	  | 	 OOM after warmup	  | 	 2.00s	  |
| 	 SD - inpaint	  | 	 2.30s	  | 	 2.26s	  | 	 OOM after 2nd iteration	  | 	 1.95s	  |
| 	 SD - controlnet	  | 	 OOM after 2nd iteration	  | 	 OOM after 2nd iteration	  | 	 OOM after warmup	  | 	 OOM after warmup	  |
| 	 IF \*	  | 	 1.44	  | 	 1.44	  | 	 ❌	  | 	 1.94	  |
| 	 SDXL - txt2img	  | 	 OOM	  | 	 OOM	  | 	 -	  | 	 -	  |


### 


 RTX 3090（批量大小：1）


| **Pipeline**  | **torch 2.0 -	 	 no compile**  | **torch nightly -	 	 no compile**  | **torch 2.0 -	 	 compile**  | **torch nightly -	 	 compile**  |
| --- | --- | --- | --- | --- |
| 	 SD - txt2img	  | 	 22.56	  | 	 22.84	  | 	 23.84	  | 	 25.69	  |
| 	 SD - img2img	  | 	 22.25	  | 	 22.61	  | 	 24.1	  | 	 25.83	  |
| 	 SD - inpaint	  | 	 22.22	  | 	 22.54	  | 	 24.26	  | 	 26.02	  |
| 	 SD - controlnet	  | 	 16.03	  | 	 16.33	  | 	 17.38	  | 	 18.56	  |
| 	 IF	  | 	 27.08 /	 	 9.07 /	 	 31.23	  | 	 26.75 /	 	 8.92 /	 	 31.47	  | 	 ❌	  | 	 68.08 /	 	 11.16 /	 	 65.29	  |


### 


 RTX 3090（批量大小：4）


| **Pipeline**  | **torch 2.0 -	 	 no compile**  | **torch nightly -	 	 no compile**  | **torch 2.0 -	 	 compile**  | **torch nightly -	 	 compile**  |
| --- | --- | --- | --- | --- |
| 	 SD - txt2img	  | 	 6.46	  | 	 6.35	  | 	 7.29	  | 	 7.3	  |
| 	 SD - img2img	  | 	 6.33	  | 	 6.27	  | 	 7.31	  | 	 7.26	  |
| 	 SD - inpaint	  | 	 6.47	  | 	 6.4	  | 	 7.44	  | 	 7.39	  |
| 	 SD - controlnet	  | 	 4.59	  | 	 4.54	  | 	 5.27	  | 	 5.26	  |
| 	 IF	  | 	 16.81	  | 	 16.62	  | 	 ❌	  | 	 21.57	  |


### 


 RTX 3090（批量大小：16）


| **Pipeline**  | **torch 2.0 -	 	 no compile**  | **torch nightly -	 	 no compile**  | **torch 2.0 -	 	 compile**  | **torch nightly -	 	 compile**  |
| --- | --- | --- | --- | --- |
| 	 SD - txt2img	  | 	 1.7	  | 	 1.69	  | 	 1.93	  | 	 1.91	  |
| 	 SD - img2img	  | 	 1.68	  | 	 1.67	  | 	 1.93	  | 	 1.9	  |
| 	 SD - inpaint	  | 	 1.72	  | 	 1.71	  | 	 1.97	  | 	 1.94	  |
| 	 SD - controlnet	  | 	 1.23	  | 	 1.22	  | 	 1.4	  | 	 1.38	  |
| 	 IF	  | 	 5.01	  | 	 5.00	  | 	 ❌	  | 	 6.33	  |


### 


 RTX 4090（批量大小：1）


| **Pipeline**  | **torch 2.0 -	 	 no compile**  | **torch nightly -	 	 no compile**  | **torch 2.0 -	 	 compile**  | **torch nightly -	 	 compile**  |
| --- | --- | --- | --- | --- |
| 	 SD - txt2img	  | 	 40.5	  | 	 41.89	  | 	 44.65	  | 	 49.81	  |
| 	 SD - img2img	  | 	 40.39	  | 	 41.95	  | 	 44.46	  | 	 49.8	  |
| 	 SD - inpaint	  | 	 40.51	  | 	 41.88	  | 	 44.58	  | 	 49.72	  |
| 	 SD - controlnet	  | 	 29.27	  | 	 30.29	  | 	 32.26	  | 	 36.03	  |
| 	 IF	  | 	 69.71 /	 	 18.78 /	 	 85.49	  | 	 69.13 /	 	 18.80 /	 	 85.56	  | 	 ❌	  | 	 124.60 /	 	 26.37 /	 	 138.79	  |
| 	 SDXL - txt2img	  | 	 6.8	  | 	 8.18	  | 	 -	  | 	 -	  |


### 


 RTX 4090（批量大小：4）


| **Pipeline**  | **torch 2.0 -	 	 no compile**  | **torch nightly -	 	 no compile**  | **torch 2.0 -	 	 compile**  | **torch nightly -	 	 compile**  |
| --- | --- | --- | --- | --- |
| 	 SD - txt2img	  | 	 12.62	  | 	 12.84	  | 	 15.32	  | 	 15.59	  |
| 	 SD - img2img	  | 	 12.61	  | 	 12,.79	  | 	 15.35	  | 	 15.66	  |
| 	 SD - inpaint	  | 	 12.65	  | 	 12.81	  | 	 15.3	  | 	 15.58	  |
| 	 SD - controlnet	  | 	 9.1	  | 	 9.25	  | 	 11.03	  | 	 11.22	  |
| 	 IF	  | 	 31.88	  | 	 31.14	  | 	 ❌	  | 	 43.92	  |
| 	 SDXL - txt2img	  | 	 2.19	  | 	 2.35	  | 	 -	  | 	 -	  |


### 


 RTX 4090（批量大小：16）


| **Pipeline**  | **torch 2.0 -	 	 no compile**  | **torch nightly -	 	 no compile**  | **torch 2.0 -	 	 compile**  | **torch nightly -	 	 compile**  |
| --- | --- | --- | --- | --- |
| 	 SD - txt2img	  | 	 3.17	  | 	 3.2	  | 	 3.84	  | 	 3.85	  |
| 	 SD - img2img	  | 	 3.16	  | 	 3.2	  | 	 3.84	  | 	 3.85	  |
| 	 SD - inpaint	  | 	 3.17	  | 	 3.2	  | 	 3.85	  | 	 3.85	  |
| 	 SD - controlnet	  | 	 2.23	  | 	 2.3	  | 	 2.7	  | 	 2.75	  |
| 	 IF	  | 	 9.26	  | 	 9.2	  | 	 ❌	  | 	 13.31	  |
| 	 SDXL - txt2img	  | 	 0.52	  | 	 0.53	  | 	 -	  | 	 -	  |


## 笔记



* 按照这个
 [公关](https://github.com/huggingface/diffusers/pull/3313)
 有关用于进行基准测试的环境的更多详细信息。
* 对于批量大小 > 1 的 DeepFloyd IF 管道，我们仅在第一个 IF 管道中使用 > 1 的批量大小来生成文本到图像，而不是进行升级。这意味着两个升级管道收到的批量大小为 1。


*谢谢
 [何瑞斯](https://github.com/Chillee)
 来自 PyTorch 团队的支持，感谢他们在改善我们的支持方面提供的支持
 `torch.compile()`
 在扩散器中。*