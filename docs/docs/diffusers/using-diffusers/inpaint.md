# 修复

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/inpaint>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/inpaint>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


修复会替换或编辑图像的特定区域。这使其成为图像修复的有用工具，例如消除缺陷和伪影，甚至用全新的东西替换图像区域。修复依靠蒙版来确定要填充图像的哪些区域；要修复的区域由白色像素表示，要保留的区域由黑色像素表示。白色像素由提示填充。


使用 🤗 扩散器，您可以按照以下方式进行修复：


1. 使用以下命令加载修复检查点
 [AutoPipelineForInpainting](/docs/diffusers/v0.23.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForInpainting)
 班级。这将根据检查点自动检测要加载的适当管道类：



```
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()
```


您会在整个指南中注意到，我们使用
 [启用\_model\_cpu\_offload()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/gligen#diffusers.StableDiffusionGLIGENTextImagePipeline.enable_model_cpu_offload)
 和
 [启用\_xformers\_memory\_efficient\_attention()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.StableDiffusionImageVariationPipeline.enable_xformers_memory_efficient_attention)
 ，以节省内存并提高推理速度。如果您使用的是 PyTorch 2.0，则无需调用
 [启用\_xformers\_memory\_efficient\_attention()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.StableDiffusionImageVariationPipeline.enable_xformers_memory_efficient_attention)
 在你的管道上，因为它已经在使用 PyTorch 2.0 的原生
 [缩放点积注意力](../optimization/torch2.0#scaled-dot-product-attention)
 。


2. 加载基础图像和蒙版图像：



```
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint\_mask.png")
```


3. 创建一个提示来修复图像，并将其与基础图像和蒙版图像一起传递到管道：



```
prompt = "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
negative_prompt = "bad anatomy, deformed, ugly, disfigured"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png)

 基础图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png)

 掩模图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-cat.png)

 生成的图像


## 创建蒙版图像



为了方便起见，在本指南中，所有代码示例中都提供了遮罩图像。您可以修复自己的图像，但需要为其创建蒙版图像。使用下面的空间轻松创建蒙版图像。


上传要修复的基本图像，然后使用草图工具绘制蒙版。完成后，单击
 **跑步**
 生成并下载掩模图像。


## 热门车型



[稳定扩散修复](https://huggingface.co/runwayml/stable-diffusion-inpainting)
 ,
 [稳定扩散 XL (SDXL) 修复](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
 ， 和
 [康定斯基2.2修复](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint)
 是最受欢迎的修复模型之一。 SDXL 通常会生成比 Stable Diffusion v1.5 更高分辨率的图像，Kandinsky 2.2 也能够生成高质量图像。


### 


 稳定扩散修复


稳定扩散修复是一种在修复时对 512x512 图像进行微调的潜在扩散模型。这是一个很好的起点，因为它相对较快并生成高质量的图像。要使用此模型进行修复，您需要将提示、基础图像和遮罩图像传递到管道：



```
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint\_mask.png")

generator = torch.Generator("cuda").manual_seed(92)
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```


### 


 稳定扩散 XL (SDXL) 修复


SDXL 是 Stable Diffusion v1.5 的更大、更强大的版本。该模型可以遵循两阶段模型过程（尽管每个模型也可以单独使用）；基础模型生成图像，精炼模型获取该图像并进一步增强其细节和质量。看看
 [SDXL](sdxl)
 指南，获取有关如何使用 SDXL 及其参数配置的更全面指南。



```
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint\_mask.png")

generator = torch.Generator("cuda").manual_seed(92)
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```


### 


 康定斯基2.2 修复


康定斯基模型系列与 SDXL 类似，因为它也使用两种模型；图像先验模型创建图像嵌入，扩散模型从中生成图像。您可以分别加载图像先验和扩散模型，但使用 Kandinsky 2.2 最简单的方法是将其加载到
 [AutoPipelineForInpainting](/docs/diffusers/v0.23.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForInpainting)
 使用的类
 [KandinskyV22InpaintCombinedPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/kandinsky_v22#diffusers.KandinskyV22InpaintCombinedPipeline)
 在引擎盖下。



```
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint\_mask.png")

generator = torch.Generator("cuda").manual_seed(92)
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png)

 基础图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-sdv1.5.png)

 稳定扩散修复


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-sdxl.png)

 稳定扩散 XL 修复


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-kandinsky.png)

 康定斯基2.2 修复


## 配置管道参数



图像特征（例如质量和“创造力”）取决于管道参数。了解这些参数的作用对于获得您想要的结果非常重要。让我们看一下最重要的参数，看看更改它们如何影响输出。


### 


 力量


‘力量’
 是对基础图像添加了多少噪声的度量，这会影响输出与基础图像的相似程度。


*📈高
 ‘力量’
 值意味着图像中添加了更多噪声，并且去噪过程需要更长的时间，但您将获得与基础图像差异更大的更高质量的图像
* 📉 低
 ‘力量’
 值意味着向图像添加的噪声更少，去噪过程更快，但图像质量可能不太好，生成的图像更类似于基础图像



```
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint\_mask.png")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.6).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-strength-0.6.png)

 强度=0.6


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-strength-0.8.png)

 强度=0.8


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-strength-1.0.png)

 强度=1.0


### 


 指导尺度


`指导规模`
 影响文本提示和生成图像的对齐方式。


*📈高
 `指导规模`
 value 表示提示和生成的图像紧密对齐，因此输出是对提示的更严格的解释
* 📉 低
 `指导规模`
 值意味着提示和生成的图像更加松散地对齐，因此输出可能与提示更加不同


您可以使用
 ‘力量’
 和
 `指导规模`
 一起更好地控制模型的表现力。例如，组合高
 ‘力量’
 和
 `指导规模`
 值赋予模型最大的创作自由度。



```
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint\_mask.png")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, guidance_scale=2.5).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-guidance-2.5.png)

 指导规模= 2.5


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-guidance-7.5.png)

 指导规模= 7.5


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-guidance-12.5.png)

 指导规模= 12.5


### 


 负面提示


否定提示承担与提示相反的角色；它引导模型远离在图像中生成某些东西。这对于快速提高图像质量并防止模型生成您不想要的东西非常有用。



```
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint\_mask.png")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
negative_prompt = "bad architecture, unstable, poor details, blurry"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-negative.png)

 negative\_prompt =“糟糕的架构，不稳定，细节差，模糊”


## 保留未遮盖的区域



这
 [AutoPipelineForInpainting](/docs/diffusers/v0.23.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForInpainting)
 （和其他修复管道）通常会更改图像的未屏蔽部分，以在屏蔽和未屏蔽区域之间创建更自然的过渡。如果这种行为是不可取的，您可以强制未遮蔽的区域保持不变。然而，强制图像的未遮蔽部分保持不变可能会导致未遮蔽区域和遮蔽区域之间出现一些不寻常的过渡。



```
import PIL
import numpy as np
import torch

from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

device = "cuda"
pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)
pipeline = pipeline.to(device)

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting\_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting\_examples/overture-creations-5sI6fQgYIuo\_mask.png"

init_image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
repainted_image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
repainted_image.save("repainted\_image.png")

# Convert mask to grayscale NumPy array
mask_image_arr = np.array(mask_image.convert("L"))
# Add a channel dimension to the end of the grayscale mask
mask_image_arr = mask_image_arr[:, :, None]
# Binarize the mask: 1s correspond to the pixels which are repainted
mask_image_arr = mask_image_arr.astype(np.float32) / 255.0
mask_image_arr[mask_image_arr < 0.5] = 0
mask_image_arr[mask_image_arr >= 0.5] = 1

# Take the masked pixels from the repainted image and the unmasked pixels from the initial image
unmasked_unchanged_image_arr = (1 - mask_image_arr) * init_image + mask_image_arr * repainted_image
unmasked_unchanged_image = PIL.Image.fromarray(unmasked_unchanged_image_arr.round().astype("uint8"))
unmasked_unchanged_image.save("force\_unmasked\_unchanged.png")
make_image_grid([init_image, mask_image, repainted_image, unmasked_unchanged_image], rows=2, cols=2)
```


## 连锁修补管道



[AutoPipelineForInpainting](/docs/diffusers/v0.23.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForInpainting)
 可以与其他 🤗 Diffusers 管道链接以编辑其输出。这对于提高其他扩散管道的输出质量通常很有用，并且如果您使用多个管道，则将它们链接在一起以将输出保留在潜在空间中并重用相同的管道组件可以更节省内存。


### 


 文本到图像到修复


链接文本到图像和修复管道允许您修复生成的图像，并且您不必首先提供基础图像。这使得您可以方便地编辑您喜欢的文本到图像输出，而无需生成全新的图像。


从文本到图像管道开始创建一座城堡：



```
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

text2image = pipeline("concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k").images[0]
```


加载上面输出的掩模图像：



```
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint\_text-chain-mask.png")
```


让我们用瀑布修复遮罩区域：



```
pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

prompt = "digital painting of a fantasy waterfall, cloudy"
image = pipeline(prompt=prompt, image=text2image, mask_image=mask_image).images[0]
make_image_grid([text2image, mask_image, image], rows=1, cols=3)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-text-chain.png)

 文本到图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-text-chain-out.png)

 修复


### 


 修复到图像到图像


您还可以将修复管道链接到另一个管道（例如图像到图像或升级器）之前，以提高质量。


首先修复图像：



```
import torch
from diffusers import AutoPipelineForInpainting, AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint\_mask.png")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image_inpainting = pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

# resize image to 1024x1024 for SDXL
image_inpainting = image_inpainting.resize((1024, 1024))
```


现在，让我们使用 SDXL 的细化器模型将图像传递到另一个修复管道，以增强图像细节和质量：



```
pipeline = AutoPipelineForInpainting.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt=prompt, image=image_inpainting, mask_image=mask_image, output_type="latent").images[0]
```


重要的是要指定
 `output_type="潜在"`
 在管道中将所有输出保留在潜在空间中，以避免不必要的解码-编码步骤。仅当链接的管道使用相同的 VAE 时，这才有效。例如，在
 [文本转图像修复](#文本转图像修复)
 部分，Kandinsky 2.2 使用与稳定扩散模型不同的 VAE 类，因此它不起作用。但是，如果您对两个管道都使用 Stable Diffusion v1.5，那么您可以将所有内容保留在潜在空间中，因为它们都使用
 [AutoencoderKL](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)
 。


最后，您可以将此图像传递到图像到图像管道以对其进行最后的修饰。使用更有效
 [from\_pipe()](/docs/diffusers/v0.23.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForImage2Image.from_pipe)
 方法重用现有的管道组件，并避免不必要地再次将所有管道组件加载​​到内存中。



```
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt=prompt, image=image).images[0]
make_image_grid([init_image, mask_image, image_inpainting, image], rows=2, cols=2)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png)

 初始图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-to-image-chain.png)

 修复


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-to-image-final.png)

 图像到图像


图像到图像和修复实际上是非常相似的任务。图像到图像生成类似于现有提供的图像的新图像。修复做同样的事情，但它只变换由蒙版定义的图像区域，图像的其余部分保持不变。您可以将修复视为进行特定更改的更精确的工具，而图像到图像具有更广泛的范围来进行更彻底的更改。


## 控制图像生成



让图像看起来完全符合您想要的方式是具有挑战性的，因为去噪过程是随机的。虽然您可以通过配置参数来控制生成的某些方面，例如
 `否定提示`
 ，有更好、更有效的方法来控制图像生成。


### 


 提示称重


提示权重提供了一种可量化的方法来衡量提示中概念的表示。您可以使用它来增加或减少提示中每个概念的文本嵌入向量的大小，这随后决定了每个概念的生成量。这
 [强制](https://github.com/damian0815/compel)
 库提供了一种直观的语法来缩放提示权重和生成嵌入。了解如何在中创建嵌入
 [提示权重](../using-diffusers/weighted_prompts)
 指导。


生成嵌入后，将它们传递给
 `提示嵌入`
 （和
 `否定提示嵌入`
 如果您使用否定提示）参数
 [AutoPipelineForInpainting](/docs/diffusers/v0.23.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForInpainting)
 。嵌入取代了
 `提示`
 范围：



```
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16,
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt_embeds=prompt_embeds, # generated from Compel
    negative_prompt_embeds=negative_prompt_embeds, # generated from Compel
    image=init_image,
    mask_image=mask_image
).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```


### 


 控制网


ControlNet 模型与稳定扩散等其他扩散模型一起使用，它们提供了一种更灵活、更准确的方法来控制图像的生成方式。 ControlNet 接受额外的调节图像输入，指导扩散模型保留其中的特征。


例如，让我们使用在修复图像上预训练的 ControlNet 来调节图像：



```
import torch
import numpy as np
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers.utils import load_image, make_image_grid

# load ControlNet
controlnet = ControlNetModel.from_pretrained("lllyasviel/control\_v11p\_sd15\_inpaint", torch_dtype=torch.float16, variant="fp16")

# pass ControlNet to the pipeline
pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint\_mask.png")

# prepare control image
def make\_inpaint\_condition(init\_image, mask\_image):
    init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
    mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

    assert init_image.shape[0:1] == mask_image.shape[0:1], "image and image\_mask must have the same image size"
    init_image[mask_image > 0.5] = -1.0  # set as masked pixel
    init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image)
    return init_image

control_image = make_inpaint_condition(init_image, mask_image)
```


现在从基础图像、蒙版图像和控制图像生成图像。您会注意到生成的图像中充分保留了基础图像的特征。



```
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, control_image=control_image).images[0]
make_image_grid([init_image, mask_image, PIL.Image.fromarray(np.uint8(control_image[0][0])).convert('RGB'), image], rows=2, cols=2)
```


您可以更进一步，将其与图像到图像管道链接起来以应用新的
 [风格](https://huggingface.co/nitrosocke/elden-ring-diffusion)
 :



```
from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "nitrosocke/elden-ring-diffusion", torch_dtype=torch.float16,
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

prompt = "elden ring style castle" # include the token "elden ring style" in the prompt
negative_prompt = "bad architecture, deformed, disfigured, poor details"

image_elden_ring = pipeline(prompt, negative_prompt=negative_prompt, image=image).images[0]
make_image_grid([init_image, mask_image, image, image_elden_ring], rows=2, cols=2)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png)

 初始图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-controlnet.png)

 ControlNet修复


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint-img2img.png)

 图像到图像


## 优化



如果资源有限，运行扩散模型可能会很困难且缓慢，但不一定需要一些优化技巧。您可以启用的最大（也是最简单）的优化之一是切换到内存高效的注意力。如果您使用的是 PyTorch 2.0，
 [缩放点积注意力](../optimization/torch2.0#scaled-dot-product-attention)
 会自动启用，您无需执行任何其他操作。对于非PyTorch 2.0用户，您可以安装并使用
 [xFormers](../优化/xformers)
 内存高效注意力的实现。这两个选项都可以减少内存使用并加速推理。


您还可以将模型卸载到 CPU 以节省更多内存：



```
+ pipeline.enable\_xformers\_memory\_efficient\_attention()
+ pipeline.enable\_model\_cpu\_offload()
```


要进一步加快推理代码的速度，请使用
 [`torch_compile`](../optimization/torch2.0#torchcompile)
 。你应该包裹
 `torch.编译`
 围绕管道中最密集的组件（通常是 UNet）：



```
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
```


了解更多信息
 [减少内存使用](../优化/内存)
 和
 [torch2.0](../优化/torch2.0)
 指南。