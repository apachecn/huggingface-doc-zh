# 图像到图像

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/img2img>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/img2img>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


图像到图像类似于
 [文本到图像]（条件图像生成）
 ，但除了提示之外，您还可以传递初始图像作为扩散过程的起点。初始图像被编码到潜在空间，并向其中添加噪声。然后，潜在扩散模型采用提示和噪声潜在图像，预测添加的噪声，并从初始潜在图像中去除预测的噪声以获得新的潜在图像。最后，解码器将新的潜在图像解码回图像。


使用 🤗 扩散器，这就像 1-2-3 一样简单：


1. 将检查点加载到
 [AutoPipelineForImage2Image](/docs/diffusers/v0.23.1/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForImage2Image)
 班级;该管道根据检查点自动处理加载正确的管道类：



```
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()
```


您会在整个指南中注意到，我们使用
 [启用\_model\_cpu\_offload()](/docs/diffusers/v0.23.1/en/api/pipelines/stable_diffusion/gligen#diffusers.StableDiffusionGLIGENTextImagePipeline.enable_model_cpu_offload)
 和
 [启用\_xformers\_memory\_efficient\_attention()](/docs/diffusers/v0.23.1/en/api/pipelines/stable_diffusion/adapter#diffusers.StableDiffusionXLAdapterPipeline.enable_xformers_memory_efficient_attention)
 ，以节省内存并提高推理速度。如果您使用的是 PyTorch 2.0，那么您不需要调用
 [启用\_xformers\_memory\_efficient\_attention()](/docs/diffusers/v0.23.1/en/api/pipelines/stable_diffusion/adapter#diffusers.StableDiffusionXLAdapterPipeline.enable_xformers_memory_efficient_attention)
 在你的管道上，因为它已经在使用 PyTorch 2.0 的原生
 [缩放点积注意力](../optimization/torch2.0#scaled-dot-product-attention)
 。


2. 加载图像以传递到管道：



```
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
```


3. 将提示和图像传递到管道以生成图像：



```
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png)

 初始图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img.png)

 生成的图像


## 热门车型



最流行的图像到图像模型是
 [稳定扩散 v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 ,
 [稳定扩散 XL (SDXL)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
 ， 和
 [康定斯基2.2](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)
 。稳定扩散模型和康定斯基模型的结果因其架构差异和训练过程而有所不同；通常，您可以期望 SDXL 生成比 Stable Diffusion v1.5 更高质量的图像。让我们快速了解一下如何使用每个模型并比较它们的结果。


### 


 稳定扩散 v1.5


Stable Diffusion v1.5 是从早期检查点初始化的潜在扩散模型，并针对 512x512 图像上的 595K 步骤进行了进一步微调。要使用此管道进行图像到图像，您需要准备一个初始图像以传递到管道。然后您可以将提示和图像传递到管道以生成新图像：



```
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png)

 初始图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdv1.5.png)

 生成的图像


### 


 稳定扩散 XL (SDXL)


SDXL 是稳定扩散模型的更强大版本。它使用更大的基础模型和附加的精炼模型来提高基础模型输出的质量。阅读
 [SDXL](sdxl)
 指南，更详细地介绍如何使用此模型以及它用于生成高质量图像的其他技术。



```
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, strength=0.5).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png)

 初始图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl.png)

 生成的图像


### 


 康定斯基2.2


康定斯基模型与稳定扩散模型不同，因为它使用图像先验模型来创建图像嵌入。嵌入有助于在文本和图像之间创建更好的对齐，从而允许潜在扩散模型生成更好的图像。


使用 Kandinsky 2.2 最简单的方法是：



```
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png)

 初始图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-kandinsky.png)

 生成的图像


## 配置管道参数



您可以在管道中配置几个重要参数，这些参数会影响图像生成过程和图像质量。让我们仔细看看这些参数的作用以及更改它们如何影响输出。


### 


 力量


‘力量’
 是需要考虑的最重要的参数之一，它将对您生成的图像产生巨大影响。它确定生成的图像与初始图像的相似程度。换句话说：


* 📈 更高
 ‘力量’
 value 赋予模型更多的“创造力”来生成与初始图像不同的图像； A
 ‘力量’
 值为 1.0 意味着初始图像或多或少被忽略
* 📉 较低
 ‘力量’
 value 表示生成的图像与初始图像更相似


这
 ‘力量’
 和
 `num_inference_steps`
 参数是相关的，因为
 ‘力量’
 确定要添加的噪声步数。例如，如果
 `num_inference_steps`
 是 50 并且
 ‘力量’
 是 0.8，那么这意味着向初始图像添加 40 (50 \* 0.8) 步的噪声，然后再进行 40 步的去噪以获得新生成的图像。



```
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, strength=0.8).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-strength-0.4.png)

 强度=0.4


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-strength-0.6.png)

 强度=0.6


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-strength-1.0.png)

 强度=1.0


### 


 指导尺度


这
 `指导规模`
 参数用于控制生成的图像和文本提示的对齐程度。更高的
 `指导规模`
 值意味着您生成的图像与提示更加一致，而较低
 `指导规模`
 值意味着您生成的图像有更多空间偏离提示。


你可以结合
 `指导规模`
 和
 ‘力量’
 更精确地控制模型的表现力。例如，结合高
 `强度+指导规模`
 以获得最大的创造力或使用低的组合
 ‘力量’
 和低
 `指导规模`
 生成类似于初始图像但不严格绑定到提示的图像。



```
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, guidance_scale=8.0).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-guidance-0.1.png)

 指导\_scale = 0.1


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-guidance-3.0.png)

 指导规模= 5.0


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-guidance-7.5.png)

 指导\_scale = 10.0


### 


 负面提示


负面提示条件使模型
 *不是*
 将事物包含在图像中，它可用于提高图像质量或修改图像。例如，您可以通过添加“细节较差”或“模糊”等负面提示来鼓励模型生成更高质量的图像，从而提高图像质量。或者，您可以通过指定要从图像中排除的内容来修改图像。



```
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"

# pass prompt and image to pipeline
image = pipeline(prompt, negative_prompt=negative_prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-negative-1.png)

 negative\_prompt =“丑陋、变形、毁容、细节差、解剖结构不良”


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-negative-2.png)

 否定\_prompt =“丛林”


## 链接的图像到图像管道



除了生成图像之外，还有其他一些有趣的方法可以使用图像到图像管道（尽管这也很酷）。您可以更进一步，将其与其他管道链接起来。


### 


 文本到图像到图像


通过链接文本到图像和图像到图像管道，您可以从文本生成图像，并将生成的图像用作图像到图像管道的初始图像。如果您想完全从头开始生成图像，这非常有用。例如，让我们链接一个稳定扩散模型和一个康定斯基模型。


首先使用文本到图像管道生成图像：



```
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
from diffusers.utils import make_image_grid

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

text2image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k").images[0]
text2image
```


现在您可以将此生成的图像传递到图像到图像管道：



```
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

image2image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", image=text2image).images[0]
make_image_grid([text2image, image2image], rows=1, cols=2)
```


### 


 图像到图像到图像


您还可以将多个图像到图像管道链接在一起以创建更有趣的图像。这对于在图像上迭代执行样式转换、生成短 GIF、恢复图像颜色或恢复图像的缺失区域非常有用。


首先生成图像：



```
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, output_type="latent").images[0]
```


重要的是要指定
 `output_type="潜在"`
 在管道中将所有输出保留在潜在空间中，以避免不必要的解码-编码步骤。仅当链接的管道使用相同的 VAE 时，这才有效。


将此管道的潜在输出传递到下一个管道以生成图像
 [漫画艺术风格](https://huggingface.co/ogkalu/Comic-Diffusion)
 :



```
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "ogkalu/Comic-Diffusion", torch_dtype=torch.float16
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# need to include the token "charliebo artstyle" in the prompt to use this checkpoint
image = pipeline("Astronaut in a jungle, charliebo artstyle", image=image, output_type="latent").images[0]
```


再重复一次以生成最终图像
 [像素艺术风格](https://huggingface.co/kohbanye/pixel-art-style)
 :



```
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kohbanye/pixel-art-style", torch_dtype=torch.float16
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# need to include the token "pixelartstyle" in the prompt to use this checkpoint
image = pipeline("Astronaut in a jungle, pixelartstyle", image=image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```


### 


 图像到放大器到超分辨率


链接图像到图像管道的另一种方法是使用升级器和超分辨率管道来真正提高图像的细节水平。


从图像到图像的管道开始：



```
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image_1 = pipeline(prompt, image=init_image, output_type="latent").images[0]
```


重要的是要指定
 `output_type="潜在"`
 在管道中以保留所有输出
 *潜*
 空间以避免不必要的解码编码步骤。仅当链接的管道使用相同的 VAE 时，这才有效。


将其链接到升级管道以提高图像分辨率：



```
from diffusers import StableDiffusionLatentUpscalePipeline

upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
    "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
upscaler.enable_model_cpu_offload()
upscaler.enable_xformers_memory_efficient_attention()

image_2 = upscaler(prompt, image=image_1, output_type="latent").images[0]
```


最后，将其链接到超分辨率管道以进一步提高分辨率：



```
from diffusers import StableDiffusionUpscalePipeline

super_res = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
super_res.enable_model_cpu_offload()
super_res.enable_xformers_memory_efficient_attention()

image_3 = super_res(prompt, image=image_2).images[0]
make_image_grid([init_image, image_3.resize((512, 512))], rows=1, cols=2)
```


## 控制图像生成



尝试生成看起来完全符合您想要的方式的图像可能很困难，这就是控制生成技术和模型如此有用的原因。虽然您可以使用
 `否定提示`
 为了部分控制图像生成，有更强大的方法，例如提示加权和 ControlNet。


### 


 提示称重


提示权重允许您缩放提示中每个概念的表示。例如，在“丛林中的宇航员，冷色调，柔和的颜色，详细，8k”这样的提示中，您可以选择增加或减少“宇航员”和“丛林”的嵌入。这
 [强制](https://github.com/damian0815/compel)
 库提供了一个简单的语法来调整提示权重和生成嵌入。您可以学习如何在中创建嵌入
 [提示权重](weighted_prompts)
 指导。


[AutoPipelineForImage2Image](/docs/diffusers/v0.23.1/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForImage2Image)
 有一个
 `提示嵌入`
 （和
 `否定提示嵌入`
 如果您使用否定提示）参数，您可以在其中传递替换的嵌入
 `提示`
 范围。



```
from diffusers import AutoPipelineForImage2Image
import torch

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt_embeds=prompt_embeds, # generated from Compel
    negative_prompt_embeds=negative_prompt_embeds, # generated from Compel
    image=init_image,
).images[0]
```


### 


 控制网


ControlNet 提供了一种更灵活、更准确的方法来控制图像生成，因为您可以使用额外的调节图像。调节图像可以是精明的图像、深度图、图像分割，甚至是涂鸦！无论您选择哪种类型的调节图像，ControlNet 都会生成一个保留其中信息的图像。


例如，让我们用深度图来调节图像以保留图像中的空间信息。



```
from diffusers.utils import load_image, make_image_grid

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)
init_image = init_image.resize((958, 960)) # resize to depth image dimensions
depth_image = load_image("https://huggingface.co/lllyasviel/control\_v11f1p\_sd15\_depth/resolve/main/images/control.png")
make_image_grid([init_image, depth_image], rows=1, cols=2)
```


加载以深度图为条件的 ControlNet 模型以及
 [AutoPipelineForImage2Image](/docs/diffusers/v0.23.1/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForImage2Image)
 :



```
from diffusers import ControlNetModel, AutoPipelineForImage2Image
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/control\_v11f1p\_sd15\_depth", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()
```


现在生成一个以深度图、初始图像和提示为条件的新图像：



```
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image_control_net = pipeline(prompt, image=init_image, control_image=depth_image).images[0]
make_image_grid([init_image, depth_image, image_control_net], rows=1, cols=3)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png)

 初始图像


![](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/images/control.png)

 深度图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-controlnet.png)

 ControlNet 图像


让我们应用一个新的
 [风格](https://huggingface.co/nitrosocke/elden-ring-diffusion)
 通过将其与图像到图像管道链接到从 ControlNet 生成的图像：



```
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "nitrosocke/elden-ring-diffusion", torch_dtype=torch.float16,
).to("cuda")
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

prompt = "elden ring style astronaut in a jungle" # include the token "elden ring style" in the prompt
negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"

image_elden_ring = pipeline(prompt, negative_prompt=negative_prompt, image=image_control_net, strength=0.45, guidance_scale=10.5).images[0]
make_image_grid([init_image, depth_image, image_control_net, image_elden_ring], rows=2, cols=2)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-elden-ring.png)


## 优化



运行扩散模型的计算成本很高且密集，但通过一些优化技巧，完全可以在消费级和免费 GPU 上运行它们。例如，您可以使用更节省内存的注意力形式，例如 PyTorch 2.0 的
 [缩放点积注意力](../optimization/torch2.0#scaled-dot-product-attention)
 或者
 [xFormers](../优化/xformers)
 （您可以使用其中之一，但无需同时使用两者）。您还可以将模型卸载到 GPU，而其他管道组件则在 CPU 上等待。



```
+ pipeline.enable\_model\_cpu\_offload()
+ pipeline.enable\_xformers\_memory\_efficient\_attention()
```


和
 [`torch.compile`](../optimization/torch2.0#torchcompile)
 ，您可以通过用它包装 UNet 来进一步提高推理速度：



```
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
```


要了解更多信息，请查看
 [减少内存使用](../优化/内存)
 和
 [火炬2.0](../优化/torch2.0)
 指南。