# 自动流水线

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/tutorials/autopipeline>
>
> 原始地址：<https://huggingface.co/docs/diffusers/tutorials/autopipeline>


🤗 Diffusers 能够完成许多不同的任务，并且您通常可以将相同的预训练权重重复用于多个任务，例如文本到图像、图像到图像和修复。如果您对库和扩散模型不熟悉，可能很难知道要使用哪个管道来完成任务。例如，如果您正在使用
 [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 文本到图像的检查点，您可能不知道还可以通过使用以下命令加载检查点来将其用于图像到图像和修复
 [StableDiffusionImg2ImgPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/stable_diffusion/img2img#diffusers.StableDiffusionImg2ImgPipeline)
 和
 [StableDiffusionInpaintPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline)
 分别上课。


这
 `自动管道`
 类旨在简化 🤗 Diffusers 中的各种管道。它是一个通用的，
 *任务优先*
 让您专注于任务的管道。这
 `自动管道`
 自动检测要使用的正确管道类，这使得在不知道特定管道类名称的情况下更容易加载任务的检查点。


看看
 [自动管道](../api/pipelines/auto_pipeline)
 参考以查看支持哪些任务。目前，它支持文本转图像、图像转图像和修复。


本教程向您展示如何使用
 `自动管道`
 给定预训练权重，自动推断要为特定任务加载的管道类。


## 为您的任务选择 AutoPipeline



首先选择一个检查点。例如，如果您对文本到图像感兴趣
 [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 检查点，使用
 [AutoPipelineForText2Image](/docs/diffusers/v0.23.1/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForText2Image)
 :



```
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
prompt = "peasant and dragon combat, wood cutting style, viking era, bevel with rune"

image = pipeline(prompt, num_inference_steps=25).images[0]
image
```


![生成的木头切割风格农民斗龙图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-text2img.png)


 在引擎盖下，
 [AutoPipelineForText2Image](/docs/diffusers/v0.23.1/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForText2Image)
 :


1.自动检测
 `“稳定扩散”`
 类来自
 [`model_index.json`](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/model_index.json)
 文件
2.加载对应的文字转图片
 [StableDiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 基于
 `“稳定扩散”`
 班级名称


同样，对于图像到图像，
 [AutoPipelineForImage2Image](/docs/diffusers/v0.23.1/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForImage2Image)
 检测到一个
 `“稳定扩散”`
 检查站从
 `model_index.json`
 文件，它会加载相应的
 [StableDiffusionImg2ImgPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/stable_diffusion/img2img#diffusers.StableDiffusionImg2ImgPipeline)
 在幕后。您还可以传递特定于管道类的任何其他参数，例如
 ‘力量’
 ，它确定添加到输入图像的噪声或变化量：



```
from diffusers import AutoPipelineForImage2Image
import torch
import requests
from PIL import Image
from io import BytesIO

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")
prompt = "a portrait of a dog wearing a pearl earring"

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/1665\_Girl\_with\_a\_Pearl\_Earring.jpg/800px-1665\_Girl\_with\_a\_Pearl\_Earring.jpg"

response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")
image.thumbnail((768, 768))

image = pipeline(prompt, image, num_inference_steps=200, strength=0.75, guidance_scale=10.5).images[0]
image
```


![生成的戴着珍珠耳环的狗的维米尔肖像图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-img2img.png)


 如果你想做修复，那么
 [AutoPipelineForInpainting](/docs/diffusers/v0.23.1/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForInpainting)
 加载底层
 [StableDiffusionInpaintPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline)
 以同样的方式类：



```
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForInpainting.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting\_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting\_examples/overture-creations-5sI6fQgYIuo\_mask.png"

init_image = load_image(img_url).convert("RGB")
mask_image = load_image(mask_url).convert("RGB")

prompt = "A majestic tiger sitting on a bench"
image = pipeline(prompt, image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80).images[0]
image
```


![生成的老虎坐在长凳上的图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-inpaint.png)


 如果您尝试加载不受支持的检查点，则会抛出错误：



```
from diffusers import AutoPipelineForImage2Image
import torch

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "openai/shap-e-img2img", torch_dtype=torch.float16, use_safetensors=True
)
"ValueError: AutoPipeline can't find a pipeline linked to ShapEImg2ImgPipeline for None"
```


## 使用多个管道



对于某些工作流程或如果您要加载许多管道，从检查点重用相同的组件会更节省内存，而不是重新加载它们，这会不必要地消耗额外的内存。例如，如果您正在使用文本到图像的检查点，并且想要再次将其用于图像到图像，请使用
 [from\_pipe()](/docs/diffusers/v0.23.1/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForImage2Image.from_pipe)
 方法。此方法从先前加载的管道的组件创建新管道，无需额外的内存成本。


这
 [from\_pipe()](/docs/diffusers/v0.23.1/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForImage2Image.from_pipe)
 方法检测原始管道类并将其映射到与您想要执行的任务相对应的新管道类。例如，如果您加载一个
 `“稳定扩散”`
 文本到图像的类管道：



```
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch

pipeline_text2img = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
print(type(pipeline_text2img))
"<class 'diffusers.pipelines.stable\_diffusion.pipeline\_stable\_diffusion.StableDiffusionPipeline'>"
```


然后
 [from\_pipe()](/docs/diffusers/v0.23.1/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForImage2Image.from_pipe)
 映射原始的
 `“稳定扩散”`
 管道类至
 [StableDiffusionImg2ImgPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/stable_diffusion/img2img#diffusers.StableDiffusionImg2ImgPipeline)
 :



```
pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline_text2img)
print(type(pipeline_img2img))
"<class 'diffusers.pipelines.stable\_diffusion.pipeline\_stable\_diffusion\_img2img.StableDiffusionImg2ImgPipeline'>"
```


如果您将可选参数（例如禁用安全检查器）传递给原始管道，则该参数也会传递给新管道：



```
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch

pipeline_text2img = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
    requires_safety_checker=False,
).to("cuda")

pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline_text2img)
print(pipeline_img2img.config.requires_safety_checker)
"False"
```


如果您想更改新管道的行为，您可以覆盖原始管道中的任何参数甚至配置。例如，要重新打开安全检查器并添加
 ‘力量’
 争论：



```
pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline_text2img, requires_safety_checker=True, strength=0.3)
print(pipeline_img2img.config.requires_safety_checker)
"True"
```