# 控制网

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/controlnet>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/controlnet>


ControlNet 是一种通过使用附加输入图像调节模型来控制图像扩散模型的模型。您可以使用多种类型的调节输入（精明边缘、用户草图、人体姿势、深度等）来控制扩散模型。这非常有用，因为它使您可以更好地控制图像生成，从而更轻松地生成特定图像，而无需尝试不同的文本提示或去噪值。


请参阅第 3.5 节
 [ControlNet](https://huggingface.co/papers/2302.05543)
 有关各种调节输入的 ControlNet 实现列表的论文。您可以在以下位置找到官方稳定扩散 ControlNet 条件模型
 [lllyasviel](https://huggingface.co/lllyasviel)
 的集线器简介等
 [社区培训](https://huggingface.co/models?other=stable-diffusion&other=controlnet)
 集线器上的那些。


对于 Stable Diffusion XL (SDXL) ControlNet 模型，您可以在 🤗 上找到它们
 [扩散器](https://huggingface.co/diffusers)
 中心组织，或者您可以浏览
 [社区培训](https://huggingface.co/models?other=stable-diffusion-xl&other=controlnet)
 集线器上的那些。


ControlNet 模型具有两组通过零卷积层连接的权重（或块）：


* A
 *锁定副本*
 保留大型预训练扩散模型学到的所有内容
* A
 *可训练副本*
 接受额外调节输入的训练


由于锁定的副本保留了预训练的模型，因此在新的条件输入上训练和实现 ControlNet 与微调任何其他模型一样快，因为您不是从头开始训练模型。


本指南将向您展示如何使用 ControlNet 进行文本到图像、图像到图像、修复等！有多种类型的 ControlNet 调节输入可供选择，但在本指南中我们将只关注其中的几种。请随意尝试其他调节输入！


在开始之前，请确保已安装以下库：



```
# uncomment to install the necessary libraries in Colab
#!pip install diffusers transformers accelerate safetensors opencv-python
```


## 文本转图像



对于文本到图像，您通常将文本提示传递给模型。但通过 ControlNet，您可以指定额外的调节输入。让我们用精明的图像（黑色背景上的白色轮廓图像）来调节模型。这样，ControlNet就可以使用canny图像作为控制，引导模型生成具有相同轮廓的图像。


加载图像并使用
 [opencv-python](https://github.com/opencv/opencv-python)
 提取canny图像的库：



```
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np

image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input\_image\_vermeer.png"
)

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png)

 原始图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/vermeer_canny_edged.png)

 精明的图像


接下来，加载以 canny 边缘检测为条件的 ControlNet 模型并将其传递给
 [StableDiffusionControlNetPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetPipeline)
 。使用速度越快
 [UniPCMultistepScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)
 并启用模型卸载以加速推理并减少内存使用。



```
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```


现在将您的提示和精明的图像传递到管道：



```
output = pipe(
    "the mona lisa", image=canny_image
).images[0]
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-text2img.png)


## 图像到图像



对于图像到图像，您通常会将初始图像和提示传递给管道以生成新图像。借助 ControlNet，您可以传递额外的调节输入来指导模型。让我们用深度图（包含空间信息的图像）来调节模型。这样，ControlNet 就可以使用深度图作为控制来指导模型生成保留空间信息的图像。


您将使用
 [StableDiffusionControlNetImg2ImgPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetImg2ImgPipeline)
 对于这个任务，它不同于
 [StableDiffusionControlNetPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetPipeline)
 因为它允许您传递初始图像作为图像生成过程的起点。


加载图像并使用
 `深度估计`
[管道](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/pipelines#transformers.Pipeline)
 从🤗 Transformers 中提取图像的深度图：



```
import torch
import numpy as np

from transformers import pipeline
from diffusers.utils import load_image

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg"
).resize((768, 768))


def get\_depth\_map(image, depth\_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map

depth_estimator = pipeline("depth-estimation")
depth_map = get_depth_map(image, depth_estimator).unsqueeze(0).half().to("cuda")
```


接下来，加载以深度图为条件的 ControlNet 模型并将其传递给
 [StableDiffusionControlNetImg2ImgPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetImg2ImgPipeline)
 。使用速度越快
 [UniPCMultistepScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)
 并启用模型卸载以加速推理并减少内存使用。



```
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/control\_v11f1p\_sd15\_depth", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```


现在将提示、初始图像和深度图传递到管道：



```
output = pipe(
    "lego batman and robin", image=image, control_image=depth_map,
).images[0]
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg)

 原始图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img-2.png)

 生成的图像


## 修复



对于修复，您需要一个初始图像、一个蒙版图像和一个描述用什么替换蒙版的提示。 ControlNet 模型允许您添加另一个控制图像来调节模型。让我们用精明的图像（黑色背景上的白色轮廓图像）来调节模型。这样，ControlNet就可以使用canny图像作为控制，引导模型生成具有相同轮廓的图像。


加载初始图像和掩模图像：



```
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch

init_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint.jpg"
)
init_image = init_image.resize((512, 512))

mask_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint-mask.jpg"
)
mask_image = mask_image.resize((512, 512))
```


创建一个函数以根据初始图像和掩模图像准备控制图像。这将创建一个张量来标记其中的像素
 `初始化映像`
 如果相应的像素在
 `遮罩图像`
 超过了一定的阈值。



```
def make\_inpaint\_condition(image, image\_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1]
    image[image_mask > 0.5] = 1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

control_image = make_inpaint_condition(init_image, mask_image)
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint.jpg)

 原始图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint-mask.jpg)

 掩模图像


加载以修复为条件的 ControlNet 模型并将其传递给
 [StableDiffusionControlNetInpaintPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetInpaintPipeline)
 。使用速度越快
 [UniPCMultistepScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)
 并启用模型卸载以加速推理并减少内存使用。



```
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/control\_v11p\_sd15\_inpaint", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```


现在将提示、初始图像、遮罩图像和控制图像传递到管道：



```
output = pipe(
    "corgi face with large ears, detailed, pixar, animated, disney",
    num_inference_steps=20,
    eta=1.0,
    image=init_image,
    mask_image=mask_image,
    control_image=control_image,
).images[0]
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint-result.png)


## 猜测模式



[猜测模式](https://github.com/lllyasviel/ControlNet/discussions/188)
 根本不需要向 ControlNet 提供提示！这迫使 ControlNet 编码器尽力“猜测”输入控制图的内容（深度图、姿态估计、canny 边缘等）。


猜测模式根据块深度以固定比率调整 ControlNet 输出残差的比例。最浅的
 `下块`
 对应于0.1，随着块的加深，规模呈指数级增长，使得
 `中块`
 输出变为1.0。


猜测模式对提示调节没有任何影响，如果需要，您仍然可以提供提示。


放
 `guess_mode=True`
 在管道中，它是
 [推荐](https://github.com/lllyasviel/ControlNet#guess-mode--non-prompt-mode)
 设置
 `指导规模`
 值介于 3.0 和 5.0 之间。



```
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, use_safetensors=True).to(
    "cuda"
)
image = pipe("", image=canny_image, guess_mode=True, guidance_scale=3.0).images[0]
image
```


![](https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare_guess_mode/output_images/diffusers/output_bird_canny_0.png)

 带提示的常规模式


![](https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare_guess_mode/output_images/diffusers/output_bird_canny_0_gm.png)

 无提示猜测模式


## ControlNet 具有稳定扩散 XL



目前没有太多与 Stable Diffusion XL (SDXL) 兼容的 ControlNet 模型，但我们已经根据精明的边缘检测和深度图训练了两个用于 SDXL 的全尺寸 ControlNet 模型。我们还尝试创建这些兼容 SDXL 的 ControlNet 模型的较小版本，以便更容易在资源受限的硬件上运行。您可以在 🤗 上找到这些检查点
 [扩散器](https://huggingface.co/diffusers)
 中心组织！


让我们使用以 canny 图像为条件的 SDXL ControlNet 来生成图像。首先加载图像并准备 canny 图像：



```
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np

image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd\_controlnet/hf-logo.png"
)

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
canny_image
```


![](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png)

 原始图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hf-logo-canny.png)

 精明的图像


加载以 Canny 边缘检测为条件的 SDXL ControlNet 模型，并将其传递给
 [StableDiffusionXLControlNetPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetPipeline)
 。您还可以启用模型卸载以减少内存使用。



```
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.enable_model_cpu_offload()
```


现在将您的提示（如果您正在使用提示，还可以选择是否定提示）和精明的图像传递到管道：


这
 [`controlnet_conditioning_scale`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetPipeline.__call__.controlnet_conditioning_scale)
 参数决定分配给调节输入的权重。为了获得良好的泛化效果，建议使用值 0.5，但请随意尝试这个数字！



```
prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = 'low quality, bad quality, sketches'

images = pipe(
    prompt,
    negative_prompt=negative_prompt,
    image=canny_image,
    controlnet_conditioning_scale=0.5,
).images[0]
images
```


![](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/out_hug_lab_7.png)


 您可以使用
 [StableDiffusionXLControlNetPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetPipeline)
 在猜测模式下也可以通过将参数设置为
 '真实'
 :



```
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch

import cv2
from PIL import Image

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = "low quality, bad quality, sketches"

image = load_image(
    "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd\_controlnet/hf-logo.png"
)

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16, use_safetensors=True
)
pipe.enable_model_cpu_offload()

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

image = pipe(
    prompt, controlnet_conditioning_scale=0.5, image=canny_image, guess_mode=True,
).images[0]
```


### 


 多控制网


将 SDXL 模型替换为类似模型
 [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 通过稳定扩散模型使用多个调节输入。


您可以根据不同的图像输入组合多个 ControlNet 条件来创建
 *多控制网络*
 。为了获得更好的结果，以下操作通常会有所帮助：


1. 遮盖条件，使其不重叠（例如，遮盖精明图像中姿势条件所在的区域）
2. 进行实验
 [`controlnet_conditioning_scale`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetPipeline.__call__.controlnet_conditioning_scale)
 参数来确定为每个调节输入分配多少权重


在此示例中，您将组合精明图像和人体姿势估计图像来生成新图像。


准备精明的图像调节：



```
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import cv2

canny_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
)
canny_image = np.array(canny_image)

low_threshold = 100
high_threshold = 200

canny_image = cv2.Canny(canny_image, low_threshold, high_threshold)

# zero out middle columns of image where pose will be overlaid
zero_start = canny_image.shape[1] // 4
zero_end = zero_start + canny_image.shape[1] // 2
canny_image[:, zero_start:zero_end] = 0

canny_image = canny_image[:, :, None]
canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
canny_image = Image.fromarray(canny_image).resize((1024, 1024))
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png)

 原始图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/landscape_canny_masked.png)

 精明的图像


准备人体姿势估计条件：



```
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image

openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

openpose_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"
)
openpose_image = openpose(openpose_image).resize((1024, 1024))
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png)

 原始图像


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/person_pose.png)

 人体姿势图像


加载与每个条件对应的 ControlNet 模型列表，并将它们传递给
 [StableDiffusionXLControlNetPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetPipeline)
 。使用速度越快
 [UniPCMultistepScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)
 并启用模型卸载以减少内存使用。



```
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
import torch

controlnets = [
    ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True
    ),
    ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True
    ),
]

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnets, vae=vae, torch_dtype=torch.float16, use_safetensors=True
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```


现在，您可以将提示（如果您正在使用提示，则可以选择否定提示）、精明图像并将图像摆在管道中：



```
prompt = "a giant standing in a fantasy landscape, best quality"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

generator = torch.manual_seed(1)

images = [openpose_image, canny_image]

images = pipe(
    prompt,
    image=images,
    num_inference_steps=25,
    generator=generator,
    negative_prompt=negative_prompt,
    num_images_per_prompt=3,
    controlnet_conditioning_scale=[1.0, 0.8],
).images[0]
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/multicontrolnet.png)