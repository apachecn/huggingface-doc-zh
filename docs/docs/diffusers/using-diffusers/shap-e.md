# Shap-E

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/shap-e>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/shap-e>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


Shap-E 是一种用于生成 3D 资产的条件模型，可用于视频游戏开发、室内设计和建筑。它在大型 3D 资产数据集上进行训练，并进行后处理以渲染每个对象的更多视图并生成 16K 而不是 4K 点云。 Shap-E模型的训练分两个步骤：


1. 编码器接受 3D 资产的点云和渲染视图，并输出代表该资产的隐式函数的参数
2. 根据编码器产生的潜在特征训练扩散模型，以生成神经辐射场 (NeRF) 或纹理 3D 网格，从而更轻松地在下游应用程序中渲染和使用 3D 资产


本指南将向您展示如何使用 Shap-E 开始生成您自己的 3D 资产！


在开始之前，请确保已安装以下库：



```
# uncomment to install the necessary libraries in Colab
#!pip install diffusers transformers accelerate safetensors trimesh
```


## 文本转3D



要生成 3D 对象的 gif，请将文本提示传递给
 [ShapEPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/shap_e#diffusers.ShapEPipeline)
 。该管道生成用于创建 3D 对象的图像帧列表。



```
import torch
from diffusers import ShapEPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe = pipe.to(device)

guidance_scale = 15.0
prompt = ["A firecracker", "A birthday cupcake"]

images = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    frame_size=256,
).images
```


现在使用
 [export\_to\_gif()](/docs/diffusers/v0.23.0/en/api/utilities#diffusers.utils.export_to_gif)
 函数将图像帧列表转换为 3D 对象的 gif。



```
from diffusers.utils import export_to_gif

export_to_gif(images[0], "firecracker\_3d.gif")
export_to_gif(images[1], "cake\_3d.gif")
```


![](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/firecracker_out.gif)

 鞭炮


![](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/cake_out.gif)

 纸杯蛋糕


## 图像转 3D



要从另一个图像生成 3D 对象，请使用
 [ShapEImg2ImgPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/shap_e#diffusers.ShapEImg2ImgPipeline)
 。您可以使用现有图像或生成全新图像。让我们使用
 [康定斯基2.1](../api/pipelines/kandinsky)
 模型来生成新图像。



```
from diffusers import DiffusionPipeline
import torch

prior_pipeline = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipeline = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16, use_safetensors=True).to("cuda")

prompt = "A cheeseburger, white background"

image_embeds, negative_image_embeds = prior_pipeline(prompt, guidance_scale=1.0).to_tuple()
image = pipeline(
    prompt,
    image_embeds=image_embeds,
    negative_image_embeds=negative_image_embeds,
).images[0]

image.save("burger.png")
```


将芝士汉堡传递给
 [ShapEImg2ImgPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/shap_e#diffusers.ShapEImg2ImgPipeline)
 生成它的 3D 表示。



```
from PIL import Image
from diffusers.utils import export_to_gif

pipe = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e-img2img", torch_dtype=torch.float16, variant="fp16").to("cuda")

guidance_scale = 3.0
image = Image.open("burger.png").resize((256, 256))

images = pipe(
    image,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    frame_size=256,
).images

gif_path = export_to_gif(images[0], "burger\_3d.gif")
```


![](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/burger_in.png)

 芝士汉堡


![](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/burger_out.gif)

 3D芝士汉堡


## 生成网格



Shap-E 是一种灵活的模型，还可以生成纹理网格输出以供下游应用程序渲染。在此示例中，您将把输出转换为
 `glb`
 文件，因为🤗数据集库支持网格可视化
 `glb`
 可以通过以下方式呈现的文件
 [数据集查看器](https://huggingface.co/docs/hub/datasets-viewer#dataset-preview)
 。


您可以为两者生成网格输出
 [ShapEPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/shap_e#diffusers.ShapEPipeline)
 和
 [ShapEImg2ImgPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/shap_e#diffusers.ShapEImg2ImgPipeline)
 通过指定
 `输出类型`
 参数为
 `“网格”`
 :



```
import torch
from diffusers import ShapEPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe = pipe.to(device)

guidance_scale = 15.0
prompt = "A birthday cupcake"

images = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=64, frame_size=256, output_type="mesh").images
```


使用
 `export_to_ply()`
 函数将网格输出保存为
 ‘层’
 文件：


您可以选择将网格输出另存为
 `对象`
 文件与
 `export_to_obj()`
 功能。以多种格式保存网格输出的能力使其对于下游使用更加灵活！



```
from diffusers.utils import export_to_ply

ply_path = export_to_ply(images[0], "3d\_cake.ply")
print(f"saved to folder: {ply\_path}")
```


然后你可以转换
 ‘层’
 文件到一个
 `glb`
 带有修剪网格库的文件：



```
import trimesh

mesh = trimesh.load("3d\_cake.ply")
mesh.export("3d\_cake.glb", file_type="glb")
```


默认情况下，网格输出从底部视点聚焦，但您可以通过应用旋转变换来更改默认视点：



```
import trimesh
import numpy as np

mesh = trimesh.load("3d\_cake.ply")
rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
mesh = mesh.apply_transform(rot)
mesh.export("3d\_cake.glb", file_type="glb")
```


将网格文件上传到您的数据集存储库，以便使用数据集查看器将其可视化！


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/3D-cake.gif)