# 文本倒置

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/textual_inversion_inference>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/textual_inversion_inference>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


这
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 支持文本反转，这种技术使稳定扩散等模型能够从几个样本图像中学习新概念。这使您可以更好地控制生成的图像，并允许您根据特定概念定制模型。您可以快速开始使用社区创建的概念集合
 [稳定扩散概念化器](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer)
 。


本指南将向您展示如何使用稳定扩散概念化器中预先学习的概念通过文本反演进行推理。如果您有兴趣通过文本倒置教授新概念模型，请查看
 [文本倒置](../training/text_inversion)
 培训指南。


导入必要的库：



```
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import make_image_grid
```


## 稳定扩散 1 和 2



选择一个稳定扩散检查点和一个预先学习的概念
 [稳定扩散概念化器](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer)
 :



```
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
repo_id_embeds = "sd-concepts-library/cat-toy"
```


现在您可以加载管道，并将预先学习的概念传递给它：



```
pipeline = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path, torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

pipeline.load_textual_inversion(repo_id_embeds)
```


使用特殊占位符标记创建包含预先学习的概念的提示
 `<猫玩具>`
 ，然后选择您想要生成的样本数和图像行数：



```
prompt = "a grafitti in a favela wall with a <cat-toy> on it"

num_samples_per_row = 2
num_rows = 2
```


然后运行管道（随意调整参数，例如
 `num_inference_steps`
 和
 `指导规模`
 查看它们如何影响图像质量），保存生成的图像并使用您在开始时创建的辅助函数将它们可视化：



```
all_images = []
for _ in range(num_rows):
    images = pipeline(prompt, num_images_per_prompt=num_samples_per_row, num_inference_steps=50, guidance_scale=7.5).images
    all_images.extend(images)

grid = make_image_grid(all_images, num_rows, num_samples_per_row)
grid
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/textual_inversion_inference.png)


## 稳定扩散XL



Stable Diffusion XL (SDXL) 还可以使用文本反转向量进行推理。与稳定扩散 1 和 2 相比，SDXL 有两个文本编码器，因此您需要两个文本反转嵌入 - 每个文本编码器模型一个。


让我们下载 SDXL 文本反转嵌入并仔细看看它的结构：



```
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

file = hf_hub_download("dn118/unaestheticXL", filename="unaestheticXLv31.safetensors")
state_dict = load_file(file)
state_dict
```



```
{'clip\_g': tensor([[ 0.0077, -0.0112, 0.0065, ..., 0.0195, 0.0159, 0.0275],
 ...,
 [-0.0170, 0.0213, 0.0143, ..., -0.0302, -0.0240, -0.0362]],
 'clip\_l': tensor([[ 0.0023, 0.0192, 0.0213, ..., -0.0385, 0.0048, -0.0011],
 ...,
 [ 0.0475, -0.0508, -0.0145, ..., 0.0070, -0.0089, -0.0163]],
```


有两个张量，
 `“剪辑_g”`
 和
 `“剪辑_l”`
 。
 `“剪辑_g”`
 对应于 SDXL 中较大的文本编码器，并指的是
 `pipe.text_encoder_2`
 和
 `“剪辑_l”`
 指的是
 `pipe.text_encoder`
 。


现在，您可以通过将每个张量与正确的文本编码器和分词器一起传递来单独加载它们
到
 [load\_textual\_inversion()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.load_textual_inversion)
 :



```
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

pipe.load_textual_inversion(state_dict["clip\_g"], token="unaestheticXLv31", text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
pipe.load_textual_inversion(state_dict["clip\_l"], token="unaestheticXLv31", text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)

# the embedding should be used as a negative embedding, so we pass it as a negative prompt
generator = torch.Generator().manual_seed(33)
image = pipe("a woman standing in front of a mountain", negative_prompt="unaestheticXLv31", generator=generator).images[0]
image
```