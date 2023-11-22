# 社区管道

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/custom_pipeline_examples>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_examples>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


 有关社区管道背后的设计选择的更多背景信息，请查看
 [本期](https://github.com/huggingface/diffusers/issues/841)
 。


社区管道使您能够发挥创造力并构建自己独特的管道以与社区共享。您可以在以下位置找到所有社区管道
 [diffusers/examples/community](https://github.com/huggingface/diffusers/tree/main/examples/community)
 文件夹以及有关如何使用它们的推理和培训示例。本指南展示了一些社区管道，希望它能激励您创建自己的管道（请随意使用您自己的管道打开 PR，我们将合并它！）。


要加载社区管道，请使用
 `自定义管道`
 论证中
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 指定其中一个文件
 [diffusers/examples/community](https://github.com/huggingface/diffusers/tree/main/examples/community)
 :



```
pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", custom_pipeline="filename\_in\_the\_community\_folder", use_safetensors=True
)
```


如果社区管道未按预期工作，请打开 GitHub 问题并提及作者。


您可以在“如何”中了解有关社区管道的更多信息
 [加载社区管道](custom_pipeline_overview)
 以及如何
 [贡献社区管道](contribute_pipeline)
 指南。


## 多语言稳定传播



多语言稳定扩散管道使用预训练的
 [XLM-RoBERTa](https://huggingface.co/paplca/xlm-roberta-base-language-detection)
 识别一种语言和
 [mBART-large-50](https://huggingface.co/facebook/mbart-large-50-many-to-one-mmt)
 模型来处理翻译。这允许您从 20 种语言的文本生成图像。



```
from PIL import Image
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid
from transformers import (
    pipeline,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
device_dict = {"cuda": 0, "cpu": -1}

# add language detection pipeline
language_detection_model_ckpt = "papluca/xlm-roberta-base-language-detection"
language_detection_pipeline = pipeline("text-classification",
                                       model=language_detection_model_ckpt,
                                       device=device_dict[device])

# add model for language translation
trans_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
trans_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt").to(device)

diffuser_pipeline = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    custom_pipeline="multilingual\_stable\_diffusion",
    detection_pipeline=language_detection_pipeline,
    translation_model=trans_model,
    translation_tokenizer=trans_tokenizer,
    torch_dtype=torch.float16,
)

diffuser_pipeline.enable_attention_slicing()
diffuser_pipeline = diffuser_pipeline.to(device)

prompt = ["a photograph of an astronaut riding a horse", 
          "Una casa en la playa",
          "Ein Hund, der Orange isst",
          "Un restaurant parisien"]

images = diffuser_pipeline(prompt).images
grid = make_image_grid(images, rows=2, cols=2)
grid
```


![](https://user-images.githubusercontent.com/4313860/198328706-295824a4-9856-4ce5-8e66-278ceb42fd29.png)


## 魔法混合



[MagicMix](https://huggingface.co/papers/2210.16056)
 是一个管道，可以混合图像和文本提示来生成保留图像结构的新图像。这
 `混合因子`
 确定提示对布局生成的影响有多大，
 `公里`
 控制内容生成过程中的步骤数，以及
 `最大公里`
 确定原始图像布局中保留多少信息。



```
from diffusers import DiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image

pipeline = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    custom_pipeline="magic\_mix",
    scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler"),
).to('cuda')

img = load_image("https://user-images.githubusercontent.com/59410571/209578593-141467c7-d831-4792-8b9a-b17dc5e47816.jpg")
mix_img = pipeline(img, prompt="bed", kmin = 0.3, kmax = 0.5, mix_factor = 0.5)
mix_img
```


![](https://user-images.githubusercontent.com/59410571/209578593-141467c7-d831-4792-8b9a-b17dc5e47816.jpg)

 图片提示


![](https://user-images.githubusercontent.com/59410571/209578602-70f323fa-05b7-4dd6-b055-e40683e37914.jpg)

 图文提示混合