# 提示称重

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/weighted_prompts>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/weighted_prompts>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


提示权重提供了一种强调或弱化提示的某些部分的方法，从而可以更好地控制生成的图像。提示可以包含多个概念，这些概念会转化为上下文文本嵌入。模型使用嵌入来调节其交叉注意力层以生成图像（阅读稳定扩散
 [博客文章](https://huggingface.co/blog/stable_diffusion)
 以了解有关其工作原理的更多信息）。


提示权重的工作原理是增加或减少与提示中的概念相对应的文本嵌入向量的比例，因为您可能不一定希望模型平等地关注所有概念。准备提示加权嵌入的最简单方法是使用
 [强制](https://github.com/damian0815/compel)
 ，一个文本提示加权和混合库。一旦获得提示加权嵌入，您就可以将它们传递到任何具有
 [`prompt_embeds`](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__.prompt_embeds)
 （并且可选地
 [`负提示_嵌入`](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__.负_提示_嵌入)
 ) 参数，例如
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 ,
 [StableDiffusionControlNetPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetPipeline)
 ， 和
 [StableDiffusionXLPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline)
 。


如果您最喜欢的管道没有
 `提示嵌入`
 参数，请开一个
 [问题](https://github.com/huggingface/diffusers/issues/new/choose)
 所以我们可以添加它！


本指南将向您展示如何在 🤗 Diffusers 中使用 Compel 加权和混合您的提示。


在开始之前，请确保您安装了最新版本的 Compel：



```
# uncomment to install in Colab
#!pip install compel --upgrade
```


对于本指南，让我们根据提示生成图像
 `“一只红猫在玩球”`
 使用
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 :



```
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_safetensors=True)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

prompt = "a red cat playing with a ball"

generator = torch.Generator(device="cpu").manual_seed(33)

image = pipe(prompt, generator=generator, num_inference_steps=20).images[0]
image
```


![](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/forest_0.png)


## 加权



您会注意到图像中没有“球”！让我们使用 compel 来加重提示中“球”的概念。创建一个
 [`强制`](https://github.com/damian0815/compel/blob/main/doc/compel.md#compel-objects)
 对象，并向其传递一个分词器和文本编码器：



```
from compel import Compel

compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
```


强迫使用
 `+`
 或者
 ``-``
 增加或减少提示中单词的权重。增加“球”的重量：


`+`
 对应的值
 `1.1`
 ,
 `++`
 对应于
 `1.1^2`
 ， 等等。相似地，
 ``-``
 对应于
 `0.9`
 和
 `--`
 对应于
 `0.9^2`
 。请随意尝试添加更多内容
 `+`
 或者
 ``-``
 在你的提示中！



```
prompt = "a red cat playing with a ball++"
```


将提示传递给
 `强制过程`
 创建传递到管道的新提示嵌入：



```
prompt_embeds = compel_proc(prompt)
generator = torch.manual_seed(33)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```


![](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/forest_1.png)


 要减少部分提示的重量，请使用
 ``-``
 后缀：



```
prompt = "a red------- cat playing with a ball"
prompt_embeds = compel_proc(prompt)

generator = torch.manual_seed(33)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-neg.png)


 您甚至可以在同一提示中增加或减少多个概念的权重：



```
prompt = "a red cat++ playing with a ball----"
prompt_embeds = compel_proc(prompt)

generator = torch.manual_seed(33)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-pos-neg.png)


## 混合



您还可以创建一个加权
 *混合*
 通过添加提示
 `.blend()`
 到提示列表并传递一些权重。您的混合可能并不总能产生您期望的结果，因为它打破了有关文本编码器如何工作的一些假设，所以尽情享受并尝试吧！



```
prompt_embeds = compel_proc('("a red cat playing with a ball", "jungle").blend(0.7, 0.8)')
generator = torch.Generator(device="cuda").manual_seed(33)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-blend.png)


## 连词



连接词独立地扩散每个提示，并通过它们的加权和连接它们的结果。添加
 `.and()`
 到提示列表的末尾以创建连词：



```
prompt_embeds = compel_proc('["a red cat", "playing with a", "ball"].and()')
generator = torch.Generator(device="cuda").manual_seed(55)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-conj.png)


## 文本倒置



[文本倒置](../training/text_inversion)
 是一种从某些图像中学习特定概念的技术，您可以使用该技术生成基于该概念的新图像。


创建管道并使用
 [load\_textual\_inversion()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.load_textual_inversion)
 加载文本反转嵌入的函数（请随意浏览
 [稳定扩散概念化器](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer)
 对于 100 多个经过训练的概念）：



```
import torch
from diffusers import StableDiffusionPipeline
from compel import Compel, DiffusersTextualInversionManager

pipe = StableDiffusionPipeline.from_pretrained(
  "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
  use_safetensors=True, variant="fp16").to("cuda")
pipe.load_textual_inversion("sd-concepts-library/midjourney-style")
```


强制提供了一个
 `DiffusersTextualInversionManager`
 类通过文本反转来简化提示权重。实例化
 `DiffusersTextualInversionManager`
 并将其传递给
 `强迫`
 班级：



```
textual_inversion_manager = DiffusersTextualInversionManager(pipe)
compel_proc = Compel(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder,
    textual_inversion_manager=textual_inversion_manager)
```


结合使用条件提示的概念
 `<概念>`
 句法：



```
prompt_embeds = compel_proc('("A red cat++ playing with a ball <midjourney-style>")')

image = pipe(prompt_embeds=prompt_embeds).images[0]
image
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-text-inversion.png)


## 梦想展位



[DreamBooth](../培训/dreambooth)
 是一种仅在给定要训练的主题的一些图像的情况下生成主题的上下文图像的技术。它与文本反演类似，但 DreamBooth 训练完整的模型，而文本反演仅微调文本嵌入。这意味着你应该使用
 [来自\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained)
 加载 DreamBooth 模型（请随意浏览
 [稳定扩散 Dreambooth 概念库](https://huggingface.co/sd-dreambooth-library)
 对于 100 多个经过训练的模型）：



```
import torch
from diffusers import DiffusionPipeline, UniPCMultistepScheduler
from compel import Compel

pipe = DiffusionPipeline.from_pretrained("sd-dreambooth-library/dndcoverart-v1", torch_dtype=torch.float16).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
```


创建一个
 `强迫`
 带有分词器和文本编码器的类，并将提示传递给它。根据您使用的模型，您需要将模型的唯一标识符合并到提示中。例如，
 `dndcoverart-v1`
 模型使用标识符
 'dnd封面艺术'
 :



```
compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
prompt_embeds = compel_proc('("magazine cover of a dndcoverart dragon, high quality, intricate details, larry elmore art style").and()')
image = pipe(prompt_embeds=prompt_embeds).images[0]
image
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-dreambooth.png)


## 稳定扩散XL



Stable Diffusion XL (SDXL) 有两个分词器和文本编码器，因此它的用法有点不同。为了解决这个问题，您应该将分词器和编码器传递给
 `强迫`
 班级：



```
from compel import Compel, ReturnedEmbeddingsType
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid
import torch

pipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  variant="fp16",
  use_safetensors=True,
  torch_dtype=torch.float16
).to("cuda")

compel = Compel(
  tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] ,
  text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
  returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
  requires_pooled=[False, True]
)
```


这次，我们将第一个提示的“ball”权重提高 1.5 倍，将第二个提示的“ball”权重降低 0.6 倍。这
 [StableDiffusionXLPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline)
 还要求
 [`pooled_prompt_embeds`](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline.__call__.pooled_prompt_embeds)
 （并且可选地
 [`negative_pooled_prompt_embeds`](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline.__call__.negative_pooled_prompt_embeds)
 ）所以你应该将它们与调节张量一起传递到管道：



```
# apply weights
prompt = ["a red cat playing with a (ball)1.5", "a red cat playing with a (ball)0.6"]
conditioning, pooled = compel(prompt)

# generate image
generator = [torch.Generator().manual_seed(33) for _ in range(len(prompt))]
images = pipeline(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator, num_inference_steps=30).images
make_image_grid(images, rows=1, cols=2)
```


![](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/sdxl_ball1.png)

 “一只红猫在玩（球）1.5”


![](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/sdxl_ball2.png)

 “一只红猫正在玩（球）0.6”