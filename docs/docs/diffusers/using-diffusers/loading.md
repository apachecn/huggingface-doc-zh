# 加载管道、模型和调度程序

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/loading>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/loading>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


拥有一种简单的方法来使用扩散系统进行推理对于🧨 Diffusers 来说至关重要。扩散系统通常由多个组件组成，例如参数化模型、分词器和调度器，它们以复杂的方式交互。这就是为什么我们设计了
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 将整个扩散系统的复杂性包装到一个易于使用的 API 中，同时保持足够的灵活性以适应其他用例，例如单独加载每个组件作为构建块来组装您自己的扩散系统。


推理或训练所需的一切都可以通过
 `from_pretrained()`
 方法。


本指南将向您展示如何加载：


*来自Hub和本地的管道
* 将不同组件放入管道中
* 检查点变体，例如不同的浮点类型或非指数平均 (EMA) 权重
* 模型和调度程序


## 扩散管道



💡 跳至
 [DiffusionPipeline 解释](#diffusionpipeline-explained)
 如果您有兴趣更详细地了解如何
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 类作品。


这
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 类是从以下位置加载最新趋势扩散模型的最简单、最通用的方法
 [中心](https://huggingface.co/models?library=diffusers&sort=trending)
 。这
 [DiffusionPipeline.from\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained)
 方法会自动从检查点检测正确的管道类，下载并缓存所有所需的配置和权重文件，并返回准备进行推理的管道实例。



```
from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
```


您还可以加载具有特定管道类的检查点。上面的例子加载了一个稳定扩散模型；要获得相同的结果，请使用
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 班级：



```
from diffusers import StableDiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
```


检查点（例如
 [`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4)
 或者
 [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 ）也可以用于多个任务，例如文本到图像或图像到图像。为了区分您想要使用检查点的任务，您必须直接使用其相应的特定于任务的管道类来加载它：



```
from diffusers import StableDiffusionImg2ImgPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_id)
```


### 


 本地管道


要在本地加载扩散管道，请使用
 [`git-lfs`](https://git-lfs.github.com/)
 手动下载检查点（在本例中，
 [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 ) 到您的本地磁盘。这将创建一个本地文件夹，
 `./稳定扩散-v1-5`
 ，在您的磁盘上：



```
git-lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```


然后将本地路径传递给
 [来自\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained)
 :



```
from diffusers import DiffusionPipeline

repo_id = "./stable-diffusion-v1-5"
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
```


这
 [来自\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained)
 当检测到本地路径时，方法不会从集线器下载任何文件，但这也意味着它不会下载并缓存对检查点的最新更改。


### 


 交换管道中的组件


您可以使用其他兼容组件自定义任何管道的默认组件。定制很重要，因为：


* 更改调度程序对于探索生成速度和质量之间的权衡非常重要。
* 模型的不同组件通常是独立训练的，您可以用性能更好的组件替换组件。
* 在微调期间，通常只训练一些组件（例如 UNet 或文本编码器）。


要找出哪些调度程序兼容自定义，您可以使用
 `兼容机`
 方法：



```
from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
stable_diffusion.scheduler.compatibles
```


让我们使用
 [SchedulerMixin.from\_pretrained()](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin.from_pretrained)
 替换默认的方法
 [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler)
 使用性能更高的调度程序，
 [EulerDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/euler#diffusers.EulerDiscreteScheduler)
 。这
 `子文件夹=“调度程序”`
 需要参数才能从正确的位置加载调度程序配置
 [子文件夹](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/scheduler)
 管道存储库的。


然后就可以通过新的了
 [EulerDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/euler#diffusers.EulerDiscreteScheduler)
 实例到
 `调度程序`
 论证中
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 :



```
from diffusers import DiffusionPipeline, EulerDiscreteScheduler

repo_id = "runwayml/stable-diffusion-v1-5"
scheduler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, scheduler=scheduler, use_safetensors=True)
```


### 


 安全检查员


像稳定扩散这样的扩散模型会生成有害内容，这就是为什么 🧨 扩散器具有
 [安全检查器](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py)
 根据已知的硬编码 NSFW 内容检查生成的输出。如果您出于某种原因想禁用安全检查器，请通过
 `无`
 到
 `安全检查器`
 争论：



```
from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, safety_checker=None, use_safetensors=True)
"""
You have disabled the safety checker for <class 'diffusers.pipelines.stable\_diffusion.pipeline\_stable\_diffusion.StableDiffusionPipeline'> by passing `safety\_checker=None`. Ensure that you abide by the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend keeping the safety filter enabled in all public-facing circumstances, disabling it only for use cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .
"""
```


### 


 跨管道重用组件


您还可以在多个管道中重复使用相同的组件，以避免将权重加载到 RAM 中两次。使用
 [组件](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.components)
 保存组件的方法：



```
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

model_id = "runwayml/stable-diffusion-v1-5"
stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)

components = stable_diffusion_txt2img.components
```


然后你就可以通过
 `组件`
 到另一个管道而不将权重重新加载到 RAM 中：



```
stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(**components)
```


如果您希望更灵活地重用或禁用哪些组件，您还可以将组件单独传递到管道。例如，要在图像到图像管道中重用文本到图像管道中的相同组件（安全检查器和特征提取器除外）：



```
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

model_id = "runwayml/stable-diffusion-v1-5"
stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(
    vae=stable_diffusion_txt2img.vae,
    text_encoder=stable_diffusion_txt2img.text_encoder,
    tokenizer=stable_diffusion_txt2img.tokenizer,
    unet=stable_diffusion_txt2img.unet,
    scheduler=stable_diffusion_txt2img.scheduler,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
)
```


## 检查点变体



检查点变体通常是权重为：


* 以不同的浮点类型存储，以实现更低的精度和更低的存储，例如
 [`torch.float16`](https://pytorch.org/docs/stable/tensors.html#data-types)
 ，因为下载只需要一半的带宽和存储空间。如果您正在继续训练或使用 CPU，则无法使用此变体。
* 非指数平均 (EMA) 权重，不应用于推理。您应该使用这些来继续微调模型。


💡 当检查点具有相同的模型结构，但它们在不同的数据集和不同的训练设置上进行训练时，它们应该存储在单独的存储库中而不是变体中（例如，
 `稳定扩散-v1-4`
 和
 `稳定扩散-v1-5`
 ）。


否则，一个变体是
 **完全相同的**
 到原来的检查站。它们具有完全相同的序列化格式（例如
 [安全张量](./using_safetensors)
 ）、模型结构和具有相同张量形状的权重。


| **checkpoint type**  | **weight name**  | **argument for loading weights**  |
| --- | --- | --- |
| 	 original	  | 	 diffusion\_pytorch\_model.bin	  |  |
| 	 floating point	  | 	 diffusion\_pytorch\_model.fp16.bin	  | `variant` 	 ,	 `torch_dtype`  |
| 	 non-EMA	  | 	 diffusion\_pytorch\_model.non\_ema.bin	  | `variant`  |


对于加载变体，有两个重要的参数需要了解：


* `torch_dtype`
 定义加载检查点的浮点精度。例如，如果您想通过加载
 `fp16`
 变体，你应该指定
 `torch_dtype=torch.float16`
 到
 *转换权重*
 到
 `fp16`
 。否则，
 `fp16`
 权重转换为默认值
 `fp32`
 精确。您还可以加载原始检查点而不定义
 `变体`
 参数，并将其转换为
 `fp16`
 和
 `torch_dtype=torch.float16`
 。在这种情况下，默认
 `fp32`
 首先下载权重，然后将其转换为
 `fp16`
 加载后。
* `变体`
 定义应从存储库加载哪些文件。例如，如果你想加载一个
 `非_ema`
 的变体
 [`diffusers/stable-diffusion-variants`](https://huggingface.co/diffusers/stable-diffusion-variants/tree/main/unet)
 存储库，您应该指定
 `变体=“non_ema”`
 下载
 `非_ema`
 文件。



```
from diffusers import DiffusionPipeline
import torch

# load fp16 variant
stable_diffusion = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16, use_safetensors=True
)
# load non\_ema variant
stable_diffusion = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", variant="non\_ema", use_safetensors=True
)
```


要保存以不同浮点类型存储的检查点或作为非 EMA 变体，请使用
 [DiffusionPipeline.save\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.save_pretrained)
 方法并指定
 `变体`
 争论。您应该尝试将变体保存到与原始检查点相同的文件夹中，以便可以从同一文件夹加载两者：



```
from diffusers import DiffusionPipeline

# save as fp16 variant
stable_diffusion.save_pretrained("runwayml/stable-diffusion-v1-5", variant="fp16")
# save as non-ema variant
stable_diffusion.save_pretrained("runwayml/stable-diffusion-v1-5", variant="non\_ema")
```


如果您不将变体保存到现有文件夹，则必须指定
 `变体`
 参数，否则会抛出一个
 `例外`
 因为它找不到原始检查点：



```
# 👎 this won't work
stable_diffusion = DiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
# 👍 this works
stable_diffusion = DiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16, use_safetensors=True
)
```


## 楷模



模型加载自
 [ModelMixin.from\_pretrained()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin.from_pretrained)
 方法，下载并缓存最新版本的模型权重和配置。如果本地缓存中有最新的文件，
 [来自_pretrained()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin.from_pretrained)
 重用缓存中的文件而不是重新下载它们。


可以使用以下命令从子文件夹加载模型
 `子文件夹`
 争论。例如，模型权重为
 `runwayml/稳定扩散-v1-5`
 都存储在
 [`unet`](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/unet)
 子文件夹：



```
from diffusers import UNet2DConditionModel

repo_id = "runwayml/stable-diffusion-v1-5"
model = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", use_safetensors=True)
```


或者直接从存储库
 [目录](https://huggingface.co/google/ddpm-cifar10-32/tree/main)
 ：



```
from diffusers import UNet2DModel

repo_id = "google/ddpm-cifar10-32"
model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)
```


您还可以通过指定来加载和保存模型变体
 `变体`
 论证中
 [ModelMixin.from\_pretrained()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin.from_pretrained)
 和
 [ModelMixin.save\_pretrained()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin.save_pretrained)
 :



```
from diffusers import UNet2DConditionModel

model = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet", variant="non\_ema", use_safetensors=True
)
model.save_pretrained("./local-unet", variant="non\_ema")
```


## 调度程序



调度程序是从
 [SchedulerMixin.from\_pretrained()](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin.from_pretrained)
 方法，与模型不同，调度程序是
 **未参数化**
 或者
 **受过培训**
 ;它们由配置文件定义。


加载调度程序不会消耗任何大量的内存，并且相同的配置文件可以用于各种不同的调度程序。
例如，以下调度程序兼容
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 ，这意味着您可以在以下任何类中加载相同的调度程序配置文件：



```
from diffusers import StableDiffusionPipeline
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

repo_id = "runwayml/stable-diffusion-v1-5"

ddpm = DDPMScheduler.from_pretrained(repo_id, subfolder="scheduler")
ddim = DDIMScheduler.from_pretrained(repo_id, subfolder="scheduler")
pndm = PNDMScheduler.from_pretrained(repo_id, subfolder="scheduler")
lms = LMSDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
euler_anc = EulerAncestralDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
euler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
dpm = DPMSolverMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")

# replace `dpm` with any of `ddpm`, `ddim`, `pndm`, `lms`, `euler\_anc`, `euler`
pipeline = StableDiffusionPipeline.from_pretrained(repo_id, scheduler=dpm, use_safetensors=True)
```


## 扩散管道解释



作为一个类方法，
 [DiffusionPipeline.from\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained)
 负责两件事：


* 下载推理所需的最新版本的文件夹结构并缓存。如果本地缓存中有最新的文件夹结构，
 [DiffusionPipeline.from\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained)
 重用缓存并且不会重新下载文件。
* 将缓存的权重加载到正确的管道中
 [类](../api/pipelines/overview#diffusers-summary)
 - 检索自
 `model_index.json`
 文件 - 并返回它的一个实例。


管道的底层文件夹结构与其类实例直接对应。例如，
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 对应于文件夹结构
 [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 。



```
from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
print(pipeline)
```


你会看到 pipeline 是一个实例
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 ，由七个部分组成：


* `“特征提取器”`
 ： A
 [CLIPImageProcessor](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPImageProcessor)
 来自🤗变形金刚。
* `“安全检查器”`
 ： A
 [组件](https://github.com/huggingface/diffusers/blob/e55687e1e15407f60f32242027b7bb8170e58266/src/diffusers/pipelines/stable_diffusion/safety_checker.py#L32)
 用于筛查有害内容。
* `“调度程序”`
 : 的一个实例
 [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler)
 。
* `“文本编码器”`
 ： A
 [CLIPTextModel](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPTextModel)
 来自🤗变形金刚。
* `“分词器”`
 ： A
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPTokenizer)
 来自🤗变形金刚。
* `“乌内特”`
 : 的一个实例
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)
 。
* `“vae”`
 : 的一个实例
 [AutoencoderKL](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)
 。



```
StableDiffusionPipeline {
  "feature\_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "safety\_checker": [
    "stable\_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text\_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```


将管道实例的组件与
 [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main)
 文件夹结构，您将看到存储库中的每个组件都有一个单独的文件夹：



```
.
├── feature_extractor
│   └── preprocessor_config.json
├── model_index.json
├── safety_checker
│   ├── config.json
| ├── model.fp16.safetensors
│ ├── model.safetensors
│ ├── pytorch\_model.bin
| └── pytorch\_model.fp16.bin
├── scheduler
│   └── scheduler\_config.json
├── text\_encoder
│   ├── config.json
| ├── model.fp16.safetensors
│ ├── model.safetensors
│ |── pytorch\_model.bin
| └── pytorch\_model.fp16.bin
├── tokenizer
│   ├── merges.txt
│   ├── special\_tokens\_map.json
│   ├── tokenizer\_config.json
│   └── vocab.json
├── unet
│   ├── config.json
│   ├── diffusion\_pytorch\_model.bin
| |── diffusion\_pytorch\_model.fp16.bin
│ |── diffusion\_pytorch\_model.f16.safetensors
│ |── diffusion\_pytorch\_model.non\_ema.bin
│ |── diffusion\_pytorch\_model.non\_ema.safetensors
│ └── diffusion\_pytorch\_model.safetensors
|── vae
. ├── config.json
. ├── diffusion\_pytorch\_model.bin
 ├── diffusion\_pytorch\_model.fp16.bin
 ├── diffusion\_pytorch\_model.fp16.safetensors
 └── diffusion\_pytorch\_model.safetensors
```


您可以将管道的每个组件作为属性访问以查看其配置：



```
pipeline.tokenizer
CLIPTokenizer(
    name_or_path="/root/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819/tokenizer",
    vocab_size=49408,
    model_max_length=77,
    is_fast=False,
    padding_side="right",
    truncation_side="right",
    special_tokens={
        "bos\_token": AddedToken("<|startoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "eos\_token": AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "unk\_token": AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "pad\_token": "<|endoftext|>",
    },
    clean_up_tokenization_spaces=True
)
```


每条管道都期望
 [`model_index.json`](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/model_index.json)
 文件告诉
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 :


* 从哪个管道类加载
 `_class_name`
* 使用哪个版本的 🧨 Diffusers 创建模型
 `_diffuser_version`
* 子文件夹中存储了哪个库中的哪些组件（
 `名字`
 对应于组件和子文件夹名称，
 `图书馆`
 对应于从中加载类的库的名称，并且
 `类`
 对应类名）



```
{
  "\_class\_name": "StableDiffusionPipeline",
  "\_diffusers\_version": "0.6.0",
  "feature\_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "safety\_checker": [
    "stable\_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text\_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```