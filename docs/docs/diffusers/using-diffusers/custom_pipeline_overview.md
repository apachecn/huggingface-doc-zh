# 加载社区管道和组件

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/custom_pipeline_overview>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


## 社区管道



社区管道是任何
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 与论文中指定的原始实现不同的类（例如，
 [StableDiffusionControlNetPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetPipeline)
 对应于
 [使用 ControlNet 调节生成文本到图像](https://arxiv.org/abs/2302.05543)
 纸）。它们提供附加功能或扩展管道的原始实现。


有许多很酷的社区管道，例如
 [语音转图像](https://github.com/huggingface/diffusers/tree/main/examples/community#speech-to-image)
 或者
 [可组合稳定扩散](https://github.com/huggingface/diffusers/tree/main/examples/community#composable-stable-diffusion)
 ，你可以找到所有官方社区管道
 [此处](https://github.com/huggingface/diffusers/tree/main/examples/community)
 。


要加载 Hub 上的任何社区管道，请将社区管道的存储库 ID 传递给
 `自定义管道`
 参数以及您要从中加载管道权重和组件的模型存储库。例如，下面的示例从加载虚拟管道
 [`hf-internal-testing/diffusers-dummy-pipeline`](https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py)
 以及管道重量和组件
 [`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32)
 :


🔒 通过从 Hugging Face Hub 加载社区管道，您就相信您正在加载的代码是安全的。确保在自动加载和运行代码之前在线检查代码！



```
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "google/ddpm-cifar10-32", custom_pipeline="hf-internal-testing/diffusers-dummy-pipeline", use_safetensors=True
)
```


加载官方社区管道类似，但您可以混合从官方存储库 ID 加载权重并直接传递管道组件。下面的示例加载社区
 [CLIP引导稳定扩散](https://github.com/huggingface/diffusers/tree/main/examples/community#clip-guided-stable-diffusion)
 管道，您可以将 CLIP 模型组件直接传递给它：



```
from diffusers import DiffusionPipeline
from transformers import CLIPImageProcessor, CLIPModel

clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id)

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    custom_pipeline="clip\_guided\_stable\_diffusion",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    use_safetensors=True,
)
```


有关社区管道的更多信息，请查看
 [社区管道](custom_pipeline_examples)
 有关如何使用它们的指南，如果您有兴趣添加社区管道，请查看
 [如何贡献社区管道](contribute_pipeline)
 指导！


## 社区组件



如果您的管道具有 Diffusers 尚不支持的自定义组件，则需要附带实现它们的 Python 模块。这些定制组件可以是 VAE、UNet、调度程序等。对于文本编码器，我们依赖
 `变形金刚`
 反正。因此，应该单独处理（更多信息请参见此处）。管道代码本身也可以定制。


社区组件允许用户构建可能具有不属于 Diffuser 的自定义组件的管道。本节展示用户应如何使用社区组件来构建社区管道。


您将使用
 [showlab/show-1-base](https://huggingface.co/showlab/show-1-base)
 这里以管道检查点为例。在这里，您有一个自定义的 UNet 和一个自定义的管道（
 `文本到视频IF管道`
 ）。为了方便起见，我们称之为UNet
 `ShowOneUNet3DConditionModel`
 。


“showlab/show-1-base”已经提供了 Diffusers 格式的检查点，这是一个很好的起点。那么，让我们开始加载已经得到良好支持的组件：


1. **文本编码器**



```
from transformers import T5Tokenizer, T5EncoderModel

pipe_id = "showlab/show-1-base"
tokenizer = T5Tokenizer.from_pretrained(pipe_id, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(pipe_id, subfolder="text\_encoder")
```


2. **调度程序**



```
from diffusers import DPMSolverMultistepScheduler

scheduler = DPMSolverMultistepScheduler.from_pretrained(pipe_id, subfolder="scheduler")
```


3. **图像处理器**



```
from transformers import CLIPFeatureExtractor

feature_extractor = CLIPFeatureExtractor.from_pretrained(pipe_id, subfolder="feature\_extractor")
```


现在，您需要实现自定义 UNet。实施可用
 [此处](https://github.com/showlab/Show-1/blob/main/showone/models/unet_3d_condition.py)
 。那么，让我们创建一个名为的 Python 脚本
 `showone_unet_3d_condition.py`
 并复制实现，改变
 `UNet3D条件模型`
 类名到
 `ShowOneUNet3DConditionModel`
 以避免与扩散器发生任何冲突。这是因为 Diffuser 已经有一个
 `UNet3D条件模型`
 。我们将实现该类所需的所有组件放入
 `showone_unet_3d_condition.py`
 仅有的。你可以找到整个文件
 [此处](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py)
 。


完成后，我们可以初始化 UNet：



```
from showone_unet_3d_condition import ShowOneUNet3DConditionModel

unet = ShowOneUNet3DConditionModel.from_pretrained(pipe_id, subfolder="unet")
```


然后实现自定义
 `文本到视频IF管道`
 在另一个Python脚本中：
 `pipeline_t2v_base_pixel.py`
 。这已经可用了
 [此处](https://github.com/showlab/Show-1/blob/main/showone/pipelines/pipeline_t2v_base_pixel.py)
 。


现在您已经拥有了所有组件，请初始化
 `文本到视频IF管道`
 :



```
from pipeline_t2v_base_pixel import TextToVideoIFPipeline
import torch

pipeline = TextToVideoIFPipeline(
    unet=unet, 
    text_encoder=text_encoder, 
    tokenizer=tokenizer, 
    scheduler=scheduler, 
    feature_extractor=feature_extractor
)
pipeline = pipeline.to(device="cuda")
pipeline.torch_dtype = torch.float16
```


将管道推送到 Hub 以与社区共享：



```
pipeline.push_to_hub("custom-t2v-pipeline")
```


成功推送管道后，您需要进行一些更改：


1. 在
 `model_index.json`
 文件，更改
 `_class_name`
 属性。应该是这样的
 [所以](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/model_index.json#L2)
 。
2. 上传
 `showone_unet_3d_condition.py`
 到
 `乌内特`
 目录 （
 [示例](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py)
 ）。
3. 上传
 `pipeline_t2v_base_pixel.py`
 到管道基目录（
 [示例](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py)
 ）。


要运行推理，只需执行以下操作：



```
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "<change-username>/<change-id>", trust_remote_code=True, torch_dtype=torch.float16
).to("cuda")

prompt = "hello"

# Text embeds
prompt_embeds, negative_embeds = pipeline.encode_prompt(prompt)

# Keyframes generation (8x64x40, 2fps)
video_frames = pipeline(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    num_frames=8,
    height=40,
    width=64,
    num_inference_steps=2,
    guidance_scale=9.0,
    output_type="pt"
).frames
```


在这里，请注意使用
 `信任远程代码`
 初始化管道时的参数。它负责处理幕后的所有“魔法”。