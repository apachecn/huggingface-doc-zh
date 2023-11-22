# 将文件推送到 Hub

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/push_to_hub>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/push_to_hub>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


🤗 扩散器提供了
 [PushToHubMixin](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin)
 用于将模型、调度程序或管道上传到集线器。这是在 Hub 上存储文件的简单方法，还允许您与其他人共享您的工作。在引擎盖下，
 [PushToHubMixin](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin)
 :


1.在Hub上创建一个存储库
2. 保存模型、调度程序或管道文件，以便稍后重新加载
3. 将包含这些文件的文件夹上传到集线器


本指南将向您展示如何使用
 [PushToHubMixin](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin)
 将您的文件上传到集线器。


您需要使用您的访问权限登录您的 Hub 帐户
 [令牌](https://huggingface.co/settings/tokens)
 第一的：



```
from huggingface_hub import notebook_login

notebook_login()
```


## 楷模



要将模型推送到 Hub，请调用
 [push\_to\_hub()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub)
 并指定要存储在 Hub 上的模型的存储库 ID：



```
from diffusers import ControlNetModel

controlnet = ControlNetModel(
    block_out_channels=(32, 64),
    layers_per_block=2,
    in_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    cross_attention_dim=32,
    conditioning_embedding_out_channels=(16, 32),
)
controlnet.push_to_hub("my-controlnet-model")
```


对于模型，您还可以指定
 [*变体*]（加载#checkpoint-variants）
 推到轮毂的重量。例如，要推
 `fp16`
 重量：



```
controlnet.push_to_hub("my-controlnet-model", variant="fp16")
```


这
 [push\_to\_hub()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub)
 函数保存模型的
 `config.json`
 文件和权重自动保存在
 `安全张量`
 格式。


现在您可以从 Hub 上的存储库重新加载模型：



```
model = ControlNetModel.from_pretrained("your-namespace/my-controlnet-model")
```


## 调度程序



要将调度程序推送到集线器，请调用
 [push\_to\_hub()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub)
 并指定要存储在集线器上的调度程序的存储库 ID：



```
from diffusers import DDIMScheduler

scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled\_linear",
    clip_sample=False,
    set_alpha_to_one=False,
)
scheduler.push_to_hub("my-controlnet-scheduler")
```


这
 [push\_to\_hub()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub)
 函数保存调度程序的
 `scheduler_config.json`
 文件到指定的存储库。


现在您可以从 Hub 上的存储库重新加载调度程序：



```
scheduler = DDIMScheduler.from_pretrained("your-namepsace/my-controlnet-scheduler")
```


## 管道



您还可以将整个管道及其所有组件推送到集线器。例如，初始化a的组件
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 与你想要的参数：



```
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
)
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTokenizer

unet = UNet2DConditionModel(
    block_out_channels=(32, 64),
    layers_per_block=2,
    sample_size=32,
    in_channels=4,
    out_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    cross_attention_dim=32,
)

scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled\_linear",
    clip_sample=False,
    set_alpha_to_one=False,
)

vae = AutoencoderKL(
    block_out_channels=[32, 64],
    in_channels=3,
    out_channels=3,
    down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
    up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
    latent_channels=4,
)

text_encoder_config = CLIPTextConfig(
    bos_token_id=0,
    eos_token_id=2,
    hidden_size=32,
    intermediate_size=37,
    layer_norm_eps=1e-05,
    num_attention_heads=4,
    num_hidden_layers=5,
    pad_token_id=1,
    vocab_size=1000,
)
text_encoder = CLIPTextModel(text_encoder_config)
tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
```


将所有组件传递给
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 并打电话
 [push\_to\_hub()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub)
 将管道推送到 Hub：



```
components = {
    "unet": unet,
    "scheduler": scheduler,
    "vae": vae,
    "text\_encoder": text_encoder,
    "tokenizer": tokenizer,
    "safety\_checker": None,
    "feature\_extractor": None,
}

pipeline = StableDiffusionPipeline(**components)
pipeline.push_to_hub("my-pipeline")
```


这
 [push\_to\_hub()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub)
 函数将每个组件保存到存储库中的子文件夹中。现在您可以从 Hub 上的存储库重新加载管道：



```
pipeline = StableDiffusionPipeline.from_pretrained("your-namespace/my-pipeline")
```


## 隐私



放
 `私人=真`
 在里面
 [push\_to\_hub()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub)
 保持模型、调度程序或管道文件私有的函数：



```
controlnet.push_to_hub("my-controlnet-model-private", private=True)
```


私有存储库仅对您可见，其他用户将无法克隆该存储库，并且您的存储库不会出现在搜索结果中。即使用户拥有您的私有存储库的 URL，他们也会收到
 “404 - 抱歉，我们找不到您要查找的页面。”


要从私有或门控存储库加载模型、调度程序或管道，请设置
 `use_auth_token=True`
 :



```
model = ControlNetModel.from_pretrained("your-namespace/my-controlnet-model-private", use_auth_token=True)
```