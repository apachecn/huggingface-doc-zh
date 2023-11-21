# 使模型适应新任务

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/adapt_a_model>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/adapt_a_model>


许多扩散系统共享相同的组件，允许您将一项任务的预训练模型调整为完全不同的任务。


本指南将向您展示如何通过初始化和修改预训练的体系结构来调整预训练的文本到图像模型以进行修复
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)
 。


## 配置 UNet2DConditionModel 参数



A
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)
 默认情况下接受 4 个通道
 [输入示例](https://huggingface.co/docs/diffusers/v0.16.0/en/api/models#diffusers.UNet2DConditionModel.in_channels)
 。例如，加载预训练的文本到图像模型，例如
 [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 看看有多少
 `in_channels`
 :



```
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
pipeline.unet.config["in\_channels"]
4
```


修复需要输入样本中有 9 个通道。您可以在预训练的修复模型中检查该值，例如
 [`runwayml/stable-diffusion-inpainting`](https://huggingface.co/runwayml/stable-diffusion-inpainting)
 :



```
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", use_safetensors=True)
pipeline.unet.config["in\_channels"]
9
```


要调整文本到图像模型以进行修复，您需要更改
 `in_channels`
 从 4 到 9。


初始化一个
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)
 使用预训练的文本到图像模型权重，并更改
 `in_channels`
 至 9. 更改数量
 `in_channels`
 意味着你需要设置
 `ignore_mismatched_sizes=True`
 和
 `low_cpu_mem_usage=False`
 以避免尺寸不匹配错误，因为现在形状不同。



```
from diffusers import UNet2DConditionModel

model_id = "runwayml/stable-diffusion-v1-5"
unet = UNet2DConditionModel.from_pretrained(
    model_id,
    subfolder="unet",
    in_channels=9,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
    use_safetensors=True,
)
```


文本到图像模型中其他组件的预训练权重是从其检查点初始化的，但输入通道权重 (
 `conv_in.weight`
 ） 的
 `乌内特`
 是随机初始化的。微调模型以进行修复非常重要，否则模型会返回噪声。