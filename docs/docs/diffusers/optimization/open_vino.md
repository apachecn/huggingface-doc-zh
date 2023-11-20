# 开放VINO

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/optimization/open_vino>
>
> 原始地址：<https://huggingface.co/docs/diffusers/optimization/open_vino>


🤗
 [最佳](https://github.com/huggingface/optimum-intel)
 提供与 OpenVINO 兼容的稳定扩散管道，以在各种 Intel 处理器上执行推理（请参阅
 [完整列表]((https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html))
 支持的设备数）。


您需要安装 🤗 Optimum Intel 以及
 `--升级策略渴望`
 选项以确保
 [`optimum-intel`](https://github.com/huggingface/optimum-intel)
 正在使用最新版本：



```
pip install --upgrade-strategy eager optimum["openvino"]
```


本指南将向您展示如何将 Stable Diffusion 和 Stable Diffusion XL (SDXL) 管道与 OpenVINO 结合使用。


## 稳定扩散



要加载并运行推理，请使用
 `OV稳定扩散管道`
 。如果您想加载 PyTorch 模型并将其即时转换为 OpenVINO 格式，请设置
 `导出=真`
 ：



```
from optimum.intel import OVStableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)
prompt = "sailing ship in storm by Rembrandt"
image = pipeline(prompt).images[0]

# Don't forget to save the exported model
pipeline.save_pretrained("openvino-sd-v1-5")
```


为了进一步加速推理，静态地重塑模型。如果更改任何参数（例如输出高度或宽度），则需要再次静态地重塑模型。



```
# Define the shapes related to the inputs and desired outputs
batch_size, num_images, height, width = 1, 1, 512, 512

# Statically reshape the model
pipeline.reshape(batch_size, height, width, num_images)
# Compile the model before inference
pipeline.compile()

image = pipeline(
    prompt,
    height=height,
    width=width,
    num_images_per_prompt=num_images,
).images[0]
```


![](https://huggingface.co/datasets/optimum/documentation-images/resolve/main/intel/openvino/stable_diffusion_v1_5_sail_boat_rembrandt.png)


 您可以在 🤗 最优中找到更多示例
 [文档](https://huggingface.co/docs/optimum/intel/inference#stable-diffusion)
 ，并且稳定扩散支持文本到图像、图像到图像和修复。


## 稳定扩散XL



要使用 SDXL 加载并运行推理，请使用
 `OVStableDiffusionXLPipeline`
 ：



```
from optimum.intel import OVStableDiffusionXLPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = OVStableDiffusionXLPipeline.from_pretrained(model_id)
prompt = "sailing ship in storm by Rembrandt"
image = pipeline(prompt).images[0]
```


为了进一步加快推理速度，
 [静态重塑](#stable-diffusion)
 模型如稳定扩散部分所示。


您可以在 🤗 最优中找到更多示例
 [文档](https://huggingface.co/docs/optimum/intel/inference#stable-diffusion-xl)
 ，并且支持在 OpenVINO 中运行 SDXL 以进行文本到图像和图像到图像。