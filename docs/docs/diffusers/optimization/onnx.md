# ONNX 运行时

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/optimization/onnx>
>
> 原始地址：<https://huggingface.co/docs/diffusers/optimization/onnx>


🤗
 [最佳](https://github.com/huggingface/optimum)
 提供与 ONNX 运行时兼容的稳定扩散管道。您需要使用以下命令安装 🤗 Optimum 以获得 ONNX 运行时支持：



```
pip install optimum["onnxruntime"]
```


本指南将向您展示如何将 Stable Diffusion 和 Stable Diffusion XL (SDXL) 管道与 ONNX Runtime 结合使用。


## 稳定扩散



要加载并运行推理，请使用
 [ORTStableDiffusionPipeline](https://huggingface.co/docs/optimum/v1.14.0/en/onnxruntime/package_reference/modeling_ort#optimum.onnxruntime.ORTStableDiffusionPipeline)
 。如果您想加载 PyTorch 模型并将其即时转换为 ONNX 格式，请设置
 `导出=真`
 :



```
from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id, export=True)
prompt = "sailing ship in storm by Leonardo da Vinci"
image = pipeline(prompt).images[0]
pipeline.save_pretrained("./onnx-stable-diffusion-v1-5")
```


批量生成多个提示似乎占用太多内存。当我们研究它时，您可能需要迭代而不是批处理。


要离线导出 ONNX 格式的管道并在以后用于推理，
使用
 [`optimum-cli 导出`](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli)
 命令：



```
optimum-cli export onnx --model runwayml/stable-diffusion-v1-5 sd_v15_onnx/
```


然后执行推理（您不必指定
 `导出=真`
 再次）：



```
from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = "sd\_v15\_onnx"
pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id)
prompt = "sailing ship in storm by Leonardo da Vinci"
image = pipeline(prompt).images[0]
```


![](https://huggingface.co/datasets/optimum/documentation-images/resolve/main/onnxruntime/stable_diffusion_v1_5_ort_sail_boat.png)


 您可以在 🤗 Optimum 中找到更多示例
 [文档](https://huggingface.co/docs/optimum/)
 ，并且稳定扩散支持文本到图像、图像到图像和修复。


## 稳定扩散XL



要使用 SDXL 加载并运行推理，请使用
 [ORTStableDiffusionXLPipeline](https://huggingface.co/docs/optimum/v1.14.0/en/onnxruntime/package_reference/modeling_ort#optimum.onnxruntime.ORTStableDiffusionXLPipeline)
 ：



```
from optimum.onnxruntime import ORTStableDiffusionXLPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = ORTStableDiffusionXLPipeline.from_pretrained(model_id)
prompt = "sailing ship in storm by Leonardo da Vinci"
image = pipeline(prompt).images[0]
```


要以 ONNX 格式导出管道并稍后将其用于推理，请使用
 [`optimum-cli 导出`](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli)
 命令：



```
optimum-cli export onnx --model stabilityai/stable-diffusion-xl-base-1.0 --task stable-diffusion-xl sd_xl_onnx/
```


ONNX 格式的 SDXL 支持文本到图像和图像到图像。