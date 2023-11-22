# 哈瓦那高迪

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/optimization/habana>
>
> 原始地址：<https://huggingface.co/docs/diffusers/optimization/habana>


🤗 Diffusers 通过 🤗 与 Habana Gaudi 兼容
 [最佳](https://huggingface.co/docs/optimum/habana/usage_guides/stable_diffusion)
 。跟着
 [安装](https://docs.habana.ai/en/latest/Installation_Guide/index.html)
 安装 SynapseAI 和 Gaudi 驱动程序的指南，然后安装 Optimum Habana：



```
python -m pip install --upgrade-strategy eager optimum[habana]
```


要在 Gaudi 上生成具有稳定扩散 1 和 2 的图像，您需要实例化两个实例：


* `高迪稳定扩散管道`
 ，文本到图像生成的管道。
* `GaudiDDIMSScheduler`
 ，一个高迪优化的调度程序。


当你初始化管道时，你必须指定
 `use_habana=True`
 要将其部署在 HPU 上并获得尽可能快的生成，您应该启用
 **HPU 图表**
 和
 `use_hpu_graphs=True`
 。


最后指定一个
 `高迪配置`
 可以从以下位置下载
 [哈瓦那](https://huggingface.co/Habana)
 Hub 上的组织。



```
from optimum.habana import GaudiConfig
from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline

model_name = "stabilityai/stable-diffusion-2-base"
scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
pipeline = GaudiStableDiffusionPipeline.from_pretrained(
    model_name,
    scheduler=scheduler,
    use_habana=True,
    use_hpu_graphs=True,
    gaudi_config="Habana/stable-diffusion-2",
)
```


现在您可以根据一个或多个提示调用管道批量生成图像：



```
outputs = pipeline(
    prompt=[
        "High quality photo of an astronaut riding a horse in space",
        "Face of a yellow cat, high resolution, sitting on a park bench",
    ],
    num_images_per_prompt=10,
    batch_size=4,
)
```


欲了解更多信息，请查看 🤗 Optimum Habana
 [文档](https://huggingface.co/docs/optimum/habana/usage_guides/stable_diffusion)
 和
 [示例](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion)
 官方 Github 存储库中提供。


## 基准



我们对 Habana 的第一代 Gaudi 和 Gaudi2 进行了基准测试
 [Habana/stable-diffusion](https://huggingface.co/Habana/stable-diffusion)
 和
 [Habana/stable-diffusion-2](https://huggingface.co/Habana/stable-diffusion-2)
 Gaudi 配置（混合精度 bf16/fp32）来展示其性能。


为了
 [稳定扩散 v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 在 512x512 图像上：


|  | 	 Latency (batch size = 1)	  | 	 Throughput	  |
| --- | --- | --- |
| 	 first-generation Gaudi	  | 	 3.80s	  | 	 0.308 images/s (batch size = 8)	  |
| 	 Gaudi2	  | 	 1.33s	  | 	 1.081 images/s (batch size = 8)	  |


为了
 [稳定扩散 v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
 在 768x768 图像上：


|  | 	 Latency (batch size = 1)	  | 	 Throughput	  |
| --- | --- | --- |
| 	 first-generation Gaudi	  | 	 10.2s	  | 	 0.108 images/s (batch size = 4)	  |
| 	 Gaudi2	  | 	 3.17s	  | 	 0.379 images/s (batch size = 8)	  |