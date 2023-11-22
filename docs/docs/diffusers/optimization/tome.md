# 代币合并

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/optimization/tome>
>
> 原始地址：<https://huggingface.co/docs/diffusers/optimization/tome>


[令牌合并](https://huggingface.co/papers/2303.17604)
 (ToMe) 在基于 Transformer 的网络的前向传递中逐步合并冗余令牌/补丁，这可以加快推理延迟
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 。


您可以从以下位置使用 ToMe
 [`tomesd`](https://github.com/dbolya/tomesd)
 图书馆与
 [`apply_patch`](https://github.com/dbolya/tomesd?tab=readme-ov-file#usage)
 功能：



```
from diffusers import StableDiffusionPipeline
import tomesd

pipeline = StableDiffusionPipeline.from_pretrained(
      "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True,
).to("cuda")
+ tomesd.apply\_patch(pipeline, ratio=0.5)

image = pipeline("a photo of an astronaut riding a horse on mars").images[0]
```


这
 `应用补丁`
 函数暴露了一些
 [参数](https://github.com/dbolya/tomesd#usage)
 帮助在管道推理速度和生成令牌的质量之间取得平衡。最重要的论据是
 `比率`
 它控制在前向传递过程中合并的令牌数量。


据报道
 [论文](https://huggingface.co/papers/2303.17604)
 ，ToMe 可以极大地保留生成图像的质量，同时提高推理速度。通过增加
 `比率`
 ，您可以进一步加快推理速度，但代价是图像质量下降。


为了测试生成图像的质量，我们采样了一些提示
 [Parti 提示](https://parti.research.google/)
 并进行推理
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 具有以下设置：


![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/tome/tome_samples.png)


 我们没有注意到生成的样本质量有任何显着下降，您可以在此处查看生成的样本
 [WandB 报告](https://wandb.ai/sayakpaul/tomesd-results/runs/23j4bj3i?workspace=)
 。如果您有兴趣重现此实验，请使用此
 [脚本](https://gist.github.com/sayakpaul/8cac98d7f22399085a060992f411ecbd)
 。


## 基准测试



我们还对影响进行了基准测试
 `托梅斯德`
 于
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 和
 [xFormers](https://huggingface.co/docs/diffusers/optimization/xformers)
 跨多种图像分辨率启用。结果是在以下开发环境中从 A100 和 V100 GPU 获得的：



```
- `diffusers` version: 0.15.1
- Python version: 3.8.16
- PyTorch version (GPU?): 1.13.1+cu116 (True)
- Huggingface_hub version: 0.13.2
- Transformers version: 4.27.2
- Accelerate version: 0.18.0
- xFormers version: 0.0.16
- tomesd version: 0.1.2
```


要重现此基准测试，请随意使用此
 [脚本](https://gist.github.com/sayakpaul/27aec6bca7eb7b0e0aa4112205850335)
 。结果以秒为单位报告，并且在适用的情况下，我们报告使用 ToMe 和 ToMe + xFormers 时相对于普通管道的加速百分比。


| **GPU**  | **Resolution**  | **Batch size**  | **Vanilla**  | **ToMe**  | **ToMe + xFormers**  |
| --- | --- | --- | --- | --- | --- |
| **A100**  | 	 512	  | 	 10	  | 	 6.88	  | 	 5.26 (+23.55%)	  | 	 4.69 (+31.83%)	  |
|  | 	 768	  | 	 10	  | 	 OOM	  | 	 14.71	  | 	 11	  |
|  |  | 	 8	  | 	 OOM	  | 	 11.56	  | 	 8.84	  |
|  |  | 	 4	  | 	 OOM	  | 	 5.98	  | 	 4.66	  |
|  |  | 	 2	  | 	 4.99	  | 	 3.24 (+35.07%)	  | 	 2.1 (+37.88%)	  |
|  |  | 	 1	  | 	 3.29	  | 	 2.24 (+31.91%)	  | 	 2.03 (+38.3%)	  |
|  | 	 1024	  | 	 10	  | 	 OOM	  | 	 OOM	  | 	 OOM	  |
|  |  | 	 8	  | 	 OOM	  | 	 OOM	  | 	 OOM	  |
|  |  | 	 4	  | 	 OOM	  | 	 12.51	  | 	 9.09	  |
|  |  | 	 2	  | 	 OOM	  | 	 6.52	  | 	 4.96	  |
|  |  | 	 1	  | 	 6.4	  | 	 3.61 (+43.59%)	  | 	 2.81 (+56.09%)	  |
| **V100**  | 	 512	  | 	 10	  | 	 OOM	  | 	 10.03	  | 	 9.29	  |
|  |  | 	 8	  | 	 OOM	  | 	 8.05	  | 	 7.47	  |
|  |  | 	 4	  | 	 5.7	  | 	 4.3 (+24.56%)	  | 	 3.98 (+30.18%)	  |
|  |  | 	 2	  | 	 3.14	  | 	 2.43 (+22.61%)	  | 	 2.27 (+27.71%)	  |
|  |  | 	 1	  | 	 1.88	  | 	 1.57 (+16.49%)	  | 	 1.57 (+16.49%)	  |
|  | 	 768	  | 	 10	  | 	 OOM	  | 	 OOM	  | 	 23.67	  |
|  |  | 	 8	  | 	 OOM	  | 	 OOM	  | 	 18.81	  |
|  |  | 	 4	  | 	 OOM	  | 	 11.81	  | 	 9.7	  |
|  |  | 	 2	  | 	 OOM	  | 	 6.27	  | 	 5.2	  |
|  |  | 	 1	  | 	 5.43	  | 	 3.38 (+37.75%)	  | 	 2.82 (+48.07%)	  |
|  | 	 1024	  | 	 10	  | 	 OOM	  | 	 OOM	  | 	 OOM	  |
|  |  | 	 8	  | 	 OOM	  | 	 OOM	  | 	 OOM	  |
|  |  | 	 4	  | 	 OOM	  | 	 OOM	  | 	 19.35	  |
|  |  | 	 2	  | 	 OOM	  | 	 13	  | 	 10.78	  |
|  |  | 	 1	  | 	 OOM	  | 	 6.66	  | 	 5.54	  |


如上表所示，加速比
 `托梅斯德`
 对于较大的图像分辨率变得更加明显。还有趣的是，随着
 `托梅斯德`
 ，可以在更高分辨率（如 1024x1024）上运行管道。您可以使用以下方法进一步加快推理速度
 [`torch.compile`](torch2.0)
 。