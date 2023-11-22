# 蒸馏稳定扩散推理

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/distilled_sd>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/distilled_sd>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


稳定扩散推理可能是一个计算密集型过程，因为它必须迭代地对潜在图像进行去噪以生成图像。为了减少计算负担，您可以使用
 *蒸馏*
 稳定扩散模型的版本
 [Nota AI](https://huggingface.co/nota-ai)
 。他们的稳定扩散模型的精炼版本消除了 UNet 中的一些残差和注意力块，将模型大小减少了 51%，并将 CPU/GPU 的延迟改善了 43%。


读这个
 [博客文章](https://huggingface.co/blog/sd_distillation)
 详细了解知识蒸馏训练如何产生更快、更小、更便宜的生成模型。


让我们加载经过提炼的稳定扩散模型并将其与原始稳定扩散模型进行比较：



```
from diffusers import StableDiffusionPipeline
import torch

distilled = StableDiffusionPipeline.from_pretrained(
    "nota-ai/bk-sdm-small", torch_dtype=torch.float16, use_safetensors=True,
).to("cuda")

original = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, use_safetensors=True,
).to("cuda")
```


根据提示，获取原始模型的推理时间：



```
import time

seed = 2023
generator = torch.manual_seed(seed)

NUM_ITERS_TO_RUN = 3
NUM_INFERENCE_STEPS = 25
NUM_IMAGES_PER_PROMPT = 4

prompt = "a golden vase with different flowers"

start = time.time_ns()
for _ in range(NUM_ITERS_TO_RUN):
    images = original(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT
    ).images
end = time.time_ns()
original_sd = f"{(end - start) / 1e6:.1f}"

print(f"Execution time -- {original\_sd} ms
")
"Execution time -- 45781.5 ms"
```


计算蒸馏模型推理的时间：



```
start = time.time_ns()
for _ in range(NUM_ITERS_TO_RUN):
    images = distilled(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT
    ).images
end = time.time_ns()

distilled_sd = f"{(end - start) / 1e6:.1f}"
print(f"Execution time -- {distilled\_sd} ms
")
"Execution time -- 29884.2 ms"
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/original_sd.png)

 原始稳定扩散（45781.5 ms）


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/distilled_sd.png)

 蒸馏稳定扩散 (29884.2 ms)


## 微型自动编码器



为了进一步加快推理速度，请使用一个微小的蒸馏版本
 [稳定扩散 VAE](https://huggingface.co/sayakpaul/taesdxl-diffusers)
 将潜在图像去噪。将蒸馏稳定扩散模型中的 VAE 替换为微小的 VAE：



```
from diffusers import AutoencoderTiny

distilled.vae = AutoencoderTiny.from_pretrained(
    "sayakpaul/taesd-diffusers", torch_dtype=torch.float16, use_safetensors=True,
).to("cuda")
```


对蒸馏模型和蒸馏 VAE 推理进行计时：



```
start = time.time_ns()
for _ in range(NUM_ITERS_TO_RUN):
    images = distilled(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT
    ).images
end = time.time_ns()

distilled_tiny_sd = f"{(end - start) / 1e6:.1f}"
print(f"Execution time -- {distilled\_tiny\_sd} ms
")
"Execution time -- 27165.7 ms"
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/distilled_sd_vae.png)

 蒸馏稳定扩散 + 微型自动编码器 (27165.7 ms)