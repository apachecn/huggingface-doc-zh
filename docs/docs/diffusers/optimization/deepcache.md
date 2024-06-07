# [](#deepcache)深度缓存 (DeepCache)

> 译者：[疾风兔X](https://github.com/jifnegtu)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/optimization/deepcache>
>
> 原始地址：<https://huggingface.co/docs/diffusers/optimization/deepcache>


[DeepCache](https://huggingface.co/papers/2312.00858) 通过战略性地缓存和重用高级功能来加速 [StableDiffusionPipeline](/docs/diffusers/v0.28.2/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline) 和 [StableDiffusionXLPipeline](/docs/diffusers/v0.28.2/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline)，同时利用 U-Net 架构高效更新低级功能。

首先安装 [DeepCache](https://github.com/horseee/DeepCache)：

```
pip install DeepCache
```
然后加载并启用 [`DeepCacheSDHelper`](https://github.com/horseee/DeepCache#usage)：

```
  import torch
  from diffusers import StableDiffusionPipeline
  pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16).to("cuda")

+ from DeepCache import DeepCacheSDHelper
+ helper = DeepCacheSDHelper(pipe=pipe)
+ helper.set_params(
+     cache_interval=3,
+     cache_branch_id=0,
+ )
+ helper.enable()

  image = pipe("a photo of an astronaut on a moon").images[0]
```
`set_params`方法接受两个参数：`cache_interval` 和 `cache_branch_id` 。`cache_interval` 表示功能缓存的频率，指定为每个缓存操作之间的步骤数。 `cache_branch_id`标识网络的哪个分支（从最浅层到最深层排序）负责执行缓存过程。 选择更低的`cache_branch_id`或更大的`cache_interval`值可能会导致更快的推理速度，但会降低图像质量（这两个超参数的消融实验可以在[论文](https://arxiv.org/abs/2312.00858)中找到）。设置这些参数后，使用 `enable` or `disable` 方法激活或停用 `DeepCacheSDHelper` .

![](https://github.com/horseee/Diffusion_DeepCache/raw/master/static/images/example.png)

您可以在 [WandB 报告](https://wandb.ai/horseee/DeepCache/runs/jwlsqqgt?workspace=user-horseee)中找到更多生成的样本（original pipeline 与 DeepCache）和相应的推理延迟。提示是从 [MS-COCO 2017](https://cocodataset.org/#home) 数据集中随机选择的。

## [](#benchmark)基准 (Benchmark)

我们测试了 DeepCache 在 NVIDIA RTX A5000 上通过 50 个推理步骤加速 [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) 的速度有多快，使用不同的分辨率(resolution)、批处理大小(batch size)、缓存间隔（I）( cache interval I) 和缓存分支（B）(cache branch B) 配置  。

| **Resolution** | **Batch size** | **Original** | **DeepCache(I=3, B=0)** | **DeepCache(I=5, B=0)** | **DeepCache(I=5, B=1)** |
| --- | --- | --- | --- | --- | --- |
| 512 | 8 | 15.96 | 6.88(2.32x) | 5.03(3.18x) | 7.27(2.20x) |
|  | 4 | 8.39 | 3.60(2.33x) | 2.62(3.21x) | 3.75(2.24x) |
|  | 1 | 2.61 | 1.12(2.33x) | 0.81(3.24x) | 1.11(2.35x) |
| 768 | 8 | 43.58 | 18.99(2.29x) | 13.96(3.12x) | 21.27(2.05x) |
|  | 4 | 22.24 | 9.67(2.30x) | 7.10(3.13x) | 10.74(2.07x) |
|  | 1 | 6.33 | 2.72(2.33x) | 1.97(3.21x) | 2.98(2.12x) |
| 1024 | 8 | 101.95 | 45.57(2.24x) | 33.72(3.02x) | 53.00(1.92x) |
|  | 4 | 49.25 | 21.86(2.25x) | 16.19(3.04x) | 25.78(1.91x) |
|  | 1 | 13.83 | 6.07(2.28x) | 4.43(3.12x) | 7.15(1.93x) |