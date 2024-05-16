# [](#stable-video-diffusion)Stable Video Diffusion

>译者：[疾风兔X](https://github.com/jifnegtu)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/svd>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/svd>



![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)

![在 Studio 实验室中打开](https://studiolab.sagemaker.aws/studiolab.svg)

[Stable Video Diffusion(SVD)](https://huggingface.co/papers/2311.15127) 是一种功能强大的图像到视频生成模型，能够根据输入的图像生成2-4秒高分辨率（576x1024） 视频。

本指南将向您展示如何使用 SVD 从图像生成短视频。

在开始之前，请确保已安装以下库：

```py
!pip install -q -U diffusers transformers accelerate 
```

该模型是 [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) 和 [SVD-XT](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) 的两种变体。SVD checkpoint被训练来生成14帧视频，而SVD-XT checkpoint则经过进一步微调，用于生成25帧的视频。

本指南中，您将使用SVD-XT checkpoint。

```py
import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)
```
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png)

“火箭的源图像”

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/output_rocket.gif)

“从源图像生成的视频”

## [](#torchcompile)torch.compile

通过[编译](../optimization/torch2.0#torchcompile) UNet，您可以以略微增加的内存为代价获得 20-25% 的加速。

```py
- pipe.enable_model_cpu_offload()
+ pipe.to("cuda")
+ pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```
## [](#reduce-memory-usage)减少内存使用量（Reduce memory usage）

视频生成非常耗内存，因为您实际上是一次性生成`num_frames`的所有帧，类似于具有高批量大小的文本到图像生成。为了降低内存需求，有多个选项可以权衡推理速度以降低内存需求：

+   启用模型卸载（model offloading）：一旦不再需要管道的每个组件，就会将其卸载到 CPU。
+   启用前向传递分块（ feed-forward chunking）：前向传播层不是以一个巨大的批次运行一次，而是分批循环运行。
+   减小解码块大小（`decode_chunk_size`）：VAE（变分自编码器）以块的形式解码帧，而不是一次性解码所有帧。设置`decode_chunk_size=1`意味着一次只解码一帧，这样使用的内存最少（我们建议根据你的GPU内存情况来调整这个值），但可能会导致视频出现闪烁现象。这是因为每一帧独立解码可能导致相邻帧之间的连贯性稍差。

```py
- pipe.enable_model_cpu_offload()
- frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]
+ pipe.enable_model_cpu_offload()
+ pipe.unet.enable_forward_chunking()
+ frames = pipe(image, decode_chunk_size=2, generator=generator, num_frames=25).frames[0]
```
同时使用所有这些技巧应该会将内存要求降低到低于 8GB VRAM。

## [](#micro-conditioning)微调理（Micro-conditioning）

除了调节图像外，Stable Diffusion Video 还接受微调节，这允许对生成的视频进行更多控制：

+   `fps`：生成视频的每秒帧数。
+   `motion_bucket_id`：用于生成视频的 Motion Bucket ID。这可用于控制生成的视频的运动。增加运动桶 ID 会增加生成视频的运动。
+   `noise_aug_strength`：添加到调节图像的噪点量。该值越高，视频与条件图像的相似度就越低。增加此值也会增加生成的视频的运动。

例如，要生成具有更多运动的视频，请使用`motion_bucket_id`和`noise_aug_strength`微调参数：

```py
import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
  "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1).frames[0]
export_to_video(frames, "generated.mp4", fps=7)
```
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/output_rocket_with_conditions.gif)