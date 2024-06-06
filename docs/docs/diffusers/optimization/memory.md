> 翻译任务

# [](#reduce-memory-usage)减少内存使用量(Reduce memory usage)

> 译者：[疾风兔X](https://github.com/jifnegtu)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/optimization/memory>
>
> 原始地址：<https://huggingface.co/docs/diffusers/optimization/memory>



使用扩散模型的一个障碍是其需要大量的内存。为了克服这个挑战，有几种减少内存消耗的技术可供使用，这些技术甚至能让一些最大的模型在免费层级或消费级GPU上运行。有些技术还可以相互结合使用，进一步降低内存的使用量。

>在许多情况下，优化内存使用或提升速度实际上会间接促进另一方面的性能改善，因此你应该尽可能地尝试同时在这两个方向上进行优化。本指南侧重于最大程度地减少内存使用，但您也可以了解有关如何[加快推理(Speed up inference.)](fp16)速度的更多信息。

以下结果是根据提示生成单个 512x512 图像获得的，该图像是宇航员在火星上骑马的照片，在 Nvidia Titan RTX 上以 50 个 DDIM 步长运行，展示了由于内存消耗减少而可以预期的加速。

|  | latency | speed-up |
| --- | --- | --- |
| original | 9.50s | x1 |
| fp16 | 3.61s | x2.63 |
| channels last | 3.30s | x2.88 |
| traced UNet | 3.21s | x2.96 |
| memory-efficient attention | 2.63s | x3.61 |

## [](#sliced-vae)Sliced VAE

Sliced VAE技术允许你在有限的VRAM条件下解码大规模的图像批次，或是处理包含32张以上图像的批次，通过逐张解码潜变量批次的方式来实现。如果您安装了 xFormers，您可能希望将其与 [enable\_xformers\_memory\_efficient\_attention（）](/docs/diffusers/v0.28.2/en/api/models/overview#diffusers.ModelMixin.enable_xformers_memory_efficient_attention) 结合使用，以进一步减少内存使用。

要在推断前使用Sliced VAE，你需要在你的pipeline上调用 [enable\_vae\_slicing（）](/docs/diffusers/v0.28.2/en/api/pipelines/latent_consistency_models#diffusers.LatentConsistencyModelPipeline.enable_vae_slicing)：

```
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_vae_slicing()
#pipe.enable_xformers_memory_efficient_attention()
images = pipe([prompt] * 32).images
```
您可能会看到多图像批处理的VAE解码性能略有提升，并且对单图像批处理的性能应该没有影响。

## [](#tiled-vae)Tiled VAE

Tiled VAE处理还可以在有限的VRAM上处理大型图像（例如，在8GB VRAM上生成4k图像），方法是将图像拆分为重叠的图块，解码图块，然后将输出混合在一起以组成最终图像。如果您安装了 xFormers，您还应该使用带有 [enable\_xformers\_memory\_efficient\_attention（） ](/docs/diffusers/v0.28.2/en/api/models/overview#diffusers.ModelMixin.enable_xformers_memory_efficient_attention)的Tiled VAE，以进一步减少内存使用。

要使用Tiled VAE 处理，请在推理之前在管道上调用 [enable\_vae\_tiling（）](/docs/diffusers/v0.28.2/en/api/pipelines/latent_consistency_models#diffusers.LatentConsistencyModelPipeline.enable_vae_tiling)：

```
import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
prompt = "a beautiful landscape photograph"
pipe.enable_vae_tiling()
#pipe.enable_xformers_memory_efficient_attention()

image = pipe([prompt], width=3840, height=2224, num_inference_steps=20).images[0]
```
输出图像具有一些图块到图块的色调变化，因为图块是单独解码的，但您不应该看到图块之间任何尖锐和明显的接缝。对于 512x512 或更小的图像，平铺将处于关闭状态。

## [](#cpu-offloading)CPU 卸载 (CPU offloading)

将权重卸载到 CPU 并在执行前向传递时仅将它们加载到 GPU 上也可以节省内存。通常，此技术可以将内存消耗减少到 3GB 以下。

要执行 CPU 卸载，请调用 [enable\_sequential\_cpu\_offload（）](/docs/diffusers/v0.28.2/en/api/pipelines/overview#diffusers.DiffusionPipeline.enable_sequential_cpu_offload)：

```
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
)

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_sequential_cpu_offload()
image = pipe(prompt).images[0]
```
CPU 卸载适用于子模块，而不是整个模型。这是最小化内存消耗的最佳方法，但由于扩散过程的迭代性质，推理速度要慢得多。管道的 UNet 组件运行数次（多达`num_inference_steps` ）; 每次，不同的 UNet 子模块都会根据需要按顺序加载和卸载，从而导致大量内存传输。

>如果要优化速度，请考虑使用[模型卸载(model offloading)](#model-offloading)，因为它要快得多。权衡是您的内存节省不会那么大。

>使用 [enable\_sequential\_cpu\_offload（）](/docs/diffusers/v0.28.2/en/api/pipelines/overview#diffusers.DiffusionPipeline.enable_sequential_cpu_offload) 时，不要事先将管道移动到 CUDA，否则内存消耗的减少只会非常有限（有关详细信息，请参阅此[问题](https://github.com/huggingface/diffusers/issues/1934)）。

>[enable\_sequential\_cpu\_offload（）](/docs/diffusers/v0.28.2/en/api/pipelines/overview#diffusers.DiffusionPipeline.enable_sequential_cpu_offload) 是一个有状态的操作，它会在模型上安装hook（钩子）。

## [](#model-offloading)模型卸载 (Model offloading)

>模型卸载需要 🤗 Accelerate 版本 0.17.0 或更高版本。

[顺序 CPU 卸载 (Sequential CPU offloading )](#cpu-offloading)可保留大量内存，但会使推理速度变慢，因为子模块会根据需要移动到 GPU，并且在新模块运行时会立即返回到 CPU。

完整模型卸载是一种将整个模型移动到 GPU 的替代方法，而不是处理每个模型的组成*子模块*。对推理时间的影响可以忽略不计（与将管道移动到 `cuda` 相比），它仍然提供了一些内存节省。

在模型卸载期间，只有管道的一个主要组件（通常是文本编码器、UNet 和 VAE） 放置在 GPU 上，而其他人则在 CPU 上等待。像 UNet 这样运行多次迭代的组件会保留在 GPU 上，直到不再需要它们。

通过在管道上调用 [enable\_model\_cpu\_offload（）](/docs/diffusers/v0.28.2/en/api/pipelines/overview#diffusers.DiffusionPipeline.enable_model_cpu_offload) 来启用模型卸载：

```
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
)

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_model_cpu_offload()
image = pipe(prompt).images[0]
```
>为了在调用模型后正确卸载模型，需要运行整个管道，并按管道的预期顺序调用模型。如果在安装挂钩后在管道上下文之外重用模型，请谨慎行事。有关更多信息，请参阅[删除钩子 ( Removing Hooks)](https://huggingface.co/docs/accelerate/en/package_reference/big_modeling#accelerate.hooks.remove_hook_from_module)。

>[enable\_model\_cpu\_offload（）](/docs/diffusers/v0.28.2/en/api/pipelines/overview#diffusers.DiffusionPipeline.enable_model_cpu_offload) 是一个有状态的操作，它在模型上安装了hook（钩子），并在pipeline上保持了一定的状态。

## [](#channels-last-memory-format)Channels-last 存储格式 (Channels-last memory format)

Channels-last内存格式是一种替代的内存布局方式，用于保存NCHW张量的维度顺序。在channels-last格式下，通道成为了最密集的维度（即按照像素顺序存储图像）。由于并非所有运算符目前都支持channels-last格式，因此在某些情况下可能会导致性能下降，但你仍然应该尝试一下，看看是否适用于你的模型。

例如，要设置pipeline中的UNet使用channels-last格式，请执行以下操作：

```
print(pipe.unet.conv_out.state_dict()["weight"].stride())  # (2880, 9, 3, 1)
pipe.unet.to(memory_format=torch.channels_last)  # in-place operation
print(
    pipe.unet.conv_out.state_dict()["weight"].stride()
)  # (2880, 1, 960, 320) having a stride of 1 for the 2nd dimension proves that it works
```
## [](#tracing)跟踪 (Tracing)

追踪（Tracing）是通过模型运行一个示例输入张量，并捕捉该输入在穿过模型各层时所执行的操作。返回的可执行文件或`ScriptFunction`会通过即时编译（just-in-time compilation）进行优化。

要对UNet进行追踪：

```
import time
import torch
from diffusers import StableDiffusionPipeline
import functools

# torch disable grad
torch.set_grad_enabled(False)

# set variables
n_experiments = 2
unet_runs_per_experiment = 50


# load inputs
def generate_inputs():
    sample = torch.randn((2, 4, 64, 64), device="cuda", dtype=torch.float16)
    timestep = torch.rand(1, device="cuda", dtype=torch.float16) * 999
    encoder_hidden_states = torch.randn((2, 77, 768), device="cuda", dtype=torch.float16)
    return sample, timestep, encoder_hidden_states


pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")
unet = pipe.unet
unet.eval()
unet.to(memory_format=torch.channels_last)  # use channels_last memory format
unet.forward = functools.partial(unet.forward, return_dict=False)  # set return_dict=False as default

# warmup
for _ in range(3):
    with torch.inference_mode():
        inputs = generate_inputs()
        orig_output = unet(*inputs)

# trace
print("tracing..")
unet_traced = torch.jit.trace(unet, inputs)
unet_traced.eval()
print("done tracing")


# warmup and optimize graph
for _ in range(5):
    with torch.inference_mode():
        inputs = generate_inputs()
        orig_output = unet_traced(*inputs)


# benchmarking
with torch.inference_mode():
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(unet_runs_per_experiment):
            orig_output = unet_traced(*inputs)
        torch.cuda.synchronize()
        print(f"unet traced inference took {time.time() - start_time:.2f} seconds")
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(unet_runs_per_experiment):
            orig_output = unet(*inputs)
        torch.cuda.synchronize()
        print(f"unet inference took {time.time() - start_time:.2f} seconds")

# save the model
unet_traced.save("unet_traced.pt")
```
将管道的`unet`属性替换为跟踪的模型：

```
from diffusers import StableDiffusionPipeline
import torch
from dataclasses import dataclass


@dataclass
class UNet2DConditionOutput:
    sample: torch.Tensor


pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

# use jitted unet
unet_traced = torch.jit.load("unet_traced.pt")


# del pipe.unet
class TracedUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = pipe.unet.config.in_channels
        self.device = pipe.unet.device

    def forward(self, latent_model_input, t, encoder_hidden_states):
        sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)


pipe.unet = TracedUNet()

with torch.inference_mode():
    image = pipe([prompt] * 1, num_inference_steps=50).images[0]
```
## [](#memory-efficient-attention)记忆-高效注意 (Memory-efficient attention)

最近在注意力块中优化带宽的工作极大地加快了速度并减少了 GPU 内存的使用。最新的内存效率注意力类型是 [Flash Attention](https://arxiv.org/abs/2205.14135)（您可以在 [HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention) 上查看原始代码）。

>如果你安装的PyTorch版本是2.0或更高，那么在启用`xformers`时，不应该期望在推理速度上有所提升。

要使用 Flash Attention，请安装以下软件：

+   PyTorch > 1.12 版
+   CUDA可用
+   [xFormers](xformers)

然后在管道上调用 [enable\_xformers\_memory\_efficient\_attention（）：](/docs/diffusers/v0.28.2/en/api/models/overview#diffusers.ModelMixin.enable_xformers_memory_efficient_attention)

```
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()

with torch.inference_mode():
    sample = pipe("a small cat")

# optional: You can disable it via
# pipe.disable_xformers_memory_efficient_attention()
```
使用`xformers`时的迭代速度应与[此处](torch2.0)所述的 PyTorch 2.0 的迭代速度相匹配。