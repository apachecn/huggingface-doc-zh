# 创建可重复的管道

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/reproducibility>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/reproducibility>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


再现性对于测试、复制结果非常重要，甚至可以用于
 [提高图像质量](reusing_seeds)
 。然而，扩散模型中的随机性是一个理想的属性，因为它允许管道每次运行时生成不同的图像。虽然您不能期望跨平台获得完全相同的结果，但您可以期望结果在一定的容差范围内跨版本和平台可重现。即便如此，容忍度也会根据扩散管道和检查点的不同而有所不同。


这就是为什么了解如何控制扩散模型中的随机源或使用确定性算法很重要。


💡 我们强烈建议阅读 PyTorch 的
 [关于再现性的声明](https://pytorch.org/docs/stable/notes/randomness.html)
 :


>
>
> 不保证在 PyTorch 版本、单独提交或不同平台上获得完全可重现的结果。此外，即使使用相同的种子，CPU 和 GPU 执行之间的结果也可能无法重现。
>
>
>
>


## 控制随机性



在推理过程中，管道严重依赖随机采样操作，其中包括创建
高斯噪声张量用于去噪并向调度步骤添加噪声。


看一下中的张量值
 [DDIMPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/ddim#diffusers.DDIMPipeline)
 经过两个推理步骤后：



```
from diffusers import DDIMPipeline
import numpy as np

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id, use_safetensors=True)

# run pipeline for just two steps and return numpy tensor
image = ddim(num_inference_steps=2, output_type="np").images
print(np.abs(image).sum())
```


运行上面的代码会打印一个值，但如果再次运行它，您会得到一个不同的值。这里发生了什么？


每次运行管道时，
 [`torch.randn`](https://pytorch.org/docs/stable/generated/torch.randn.html)
 使用不同的随机种子来创建逐步去噪的高斯噪声。这会导致每次运行时都会产生不同的结果，这对于扩散管道非常有用，因为它每次都会生成不同的随机图像。


但如果您需要可靠地生成相同的图像，这将取决于您是在 CPU 还是 GPU 上运行管道。


### 


 中央处理器


要在 CPU 上生成可重现的结果，您需要使用 PyTorch
 [`生成器`](https://pytorch.org/docs/stable/generated/torch.randn.html)
 并设置种子：



```
import torch
from diffusers import DDIMPipeline
import numpy as np

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id, use_safetensors=True)

# create a generator for reproducibility
generator = torch.Generator(device="cpu").manual_seed(0)

# run pipeline for just two steps and return numpy tensor
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```


现在，当您运行上面的代码时，它总是打印一个值
 `1491.1711`
 无论如何，因为
 `发电机`
 带有种子的对象被传递给管道的所有随机函数。


如果您在特定硬件和 PyTorch 版本上运行此代码示例，您应该会得到类似（如果不相同）的结果。


💡 一开始通过可能有点不直观
 `发电机`
 对象到管道而不是
只是代表种子的整数值，但这是处理时推荐的设计
PyTorch 中的概率模型为
 `发电机`
 是
 *随机状态*
 那可以是
按顺序传递到多个管道。



### 


 图形处理器


在 GPU 上编写可重现的管道有点棘手，并且无法保证跨不同硬件的完全重现性，因为矩阵乘法（扩散管道需要大量运算）在 GPU 上的确定性不如 CPU。例如，如果您在 GPU 上运行上面相同的代码示例：



```
import torch
from diffusers import DDIMPipeline
import numpy as np

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id, use_safetensors=True)
ddim.to("cuda")

# create a generator for reproducibility
generator = torch.Generator(device="cuda").manual_seed(0)

# run pipeline for just two steps and return numpy tensor
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```


即使您使用相同的种子，结果也不一样，因为 GPU 使用与 CPU 不同的随机数生成器。


为了避免这个问题，🧨 Diffusers 有一个
 `randn_tensor()`
 用于在 CPU 上创建随机噪声的函数，然后在必要时将张量移动到 GPU。这
 `randn_张量`
 函数在管道内的任何地方使用，允许用户
 **总是**
 通过CPU
 `发电机`
 即使管道在 GPU 上运行。


您会发现结果现在更加接近了！



```
import torch
from diffusers import DDIMPipeline
import numpy as np

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id, use_safetensors=True)
ddim.to("cuda")

# create a generator for reproducibility; notice you don't place it on the GPU!
generator = torch.manual_seed(0)

# run pipeline for just two steps and return numpy tensor
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```


💡 如果可重复性很重要，我们建议始终通过 CPU 生成器。
性能损失通常可以忽略不计，并且您会生成更多类似的结果
值与管道在 GPU 上运行时的值相比。


最后，对于更复杂的管道，例如
 [UnCLIPPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/unclip#diffusers.UnCLIPPipeline)
 ，这些往往是极其
容易受到精度误差传播的影响。不要指望会出现类似的结果
不同的 GPU 硬件或 PyTorch 版本。在这种情况下，您需要运行
完全相同的硬件和 PyTorch 版本，可实现完全重现性。


## 确定性算法



您还可以将 PyTorch 配置为使用确定性算法来创建可重现的管道。但是，您应该意识到确定性算法可能比非确定性算法慢，并且您可能会观察到性能下降。但如果可重复性对您来说很重要，那么这就是正确的选择！


当在多个 CUDA 流中启动操作时，会出现不确定性行为。为了避免这种情况，设置环境变量
 [`CUBLAS_WORKSPACE_CONFIG`](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)
 到
 `:16:8`
 在运行时仅使用一种缓冲区大小。


PyTorch 通常会对多种算法进行基准测试以选择最快的算法，但如果您想要可重复性，则应该禁用此功能，因为基准测试可能每次都会选择不同的算法。最后，通过
 '真实'
 到
 [`torch.use_definistic_algorithms`](https://pytorch.org/docs/stable/generated/torch.use_definistic_algorithms.html)
 启用确定性算法。



```
import os

os.environ["CUBLAS\_WORKSPACE\_CONFIG"] = ":16:8"

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```


现在，当您运行相同的管道两次时，您将获得相同的结果。



```
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
import numpy as np

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
g = torch.Generator(device="cuda")

prompt = "A bear is playing a guitar on Times Square"

g.manual_seed(0)
result1 = pipe(prompt=prompt, num_inference_steps=50, generator=g, output_type="latent").images

g.manual_seed(0)
result2 = pipe(prompt=prompt, num_inference_steps=50, generator=g, output_type="latent").images

print("L\_inf dist = ", abs(result1 - result2).max())
"L\_inf dist = tensor(0., device='cuda:0')"
```