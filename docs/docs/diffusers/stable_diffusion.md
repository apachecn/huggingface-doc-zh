# 有效且高效的扩散

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/stable_diffusion>
>
> 原始地址：<https://huggingface.co/docs/diffusers/stable_diffusion>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


得到
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 生成某种风格的图像或包含您想要的内容可能很棘手。很多时候，你必须运行
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 反复几次才能得到满意的图像。但无中生有是一个计算密集型过程，尤其是当您一遍又一遍地运行推理时。


这就是为什么获得最大收益很重要
 *计算*
 （速度）和
 *记忆*
 (GPU vRAM) 管道的效率可减少推理周期之间的时间，以便您可以更快地进行迭代。


本教程将引导您了解如何使用
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 。


首先加载
 [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 模型：



```
from diffusers import DiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
```


您将使用的示例提示是一位老战士酋长的肖像，但请随意使用您自己的提示：



```
prompt = "portrait photo of a old warrior chief"
```


## 速度



💡 如果您无法使用 GPU，您可以免费使用 GPU 提供商（例如
 [Colab](https://colab.research.google.com/)
 ！


加速推理的最简单方法之一是将管道放置在 GPU 上，就像使用任何 PyTorch 模块一样：



```
pipeline = pipeline.to("cuda")
```


为了确保您可以使用相同的图像并对其进行改进，请使用
 [`生成器`](https://pytorch.org/docs/stable/generated/torch.Generator.html)
 并设置种子
 [再现性](./using-diffusers/再现性)
 :



```
import torch

generator = torch.Generator("cuda").manual_seed(0)
```


现在您可以生成图像：



```
image = pipeline(prompt, generator=generator).images[0]
image
```


![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_1.png)


 此过程在 T4 GPU 上大约需要 30 秒（如果您分配的 GPU 比 T4 更好，则可能会更快）。默认情况下，
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 完整运行推理
 `float32`
 50 个推理步骤的精度。您可以通过切换到较低的精度来加快速度，例如
 `float16`
 或运行更少的推理步骤。


让我们首先加载模型
 `float16`
 并生成图像：



```
import torch

pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
pipeline = pipeline.to("cuda")
generator = torch.Generator("cuda").manual_seed(0)
image = pipeline(prompt, generator=generator).images[0]
image
```


![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_2.png)


 这次，生成图像只花了大约 11 秒，几乎比以前快了 3 倍！


💡 我们强烈建议始终在以下位置运行管道
 `float16`
 ，到目前为止，我们很少看到输出质量出现任何下降。


另一种选择是减少推理步骤的数量。选择更高效的调度程序可以帮助减少步骤数而不牺牲输出质量。您可以在以下位置找到与当前模型兼容的调度程序
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 通过致电
 `兼容机`
 方法：



```
pipeline.scheduler.compatibles
[
    diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler,
    diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler,
    diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler,
    diffusers.schedulers.scheduling_deis_multistep.DEISMultistepScheduler,
    diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler,
    diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler,
    diffusers.schedulers.scheduling_ddpm.DDPMScheduler,
    diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler,
    diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteScheduler,
    diffusers.utils.dummy_torch_and_torchsde_objects.DPMSolverSDEScheduler,
    diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler,
    diffusers.schedulers.scheduling_pndm.PNDMScheduler,
    diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler,
    diffusers.schedulers.scheduling_ddim.DDIMScheduler,
]
```


稳定扩散模型使用
 [PNDMScheduler](/docs/diffusers/v0.23.1/en/api/schedulers/pndm#diffusers.PNDMScheduler)
 默认情况下，通常需要约 50 个推理步骤，但性能更高的调度程序如
 [DPMSolverMultistepScheduler](/docs/diffusers/v0.23.1/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler)
 ，仅需要约 20 或 25 个推理步骤。使用
 [from\_config()](/docs/diffusers/v0.23.1/en/api/configuration#diffusers.ConfigMixin.from_config)
 加载新调度程序的方法：



```
from diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
```


现在设置
 `num_inference_steps`
 至 20：



```
generator = torch.Generator("cuda").manual_seed(0)
image = pipeline(prompt, generator=generator, num_inference_steps=20).images[0]
image
```


![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_3.png)


 太棒了，您已成功将推理时间缩短至 4 秒！ ⚡️


## 记忆



提高管道性能的另一个关键是消耗更少的内存，这间接意味着更高的速度，因为您经常尝试最大化每秒生成的图像数量。查看一次可以生成多少图像的最简单方法是尝试不同的批量大小，直到获得
 `内存不足错误`
 （OOM）。


创建一个函数，根据提示列表生成一批图像
 `发电机`
 。确保分配每个
 `发电机`
 种子，以便在产生良好结果时可以重复使用它。



```
def get\_inputs(batch\_size=1):
    generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    num_inference_steps = 20

    return {"prompt": prompts, "generator": generator, "num\_inference\_steps": num_inference_steps}
```


从...开始
 `batch_size=4`
 并查看您消耗了多少内存：



```
from diffusers.utils import make_image_grid

images = pipeline(**get_inputs(batch_size=4)).images
make_image_grid(images, 2, 2)
```


除非您有一个具有更多 vRAM 的 GPU，否则上面的代码可能会返回一个
 `OOM`
 错误！大部分内存被交叉注意力层占用。您可以按顺序运行该操作，而不是批量运行该操作，以节省大量内存。您所要做的就是配置管道以使用
 [enable\_attention\_slicing()](/docs/diffusers/v0.23.1/en/api/pipelines/stable_diffusion/adapter#diffusers.StableDiffusionXLAdapterPipeline.enable_attention_slicing)
 功能：



```
pipeline.enable_attention_slicing()
```


现在尝试增加
 `批量大小`
 到8！



```
images = pipeline(**get_inputs(batch_size=8)).images
make_image_grid(images, rows=2, cols=4)
```


![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_5.png)


 以前您甚至无法生成一批 4 张图像，现在您可以生成一批 8 张图像，每张图像大约需要 3.5 秒！这可能是在不牺牲质量的情况下在 T4 GPU 上可以达到的最快速度。


## 质量



在最后两节中，您学习了如何使用以下方法来优化管道的速度
 `fp16`
 ，通过使用性能更高的调度程序来减少推理步骤的数量，并启用注意力切片以减少内存消耗。现在您将重点关注如何提高生成图像的质量。


### 


 更好的检查站


最明显的步骤是使用更好的检查点。稳定扩散模型是一个很好的起点，自正式推出以来，也发布了多个改进版本。但是，使用较新的版本并不意味着您会获得更好的结果。您仍然需要自己尝试不同的检查点，并做一些研究（例如使用
 [负面提示](https://minimaxir.com/2022/11/stable-diffusion-negative-prompt/)
 ）以获得最佳结果。


随着该领域的发展，越来越多的高质量检查点经过微调以产生特定的风格。尝试探索
 [中心](https://huggingface.co/models?library=diffusers&sort=downloads)
 和
 [扩散器画廊](https://huggingface.co/spaces/huggingface-projects/diffusers-gallery)
 找到您感兴趣的！


### 


 更好的管道组件


您还可以尝试用更新版本替换当前的管道组件。让我们尝试加载最新的
 [自动编码器](https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main/vae)
 从 Stability AI 进入管道，并生成一些图像：



```
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
pipeline.vae = vae
images = pipeline(**get_inputs(batch_size=8)).images
make_image_grid(images, rows=2, cols=4)
```


![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_6.png)

###


 更好的及时工程


用于生成图像的文本提示非常重要，以至于它被称为
 *及时工程*
 。在快速工程过程中需要考虑的一些注意事项是：


* 我想要生成的图像或类似图像如何存储在互联网上？
* 我可以提供哪些额外的细节来引导模型走向我想要的风格？


考虑到这一点，让我们改进提示以包含颜色和更高质量的细节：



```
prompt += ", tribal panther make up, blue on red, side profile, looking away, serious eyes"
prompt += " 50mm portrait photography, hard rim lighting photography--beta --ar 2:3 --beta --upbeta"
```


使用新提示生成一批图像：



```
images = pipeline(**get_inputs(batch_size=8)).images
make_image_grid(images, rows=2, cols=4)
```


![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_7.png)


 相当令人印象深刻！让我们调整第二张图像 - 对应于
 `发电机`
 有一颗种子
 `1`
 - 添加一些有关主题年龄的文字：



```
prompts = [
    "portrait photo of the oldest warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3 --beta --upbeta",
    "portrait photo of a old warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3 --beta --upbeta",
    "portrait photo of a warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3 --beta --upbeta",
    "portrait photo of a young warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3 --beta --upbeta",
]

generator = [torch.Generator("cuda").manual_seed(1) for _ in range(len(prompts))]
images = pipeline(prompt=prompts, generator=generator, num_inference_steps=25).images
make_image_grid(images, 2, 2)
```


![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_8.png)


## 下一步



在本教程中，您学习了如何优化
 [DiffusionPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 提高计算和内存效率以及提高生成输出的质量。如果您有兴趣让您的管道更快，请查看以下资源：


* 学习如何
 [PyTorch 2.0](./优化/torch2.0)
 和
 [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html)
 推理速度可提高 5 - 300%。在 A100 GPU 上，推理速度最高可提高 50%！
* 如果您无法使用 PyTorch 2，我们建议您安装
 [xFormers](./优化/xformers)
 。其内存高效的注意力机制与 PyTorch 1.13.1 配合得很好，可实现更快的速度并减少内存消耗。
* 其他优化技术，例如模型卸载，包含在
 [本指南](./optimization/fp16)
 。