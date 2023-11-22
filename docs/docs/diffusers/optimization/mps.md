# 金属性能着色器 (MPS)

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/optimization/mps>
>
> 原始地址：<https://huggingface.co/docs/diffusers/optimization/mps>


🤗 Diffusers 使用 PyTorch 与 Apple 芯片（M1/M2 芯片）兼容
 [`mps`](https://pytorch.org/docs/stable/notes/mps.html)
 设备，它使用 Metal 框架来利用 MacOS 设备上的 GPU。您需要：


* 配备 Apple Silicon (M1/M2) 硬件的 macOS 计算机
* macOS 12.6 或更高版本（推荐 13.0 或更高版本）
*arm64版本的Python
* [PyTorch 2.0](https://pytorch.org/get-started/locally/)
 （推荐）或 1.13（支持的最低版本
 `议员`
 ）


这
 `议员`
 后端使用 PyTorch 的
 `.to()`
 用于将稳定扩散管道移至 M1 或 M2 设备的接口：



```
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

prompt = "a photo of an astronaut riding a horse on mars"
```


可以批量生成多个提示
 [崩溃](https://github.com/huggingface/diffusers/issues/363)
 或不能可靠工作。我们认为这与
 [`mps`](https://github.com/pytorch/pytorch/issues/84039)
 PyTorch 的后端。在对此进行调查时，您应该迭代而不是批处理。


如果您正在使用
 **PyTorch 1.13**
 ，您需要通过额外的一次性传递来“启动”管道。这是针对第一次推理过程产生的结果与后续推理过程略有不同的问题的临时解决方法。您只需执行一次此遍，只需一个推理步骤后，您就可以丢弃结果。



```
  from diffusers import DiffusionPipeline

  pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("mps")
  pipe.enable_attention_slicing()

  prompt = "a photo of an astronaut riding a horse on mars"
# First-time "warmup" pass if PyTorch version is 1.13
+ \_ = pipe(prompt, num\_inference\_steps=1)

# Results match those from the CPU device after the warmup pass.
  image = pipe(prompt).images[0]
```


## 故障排除



M1/M2 性能对内存压力非常敏感。发生这种情况时，系统会根据需要自动交换，这会显着降低性能。


为了防止这种情况发生，我们建议
 *注意切片*
 以减少推理期间的内存压力并防止交换。如果您的计算机的系统 RAM 小于 64GB，或者您以大于 512×512 像素的非标准分辨率生成图像，这一点尤其重要。致电
 [enable\_attention\_slicing()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.StableDiffusionImageVariationPipeline.enable_attention_slicing)
 在您的管道上运行：



```
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("mps")
pipeline.enable_attention_slicing()
```


注意力切片分多个步骤执行代价高昂的注意力操作，而不是一次全部执行。在没有通用内存的计算机中，它通常可以提高约 20% 的性能，但我们观察到
 *更好的性能*
 在大多数 Apple Silicon 计算机中，除非您有 64GB 或更多 RAM。