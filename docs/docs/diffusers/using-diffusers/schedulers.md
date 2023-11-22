# 调度程序

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/schedulers>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/schedulers>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


扩散管道本质上是彼此部分独立的扩散模型和调度器的集合。这意味着人们能够切换部分管道以更好地定制
通往某个用例的管道。最好的例子就是
 [调度程序](../api/schedulers/概述)
 。


而扩散模型通常简单地定义从噪声到噪声较小样本的前向传播，
调度程序定义整个去噪过程，
 *IE。*
 :


* 降噪步骤有多少？
* 随机性还是确定性？
* 使用什么算法来查找去噪样本？


它们可能非常复杂，并且经常定义之间的权衡
 **去噪速度**
 和
 **降噪质量**
 。
定量测量哪个调度程序最适合给定的扩散管道是极其困难的，因此通常建议简单地尝试哪个最有效。


以下段落展示了如何使用 🧨 Diffusers 库执行此操作。


## 加载管道



让我们从加载开始
 [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 模型中的
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 :



```
from huggingface_hub import login
from diffusers import DiffusionPipeline
import torch

login()

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
```


接下来，我们将其移至 GPU：



```
pipeline.to("cuda")
```


## 访问调度程序



调度器始终是管道的组件之一，通常称为
 `“调度程序”`
 。
因此可以通过以下方式访问它
 `“调度程序”`
 财产。



```
pipeline.scheduler
```


**输出**
 :



```
PNDMScheduler {
  "\_class\_name": "PNDMScheduler",
  "\_diffusers\_version": "0.21.4",
  "beta\_end": 0.012,
  "beta\_schedule": "scaled\_linear",
  "beta\_start": 0.00085,
  "clip\_sample": false,
  "num\_train\_timesteps": 1000,
  "set\_alpha\_to\_one": false,
  "skip\_prk\_steps": true,
  "steps\_offset": 1,
  "timestep\_spacing": "leading",
  "trained\_betas": null
}
```


我们可以看到调度器的类型是
 [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler)
 。
太酷了，现在让我们将调度程序的性能与其他调度程序进行比较。
首先，我们定义一个提示，我们将在其上测试所有不同的调度程序：



```
prompt = "A photograph of an astronaut riding a horse on Mars, high resolution, high definition."
```


接下来，我们从随机种子创建一个生成器，这将确保我们可以生成相似的图像并运行管道：



```
generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator).images[0]
image
```


![](https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_pndm.png)


## 更改调度程序



现在我们展示更改管道的调度程序是多么容易。每个调度程序都有一个属性
 `兼容机`
 它定义了所有兼容的调度程序。您可以查看稳定扩散管道的所有可用、兼容的调度程序，如下所示。



```
pipeline.scheduler.compatibles
```


**输出**
 :



```
[diffusers.utils.dummy\_torch\_and\_torchsde\_objects.DPMSolverSDEScheduler,
 diffusers.schedulers.scheduling\_euler\_discrete.EulerDiscreteScheduler,
 diffusers.schedulers.scheduling\_lms\_discrete.LMSDiscreteScheduler,
 diffusers.schedulers.scheduling\_ddim.DDIMScheduler,
 diffusers.schedulers.scheduling\_ddpm.DDPMScheduler,
 diffusers.schedulers.scheduling\_heun\_discrete.HeunDiscreteScheduler,
 diffusers.schedulers.scheduling\_dpmsolver\_multistep.DPMSolverMultistepScheduler,
 diffusers.schedulers.scheduling\_deis\_multistep.DEISMultistepScheduler,
 diffusers.schedulers.scheduling\_pndm.PNDMScheduler,
 diffusers.schedulers.scheduling\_euler\_ancestral\_discrete.EulerAncestralDiscreteScheduler,
 diffusers.schedulers.scheduling\_unipc\_multistep.UniPCMultistepScheduler,
 diffusers.schedulers.scheduling\_k\_dpm\_2\_discrete.KDPM2DiscreteScheduler,
 diffusers.schedulers.scheduling\_dpmsolver\_singlestep.DPMSolverSinglestepScheduler,
 diffusers.schedulers.scheduling\_k\_dpm\_2\_ancestral\_discrete.KDPM2AncestralDiscreteScheduler]
```


很酷，有很多调度程序可供查看。请随意查看它们各自的类定义：


* [EulerDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/euler#diffusers.EulerDiscreteScheduler)
 ,
* [LMSDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler)
 ,
* [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler)
 ,
* [DDPMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.DDPMScheduler)
 ,
* [HeunDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/heun#diffusers.HeunDiscreteScheduler)
 ,
* [DPMSolverMultistepScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler)
 ,
* [DEISMultistepScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/deis#diffusers.DEISMultistepScheduler)
 ,
* [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler)
 ,
* [EulerAncestralDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/euler_ancestral#diffusers.EulerAncestralDiscreteScheduler)
 ,
* [UniPCMultistepScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)
 ,
* [KDPM2DiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/dpm_discrete#diffusers.KDPM2DiscreteScheduler)
 ,
* [DPMSolverSinglestepScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/singlestep_dpm_solver#diffusers.DPMSolverSinglestepScheduler)
 ,
* [KDPM2AncestralDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/dpm_discrete_ancestral#diffusers.KDPM2AncestralDiscreteScheduler)
 。


我们现在将输入提示与所有其他调度程序进行比较。要更改管道的调度程序，您可以使用
方便的
 `配置`
 属性结合
 [from\_config()](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin.from_config)
 功能。



```
pipeline.scheduler.config
```


返回调度程序配置的字典：


**输出**
 :



```
FrozenDict([('num\_train\_timesteps', 1000),
            ('beta\_start', 0.00085),
            ('beta\_end', 0.012),
            ('beta\_schedule', 'scaled\_linear'),
            ('trained\_betas', None),
            ('skip\_prk\_steps', True),
            ('set\_alpha\_to\_one', False),
            ('prediction\_type', 'epsilon'),
            ('timestep\_spacing', 'leading'),
            ('steps\_offset', 1),
            ('\_use\_default\_values', ['timestep\_spacing', 'prediction\_type']),
            ('\_class\_name', 'PNDMScheduler'),
            ('\_diffusers\_version', '0.21.4'),
            ('clip\_sample', False)])
```


然后可以使用此配置来实例化调度程序
与管道兼容的不同类。这里，
我们将调度程序更改为
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler)
 。



```
from diffusers import DDIMScheduler

pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
```


太酷了，现在我们可以再次运行管道来比较生成质量。



```
generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator).images[0]
image
```


![](https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_ddim.png)


 如果您是 JAX/Flax 用户，请检查
 [本节](#change-the-scheduler-in-flax)
 反而。


## 比较调度程序



到目前为止，我们已经尝试使用两个调度程序运行稳定的扩散管道：
 [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler)
 和
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler)
 。
许多更好的调度程序已经发布，可以用更少的步骤运行；让我们在这里比较一下它们：


[LMSDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler)
 通常会带来更好的结果：



```
from diffusers import LMSDiscreteScheduler

pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)

generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator).images[0]
image
```


![](https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_lms.png)


[EulerDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/euler#diffusers.EulerDiscreteScheduler)
 和
 [EulerAncestralDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/euler_ancestral#diffusers.EulerAncestralDiscreteScheduler)
 只需 30 个步骤即可生成高质量结果。



```
from diffusers import EulerDiscreteScheduler

pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator, num_inference_steps=30).images[0]
image
```


![](https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_euler_discrete.png)


 和：



```
from diffusers import EulerAncestralDiscreteScheduler

pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator, num_inference_steps=30).images[0]
image
```


![](https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_euler_ancestral.png)


[DPMSolverMultistepScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler)
 给出了合理的速度/质量权衡，并且只需 20 个步骤即可运行。



```
from diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

generator = torch.Generator(device="cuda").manual_seed(8)
image = pipeline(prompt, generator=generator, num_inference_steps=20).images[0]
image
```


![](https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_dpm.png)


 正如您所看到的，大多数图像看起来非常相似，而且质量也可以说非常相似。这通常实际上取决于选择哪个调度程序的具体用例。一个好的方法始终是运行多个不同的
调度程序来比较结果。


## 更改 Flax 中的调度程序



如果您是 JAX/Flax 用户，还可以更改默认管道调度程序。这是如何使用 Flax Stable Diffusion pipeline 和超快运行推理的完整示例
 [DPM-Solver++ 调度程序](../api/schedulers/multistep_dpm_solver)
 :



```
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard

from diffusers import FlaxStableDiffusionPipeline, FlaxDPMSolverMultistepScheduler

model_id = "runwayml/stable-diffusion-v1-5"
scheduler, scheduler_state = FlaxDPMSolverMultistepScheduler.from_pretrained(
    model_id,
    subfolder="scheduler"
)
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    revision="bf16",
    dtype=jax.numpy.bfloat16,
)
params["scheduler"] = scheduler_state

# Generate 1 image per parallel device (8 on TPUv2-8 or TPUv3-8)
prompt = "a photo of an astronaut riding a horse on mars"
num_samples = jax.device_count()
prompt_ids = pipeline.prepare_inputs([prompt] * num_samples)

prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 25

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
```


以下 Flax 调度程序是
 *尚不兼容*
 使用亚麻稳定扩散管道：


* `FlaxLMS离散调度器`
* `FlaxDDPMScheduler`