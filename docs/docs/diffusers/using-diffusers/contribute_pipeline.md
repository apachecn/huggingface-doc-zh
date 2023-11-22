# 贡献社区管道

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/contribute_pipeline>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/contribute_pipeline>


💡 查看 GitHub Issue
 [#841](https://github.com/huggingface/diffusers/issues/841)
 了解更多有关我们为何添加社区管道以帮助每个人轻松共享其工作而不会减慢速度的背景信息。


社区管道允许您添加任何您想要的附加功能
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 。在之上构建的主要好处
 `扩散管道`
 任何人都可以通过添加一个参数来加载和使用您的管道，从而使社区非常容易访问。


本指南将向您展示如何创建社区管道并解释其工作原理。为了简单起见，您将创建一个“一步”管道，其中
 `UNet`
 执行一次前向传递并调用调度程序一次。


## 初始化管道



您应该首先创建一个
 `one_step_unet.py`
 您的社区管道的文件。在此文件中，创建一个继承自的管道类
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 能够从集线器加载模型权重和调度程序配置。一步式管道需要一个
 `UNet`
 和一个调度程序，因此您需要将它们作为参数添加到
 `__init__`
 功能：



```
from diffusers import DiffusionPipeline
import torch


class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
    def \_\_init\_\_(self, unet, scheduler):
        super().__init__()
```


确保您的管道及其组件（
 `乌内特`
 和
 `调度程序`
 ）可以保存为
 [save\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.save_pretrained)
 ，将它们添加到
 `注册模块`
 功能：



```
  from diffusers import DiffusionPipeline
  import torch

  class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
      def __init__(self, unet, scheduler):
          super().__init__()

+ self.register\_modules(unet=unet, scheduler=scheduler)
```


酷，那个
 `__init__`
 步骤已完成，您现在可以进入前向传递！ 🔥


## 定义前向传播



在前向传播中，我们建议将其定义为
 `__call__`
 ，您有完全的创作自由来添加您想要的任何功能。对于我们令人惊叹的一步管道，创建一个随机图像并仅调用
 `乌内特`
 和
 `调度程序`
 通过设置一次
 `时间步=1`
 :



```
  from diffusers import DiffusionPipeline
  import torch


  class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
      def __init__(self, unet, scheduler):
          super().__init__()

          self.register_modules(unet=unet, scheduler=scheduler)

+ def \_\_call\_\_(self):
+ image = torch.randn(
+ (1, self.unet.config.in\_channels, self.unet.config.sample\_size, self.unet.config.sample\_size),
+ )
+ timestep = 1

+ model\_output = self.unet(image, timestep).sample
+ scheduler\_output = self.scheduler.step(model\_output, timestep, image).prev\_sample

+ return scheduler\_output
```


就是这样！ 🚀 您现在可以通过传递一个来运行这个管道
 `乌内特`
 和
 `调度程序`
 对它：



```
from diffusers import DDPMScheduler, UNet2DModel

scheduler = DDPMScheduler()
unet = UNet2DModel()

pipeline = UnetSchedulerOneForwardPipeline(unet=unet, scheduler=scheduler)

output = pipeline()
```


但更好的是，如果管道结构相同，您可以将预先存在的权重加载到管道中。例如，您可以加载
 [`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32)
 权重进入一步管道：



```
pipeline = UnetSchedulerOneForwardPipeline.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True)

output = pipeline()
```


## 分享您的管道



在 🧨 扩散器上打开拉取请求
 [存储库](https://github.com/huggingface/diffusers)
 添加你很棒的管道
 `one_step_unet.py`
 到
 [示例/社区](https://github.com/huggingface/diffusers/tree/main/examples/community)
 子文件夹。


一旦合并，任何人
 `扩散器 >= 0.4.0`
 安装后可以通过在中指定来神奇地使用这个管道🪄
 `自定义管道`
 争论：



```
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "google/ddpm-cifar10-32", custom_pipeline="one\_step\_unet", use_safetensors=True
)
pipe()
```


共享社区管道的另一种方法是上传
 `one_step_unet.py`
 直接归档到您的首选
 [模型存储库](https://huggingface.co/docs/hub/models-uploading)
 在集线器上。而不是指定
 `one_step_unet.py`
 文件，将模型存储库 ID 传递给
 `自定义管道`
 争论：



```
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "google/ddpm-cifar10-32", custom_pipeline="stevhliu/one\_step\_unet", use_safetensors=True
)
```


请查看下表来比较两个共享工作流程，以帮助您确定最适合您的选择：


|  | 	 GitHub community pipeline	  | 	 HF Hub community pipeline	  |
| --- | --- | --- |
| 	 usage	  | 	 same	  | 	 same	  |
| 	 review process	  | 	 open a Pull Request on GitHub and undergo a review process from the Diffusers team before merging; may be slower	  | 	 upload directly to a Hub repository without any review; this is the fastest workflow	  |
| 	 visibility	  | 	 included in the official Diffusers repository and documentation	  | 	 included on your HF Hub profile and relies on your own usage/promotion to gain visibility	  |


💡 您可以在社区管道文件中使用您想要的任何包 - 只要用户安装了它，一切都会正常工作。确保您有且仅有一个继承自的管道类
 `扩散管道`
 因为这是自动检测到的。


## 社区管道如何运作？



社区管道是一个继承自
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 意思是：


* 可以加载
 `自定义管道`
 争论。
* 模型权重和调度器配置从加载
 `预训练模型名称或路径`
 。
* 在社区管道中实现功能的代码定义在
 `管道.py`
 文件。


有时您无法从官方存储库加载所有管道组件权重。在这种情况下，其他组件应直接传递到管道：



```
from diffusers import DiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPModel

model_id = "CompVis/stable-diffusion-v1-4"
clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id, torch_dtype=torch.float16)

pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    custom_pipeline="clip\_guided\_stable\_diffusion",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
```


社区管道背后的魔力包含在以下代码中。它允许从 GitHub 或 Hub 加载社区管道，并且所有 🧨 Diffusers 包都可以使用它。



```
# 2. Load the pipeline class, if using custom module then load it from the hub
# if we load from explicit class, let's use it
if custom_pipeline is not None:
    pipeline_class = get_class_from_dynamic_module(
        custom_pipeline, module_file=CUSTOM_PIPELINE_FILE_NAME, cache_dir=custom_pipeline
    )
elif cls != DiffusionPipeline:
    pipeline_class = cls
else:
    diffusers_module = importlib.import_module(cls.__module__.split(".")[0])
    pipeline_class = getattr(diffusers_module, config_dict["\_class\_name"])
```