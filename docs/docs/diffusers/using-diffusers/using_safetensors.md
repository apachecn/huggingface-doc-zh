# 加载安全张量

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/using_safetensors>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/using_safetensors>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


[安全张量](https://github.com/huggingface/safetensors)
 是一种用于存储和加载张量的安全且快速的文件格式。通常，PyTorch 模型权重被保存或
 *腌制*
 变成一个
 `.bin`
 使用 Python 的文件
 [`pickle`](https://docs.python.org/3/library/pickle.html)
 公用事业。然而，
 `泡菜`
 不安全，pickled 文件可能包含可以执行的恶意代码。 safetensors 是一个安全的替代品
 `泡菜`
 ，使其成为共享模型权重的理想选择。


本指南将向您展示如何加载
 `.safetensor`
 文件，以及如何将以其他格式存储的稳定扩散模型权重转换为
 `.safetensor`
 。在开始之前，请确保已安装 safetensors：



```
# uncomment to install the necessary libraries in Colab
#!pip install safetensors
```


如果你看一下
 [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main)
 存储库，您将看到里面的权重
 `文本编码器`
 ,
 `乌内特`
 和
 `瓦伊`
 子文件夹存储在
 `.safetensors`
 格式。默认情况下，🤗 Diffusers 会自动加载这些
 `.safetensors`
 如果它们在模型存储库中可用，则从其子文件夹中获取文件。


为了更明确的控制，您可以选择设置
 `use_safetensors=True`
 （如果
 `安全张量`
 尚未安装，您将收到一条错误消息，要求您安装）：



```
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
```


但是，模型权重不一定像上面的示例那样存储在单独的子文件夹中。有时，所有权重都存储在一个
 `.safetensors`
 文件。在这种情况下，如果权重是稳定扩散权重，则可以直接使用
 [来自\_single\_file()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.from_single_file)
 方法：



```
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_single_file(
    "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
)
```


## 转换为安全张量



并非集线器上的所有重量都可用
 `.safetensors`
 格式，您可能会遇到存储为的权重
 `.bin`
 。在这种情况下，请使用
 [转换空间](https://huggingface.co/spaces/diffusers/convert)
 将权重转换为
 `.safetensors`
 。转换空间下载腌制的权重，对其进行转换，然后打开拉取请求以上传新转换的权重
 `.safetensors`
 集线器上的文件。这样，如果腌制文件中包含任何恶意代码，它们就会被上传到集线器 - 该集线器有一个
 [安全扫描仪](https://huggingface.co/docs/hub/security-pickle#hubs-security-scanner)
 检测不安全的文件和可疑的 pickle 导入 - 而不是您的计算机。


您可以将该模型与新的一起使用
 `.safetensors`
 通过指定对拉取请求的引用来指定权重
 `修订`
 参数（您也可以在此测试
 [检查公关](https://huggingface.co/spaces/diffusers/check_pr)
 集线器上的空间），例如
 `refs/pr/22`
 :



```
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", revision="refs/pr/22", use_safetensors=True
)
```


## 为什么使用安全张量？



使用安全张量有几个原因：


* 安全是使用 safetensors 的首要原因。随着开源和模型分发的增长，能够相信您下载的模型权重不包含任何恶意代码非常重要。 safetensors 中标头的当前大小会阻止解析非常大的 JSON 文件。
* 切换模型之间的加载速度是使用 safetensors 的另一个原因，它执行张量的零复制。与以下相比，它特别快
 `泡菜`
 如果您将权重加载到 CPU（默认情况），则与直接将权重加载到 GPU 时一样快（如果不是更快的话）。仅当模型已加载时，您才会注意到性能差异，而如果您正在下载权重或首次加载模型，则不会注意到性能差异。


加载整个管道所需的时间：



```
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", use_safetensors=True)
"Loaded in safetensors 0:00:02.033658"
"Loaded in PyTorch 0:00:02.663379"
```


但加载 500MB 模型权重的实际时间仅为：



```
safetensors: 3.4873ms
PyTorch: 172.7537ms
```
* Lazy loading is also supported in safetensors, which is useful in distributed settings to only load some of the tensors. This format allowed the
 BLOOM 
 model to be loaded in 45 seconds on 8 GPUs instead of 10 minutes with regular PyTorch weights.