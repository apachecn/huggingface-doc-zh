# 加载不同的稳定扩散格式

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/other-formats>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/other-formats>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


稳定扩散模型有不同的格式，具体取决于它们训练和保存的框架以及下载它们的位置。转换这些格式以在 🤗 Diffusers 中使用可以让您使用该库支持的所有功能，例如
 [使用不同的调度程序]（调度程序）
 为了推断，
 [构建自定义管道](write_own_pipeline)
 以及各种技术和方法
 [优化推理速度](../optimization/opt_overview)
 。


我们强烈建议使用
 `.safetensors`
 格式，因为它比传统的 pickled 文件更安全，传统的 pickled 文件很容易受到攻击，并且可以被利用来在您的计算机上执行任何代码（了解更多信息
 [加载安全张量](using_safetensors)
 指导）。


本指南将向您展示如何转换其他稳定扩散格式以与🤗扩散器兼容。


## PyTorch.ckpt



检查站 - 或
 `.ckpt`
 - 格式通常用于存储和保存模型。这
 `.ckpt`
 文件包含整个模型，大小通常为几 GB。虽然您可以加载和使用
 `.ckpt`
 直接使用文件
 [来自\_single\_file()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.from_single_file)
 方法，通常最好将
 `.ckpt`
 文件到 🤗 Diffusers，因此两种格式都可用。


有两种转换选项
 `.ckpt`
 文件：使用空格来转换检查点或转换
 `.ckpt`
 带有脚本的文件。


### 


 用空格转换


最简单、最方便的转换方法
 `.ckpt`
 文件是使用
 [SD 到扩散器](https://huggingface.co/spaces/diffusers/sd-to-diffusers)
 空间。您可以按照空间上的说明进行转换
 `.ckpt`
 文件。


这种方法对于基本模型效果很好，但对于更定制的模型可能会遇到困难。如果它返回空的拉取请求或错误，您就会知道空间失败。在这种情况下，您可以尝试转换
 `.ckpt`
 带有脚本的文件。


### 


 用脚本转换


🤗 扩散器提供了
 [转换脚本](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py)
 用于转换
 `.ckpt`
 文件。这种方法比上面的Space更可靠。


在开始之前，请确保您有 🤗 Diffusers 的本地克隆来运行脚本并登录到您的 Hugging Face 帐户，以便您可以打开拉取请求并将转换后的模型推送到 Hub。



```
huggingface-cli login
```


要使用该脚本：


1. Git 克隆包含以下内容的存储库
 `.ckpt`
 您要转换的文件。对于这个例子，我们将其转换为
 [TemporalNet](https://huggingface.co/CiaraRowles/TemporalNet)
`.ckpt`
 文件：



```
git lfs install
git clone https://huggingface.co/CiaraRowles/TemporalNet
```


2. 在要从中转换检查点的存储库上打开拉取请求：



```
cd TemporalNet && git fetch origin refs/pr/13:pr/13
git checkout pr/13
```


3. 转换脚本中有几个需要配置的输入参数，但最重要的是：


* `检查点路径`
: 的路径
`.ckpt`
要转换的文件。
* `原始配置文件`
：定义原始架构配置的 YAML 文件。如果找不到此文件，请尝试在您找到该文件的 GitHub 存储库中搜索 YAML 文件
`.ckpt`
文件。
* `转储路径`
：转换后模型的路径。




例如，您可以采取
`cldm_v15.yaml`
文件来自
[ControlNet](https://github.com/lllyasviel/ControlNet/tree/main/models)
因为 TemporalNet 模型是稳定扩散 v1.5 和 ControlNet 模型。
4. 现在您可以运行脚本来转换
 `.ckpt`
 文件：



```
python ../diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path temporalnetv3.ckpt --original_config_file cldm_v15.yaml --dump_path ./ --controlnet
```


5. 转换完成后，上传转换后的模型并测试结果
 [拉取请求](https://huggingface.co/CiaraRowles/TemporalNet/discussions/13)
 ！



```
git push origin pr/13:refs/pr/13
```


## Keras.p​​b 或.h5



🧪 这是一个实验性功能。目前 Convert KerasCV Space 仅支持稳定扩散 v1 检查点。


[KerasCV](https://keras.io/keras_cv/)
 支持培训
 [稳定扩散](https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/stable_diffusion)
 v1 和 v2。然而，它为试验稳定扩散模型进行推理和部署提供了有限的支持，而 🤗 Diffusers 为此目的提供了一套更完整的功能，例如不同的功能
 [噪声调度程序](https://huggingface.co/docs/diffusers/using-diffusers/schedulers)
 ,
 [闪光关注](https://huggingface.co/docs/diffusers/optimization/xformers)
 ， 和
 [其他
优化技术](https://huggingface.co/docs/diffusers/optimization/fp16)
 。


这
 [转换 KerasCV](https://huggingface.co/spaces/sayakpaul/convert-kerascv-sd-diffusers)
 空间转换
 `.pb`
 或者
 `.h5`
 文件到 PyTorch，然后将它们包装在
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 所以它已经准备好进行推理了。转换后的检查点存储在 Hugging Face Hub 上的存储库中。


对于这个例子，我们将
 [`sayakpaul/textual-inversion-kerasio`](https://huggingface.co/sayakpaul/textual-inversion-kerasio/tree/main)
 使用文本反转训练的检查点。它使用特殊的令牌
 `<我的有趣的猫>`
 个性化与猫的图像。


转换 KerasCV Space 允许您输入以下内容：


* 你的拥抱标志。
* 下载 UNet 和文本编码器权重的路径。根据模型的训练方式，您不一定需要提供 UNet 和文本编码器的路径。例如，文本反转仅需要文本编码器的嵌入，而文本到图像模型仅需要 UNet 权重。
* 占位符标记仅适用于文本反转模型。
* 这
 `output_repo_prefix`
 是存储转换后的模型的存储库的名称。


点击
 **提交**
 按钮自动转换 KerasCV 检查点！检查点成功转换后，您将看到包含转换后的检查点的新存储库的链接。点击新存储库的链接，您将看到 Convert KerasCV Space 生成了一个带有推理小部件的模型卡，用于尝试转换后的模型。


如果您更喜欢使用代码运行推理，请单击
 **用于扩散器**
 模型卡右上角的按钮可复制并粘贴代码片段：



```
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "sayakpaul/textual-inversion-cat-kerascv\_sd\_diffusers\_pipeline", use_safetensors=True
)
```


然后，您可以生成如下图像：



```
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "sayakpaul/textual-inversion-cat-kerascv\_sd\_diffusers\_pipeline", use_safetensors=True
)
pipeline.to("cuda")

placeholder_token = "<my-funny-cat-token>"
prompt = f"two {placeholder\_token} getting married, photorealistic, high quality"
image = pipeline(prompt, num_inference_steps=50).images[0]
```


## A1111 LoRA 文件



[Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
 (A1111) 是一种流行的稳定扩散 Web UI，支持模型共享平台，例如
 [奇维泰](https://civitai.com/)
 。使用低秩适应 (LoRA) 技术训练的模型特别受欢迎，因为它们训练速度快，并且文件大小比完全微调的模型小得多。 🤗 Diffusers 支持加载 A1111 LoRA 检查点
 [load\_lora\_weights()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.load_lora_weights)
 :



```
from diffusers import StableDiffusionXLPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "Lykon/dreamshaper-xl-1-0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
```


从 Civitai 下载 LoRA 检查点；这个例子使用了
 [Blueprintify SD XL 1.0](https://civitai.com/models/150986/blueprintify-sd-xl-10)
 检查点，但请随意尝试任何 LoRA 检查点！



```
# uncomment to download the safetensor weights
#!wget https://civitai.com/api/download/models/168776 -O blueprintify.safetensors
```


使用以下命令将 LoRA 检查点加载到管道中
 [load\_lora\_weights()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.load_lora_weights)
 方法：



```
pipeline.load_lora_weights(".", weight_name="blueprintify.safetensors")
```


现在您可以使用管道来生成图像：



```
prompt = "bl3uprint, a highly detailed blueprint of the empire state building, explaining how to build all parts, many txt, blueprint grid backdrop"
negative_prompt = "lowres, cropped, worst quality, low quality, normal quality, artifacts, signature, watermark, username, blurry, more than one bridge, bad architecture"

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=torch.manual_seed(0),
).images[0]
image
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/blueprint-lora.png)