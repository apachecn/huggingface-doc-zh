# 文本倒装

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/text_inversion>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/text_inversion>


[文本倒置](https://arxiv.org/abs/2208.01618)
 是一种从少量示例图像中捕获新颖概念的技术。虽然该技术最初是通过
 [潜在扩散模型](https://github.com/CompVis/latent-diffusion)
 ，此后它已被应用于其他模型变体，例如
 [稳定扩散](https://huggingface.co/docs/diffusers/main/en/conceptual/stable_diffusion)
 。学习到的概念可用于更好地控制从文本到图像管道生成的图像。它在文本编码器的嵌入空间中学习新的“单词”，这些单词在文本提示中使用以生成个性化图像。


![文本反转示例](https://textual-inversion.github.io/static/images/editing/colorful_teapot.JPG)


只需使用 3-5 张图像，您就可以向模型教授新概念，例如用于生成个性化图像的稳定扩散
 [（图片来源）]（https://github.com/rinongal/textual_inversion）
 。
 

 本指南将向您展示如何训练
 [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 具有文本反转的模型。本指南中使用的所有文本反转训练脚本都可以找到
 [此处](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion)
 如果您有兴趣仔细了解事物的幕后工作原理。


有一个社区创建的训练有素的文本反转模型集合
 [稳定扩散文本反转概念库](https://huggingface.co/sd-concepts-library)
 很容易用于推断。随着时间的推移，随着更多概念的添加，这将有望成为有用的资源！


在开始之前，请确保安装了库的训练依赖项：



```
pip install diffusers accelerate transformers
```


设置完所有依赖项后，初始化一个
 [🤗加速](https://github.com/huggingface/accelerate/)
 环境：



```
accelerate config
```


要设置默认🤗加速环境而不选择任何配置：



```
accelerate config default
```


或者，如果您的环境不支持笔记本等交互式 shell，您可以使用：



```
from accelerate.utils import write_basic_config

write_basic_config()
```


最后，你尝试并
 [安装 xFormers](https://huggingface.co/docs/diffusers/main/en/training/optimization/xformers)
 通过 xFormers 内存高效关注来减少内存占用。安装 xFormers 后，添加
 `--enable_xformers_memory_efficient_attention`
 训练脚本的参数。 Flax 不支持 xFormers。


## 将模型上传到 Hub



如果要将模型存储在 Hub 上，请将以下参数添加到训练脚本中：



```
--push_to_hub
```


## 保存和加载检查点



在训练期间定期保存模型的检查点通常是一个好主意。这样，如果您的训练因任何原因中断，您可以从保存的检查点恢复训练。要保存检查点，请将以下参数传递给训练脚本，以将完整的训练状态保存在以下位置的子文件夹中：
 `输出目录`
 每 500 步：



```
--checkpointing_steps=500
```


要从保存的检查点恢复训练，请将以下参数传递给训练脚本以及您要从中恢复的特定检查点：



```
--resume_from_checkpoint="checkpoint-1500"
```


## 微调



对于您的训练数据集，请下载这些
 [猫玩具的图像](https://huggingface.co/datasets/diffusers/cat_toy_example)
 并将它们存储在一个目录中。要使用您自己的数据集，请查看
 [创建训练数据集](create_dataset)
 指导。



```
from huggingface_hub import snapshot_download

local_dir = "./cat"
snapshot_download(
    "diffusers/cat\_toy\_example", local_dir=local_dir, repo_type="dataset", ignore_patterns=".gitattributes"
)
```


指定
 `型号_名称`
 环境变量（集线器模型存储库 ID 或包含模型权重的目录的路径）并将其传递给
 [`pretrained_model_name_or_path`](https://huggingface.co/docs/diffusers/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained.pretrained_model_name_or_path)
 论证，以及
 `数据目录`
 环境变量到包含图像的目录的路径。


现在您可以启动
 [训练脚本](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py)
 。该脚本创建以下文件并将其保存到您的存储库中：
 `learned_embeds.bin`
 ,
 `token_identifier.txt`
 ， 和
 `type_of_concept.txt`
 。


💡 在一个 V100 GPU 上完整的训练运行大约需要 1 小时。当您等待培训完成时，请随时查看
 [文本倒置的工作原理](#how-it-works)
 如果您好奇的话，请在下面的部分中查看！


 火炬


隐藏 Pytorch 内容



```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./cat"

accelerate launch textual_inversion.py   --pretrained_model_name_or_path=$MODEL\_NAME   --train_data_dir=$DATA\_DIR   --learnable_property="object"   --placeholder_token="<cat-toy>" --initializer_token="toy"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=3000   --learning_rate=5.0e-04 --scale_lr   --lr_scheduler="constant"   --lr_warmup_steps=0   --output_dir="textual\_inversion\_cat"   --push_to_hub
```


💡如果你想增加可训练容量，你可以关联你的占位符token，
 *例如。*
`<猫玩具>`
 到
多个嵌入向量。这可以帮助模型更好地捕捉更多（复杂）图像的风格。
要训​​练多个嵌入向量，只需传递：



```
--num_vectors=5
```


.J{
 中风：#dce0df；
 }
.K {
 笔划线连接：圆形；
 }


贾克斯


隐藏 JAX 内容


如果您可以使用 TPU，请尝试
 [Flax训练脚本](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion_flax.py)
 训练速度更快（这也适用于 GPU）。在相同的配置设置下，Flax 训练脚本应该比 PyTorch 训练脚本至少快 70%！ ⚡️


在开始之前，请确保安装了 Flax 特定的依赖项：



```
pip install -U -r requirements_flax.txt
```


指定
 `型号_名称`
 环境变量（集线器模型存储库 ID 或包含模型权重的目录的路径）并将其传递给
 [`pretrained_model_name_or_path`](https://huggingface.co/docs/diffusers/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained.pretrained_model_name_or_path)
 争论。


然后你就可以启动
 [训练脚本](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion_flax.py)
 :



```
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export DATA_DIR="./cat"

python textual_inversion_flax.py   --pretrained_model_name_or_path=$MODEL\_NAME   --train_data_dir=$DATA\_DIR   --learnable_property="object"   --placeholder_token="<cat-toy>" --initializer_token="toy"   --resolution=512   --train_batch_size=1   --max_train_steps=3000   --learning_rate=5.0e-04 --scale_lr   --output_dir="textual\_inversion\_cat"   --push_to_hub
```


### 


 中级记录


如果您有兴趣跟踪模型训练进度，可以保存训练过程中生成的图像。将以下参数添加到训练脚本以启用中间日志记录：


* `验证提示`
 ，用于生成样本的提示（设置为
 `无`
 默认情况下，中间日志记录被禁用）
* `num_validation_images`
 ，要生成的样本图像的数量
* `验证步骤`
 , 生成前的步数
 `num_validation_images`
 来自
 `验证提示`



```
--validation_prompt="A <cat-toy> backpack"
--num_validation_images=4
--validation_steps=100
```


## 推理



训练完模型后，您可以使用它进行推理
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 。


默认情况下，文本反演脚本将仅保存具有以下特征的文本反演嵌入向量：
被添加到文本编码器嵌入矩阵中并因此被训练。


火炬


隐藏 Pytorch 内容


 💡 社区创建了一个包含不同文本反转嵌入向量的大型库，称为
 [sd-concepts-library](https://huggingface.co/sd-concepts-library)
 。
您无需从头开始训练文本反演嵌入，您还可以查看是否已将合适的文本反演嵌入添加到库中。


要加载文本反转嵌入，您首先需要加载训练时使用的基本模型
您的文本反转嵌入向量。这里我们假设
 [`runwayml/stable-diffusion-v1-5`](runwayml/stable-diffusion-v1-5)
 被用作基础模型，所以我们首先加载它：



```
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
```


接下来，我们需要加载文本反转嵌入向量，这可以通过
 `TextualInversionLoaderMixin.load_textual_inversion`
 功能。在这里，我们将加载之前“<cat-toy>”示例的嵌入。



```
pipe.load_textual_inversion("sd-concepts-library/cat-toy")
```


现在我们可以运行管道，确保占位符标记
 `<猫玩具>`
 在我们的提示中使用。



```
prompt = "A <cat-toy> backpack"

image = pipe(prompt, num_inference_steps=50).images[0]
image.save("cat-backpack.png")
```


功能
 `TextualInversionLoaderMixin.load_textual_inversion`
 不仅可以
加载以 Diffusers 格式保存的文本嵌入向量，以及嵌入向量
保存在
 [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
 格式。
为此，您可以首先从以下位置下载嵌入向量
 [civitAI](https://civitai.com/models/3036?modelVersionId=8387)
 然后在本地加载：



```
pipe.load_textual_inversion("./charturnerv2.pt")
```


.J{
 中风：#dce0df；
 }
.K {
 笔划线连接：圆形；
 }


贾克斯


隐藏 JAX 内容


目前没有
 `加载文本反转`
 Flax 的函数，因此必须确保文本反转
训练后嵌入向量保存为模型的一部分。


然后，该模型可以像任何其他 Flax 模型一样运行：



```
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline

model_path = "path-to-your-trained-model"
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(model_path, dtype=jax.numpy.bfloat16)

prompt = "A <cat-toy> backpack"
prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
image.save("cat-backpack.png")
```


## 怎么运行的



![论文中的图表显示概述](https://textual-inversion.github.io/static/images/training/training.JPG)


文本反转的架构概述
 [博客文章。](https://textual-inversion.github.io/)


 通常，文本提示在传递给模型（通常是转换器）之前会被标记为嵌入。文本反转做了类似的事情，但它学习了新的标记嵌入，
 `v*`
 ，来自一个特殊的令牌
 `S*`
 在上图中。模型输出用于调节扩散模型，这有助于扩散模型从几个示例图像中理解提示和新概念。


为此，文本反转使用生成器模型和训练图像的噪声版本。生成器尝试预测图像的噪声较小的版本，以及令牌嵌入
 `v*`
 根据生成器的性能进行优化。如果令牌嵌入成功地捕获了新概念，它就会为扩散模型提供更多有用的信息，并有助于创建噪声更少的更清晰的图像。此优化过程通常在接触各种提示和图像变体数千个步骤后发生。