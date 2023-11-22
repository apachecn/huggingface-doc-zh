# 文本转图像

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/text2image>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/text2image>


文本到图像的微调脚本是实验性的。很容易过度适应并遇到灾难性遗忘等问题。我们建议您探索不同的超参数，以便在数据集上获得最佳结果。


稳定扩散等文本到图像模型根据文本提示生成图像。本指南将向您展示如何微调
 [`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4)
 使用 PyTorch 和 Flax 在您自己的数据集上建模。本指南中使用的所有文本到图像微调的训练脚本都可以在此找到
 [存储库](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)
 如果您有兴趣仔细看看。


在运行脚本之前，请确保安装库的训练依赖项：



```
pip install git+https://github.com/huggingface/diffusers.git
pip install -U -r requirements.txt
```


并初始化一个
 [🤗 加速](https://github.com/huggingface/accelerate/)
 环境：



```
accelerate config
```


如果您已经克隆了存储库，则无需执行这些步骤。相反，您可以将本地结账的路径传递给训练脚本，它将从那里加载。


## 硬件要求



使用
 `梯度检查点`
 和
 `混合精度`
 ，应该可以在单个 24GB GPU 上微调模型。对于更高
 `批量大小`
 和更快的训练，最好使用具有 30GB 以上 GPU 内存的 GPU。您还可以使用 JAX/Flax 在 TPU 或 GPU 上进行微调，这将在后面介绍
 [下](#flax-jax-finetuning)
 。


通过使用 xFormers 启用内存高效关注，您可以进一步减少内存占用。确保你有
 [已安装 xFormers](./optimization/xformers)
 并通过
 `--enable_xformers_memory_efficient_attention`
 标记到训练脚本。


xFormers 不适用于 Flax。


## 将模型上传到 Hub



通过将以下参数添加到训练脚本中，将模型存储在 Hub 上：



```
  --push_to_hub
```


## 保存和加载检查点



定期保存检查点是一个好主意，以防训练期间发生任何情况。要保存检查点，请将以下参数传递给训练脚本：



```
  --checkpointing_steps=500
```


每 500 步，完整的训练状态就会保存在
 `输出目录`
 。检查点的格式为
 `检查点-`
 接下来是到目前为止训练的步数。例如，
 `检查点-1500`
 是 1500 个训练步骤后保存的检查点。


要加载检查点以恢复训练，请传递参数
 `--resume_from_checkpoint`
 到训练脚本并指定要从中恢复的检查点。例如，以下参数从 1500 个训练步骤后保存的检查点恢复训练：



```
  --resume_from_checkpoint="checkpoint-1500"
```


## 微调



火炬


隐藏 Pytorch 内容


启动
 [PyTorch 训练脚本](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)
 进行微调运行
 [神奇宝贝 BLIP 字幕](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)
 像这样的数据集。


指定
 `型号_名称`
 环境变量（集线器模型存储库 ID 或包含模型权重的目录的路径）并将其传递给
 [`pretrained_model_name_or_path`](https://huggingface.co/docs/diffusers/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained.pretrained_model_name_or_path)
 争论。



```
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="lambdalabs/pokemon-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image.py   --pretrained_model_name_or_path=$MODEL\_NAME   --dataset_name=$dataset\_name   --use_ema   --resolution=512 --center_crop --random_flip   --train_batch_size=1   --gradient_accumulation_steps=4   --gradient_checkpointing   --max_train_steps=15000   --learning_rate=1e-05   --max_grad_norm=1   --lr_scheduler="constant" --lr_warmup_steps=0   --output_dir="sd-pokemon-model"   --push_to_hub
```


要微调您自己的数据集，请根据🤗所需的格式准备数据集
 [数据集](https://huggingface.co/docs/datasets/index)
 。你可以
 [将数据集上传到中心](https://huggingface.co/docs/datasets/image_dataset#upload-dataset-to-the-hub)
 ，或者你可以
 [准备一个包含文件的本地文件夹](https://huggingface.co/docs/datasets/image_dataset#imagefolder)
 。


如果您想使用自定义加载逻辑，请修改脚本。我们在代码中的适当位置留下了指针来帮助您。 🤗 下面的示例脚本展示了如何在本地数据集上进行微调
 `TRAIN_DIR`
 以及模型的保存位置
 `输出目录`
 ：



```
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="path\_to\_your\_dataset"
export OUTPUT_DIR="path\_to\_save\_model"

accelerate launch train_text_to_image.py   --pretrained_model_name_or_path=$MODEL\_NAME   --train_data_dir=$TRAIN\_DIR   --use_ema   --resolution=512 --center_crop --random_flip   --train_batch_size=1   --gradient_accumulation_steps=4   --gradient_checkpointing   --mixed_precision="fp16"   --max_train_steps=15000   --learning_rate=1e-05   --max_grad_norm=1   --lr_scheduler="constant" 
  --lr_warmup_steps=0   --output_dir=${OUTPUT\_DIR}   --push_to_hub
```


#### 


 使用多个 GPU 进行训练


‘加速’
 允许无缝的多 GPU 训练。按照说明操作
 [此处](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
 用于运行分布式训练
 ‘加速’
 。这是一个示例命令：



```
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="lambdalabs/pokemon-blip-captions"

accelerate launch --mixed_precision="fp16" --multi_gpu  train_text_to_image.py   --pretrained_model_name_or_path=$MODEL\_NAME   --dataset_name=$dataset\_name   --use_ema   --resolution=512 --center_crop --random_flip   --train_batch_size=1   --gradient_accumulation_steps=4   --gradient_checkpointing   --max_train_steps=15000 \ 
  --learning_rate=1e-05   --max_grad_norm=1   --lr_scheduler="constant"   --lr_warmup_steps=0   --output_dir="sd-pokemon-model"   --push_to_hub
```


.J{
 中风：#dce0df；
 }
.K {
 笔划线连接：圆形；
 }


贾克斯


隐藏 JAX 内容


借助 Flax，可以在 TPU 和 GPU 上更快地训练稳定扩散模型，这要归功于
 [@duongna211](https://github.com/duongna21)
 。这在 TPU 硬件上非常高效，但在 GPU 上也很有效。 Flax 训练脚本尚不支持梯度检查点或梯度累积等功能，因此您需要具有至少 30GB 内存的 GPU 或 TPU v3。


在运行脚本之前，请确保您已安装以下要求：



```
pip install -U -r requirements_flax.txt
```


指定
 `型号_名称`
 环境变量（集线器模型存储库 ID 或包含模型权重的目录的路径）并将其传递给
 [`pretrained_model_name_or_path`](https://huggingface.co/docs/diffusers/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained.pretrained_model_name_or_path)
 争论。


现在您可以启动
 [Flax训练脚本](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_flax.py)
 像这样：



```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export dataset_name="lambdalabs/pokemon-blip-captions"

python train_text_to_image_flax.py   --pretrained_model_name_or_path=$MODEL\_NAME   --dataset_name=$dataset\_name   --resolution=512 --center_crop --random_flip   --train_batch_size=1   --max_train_steps=15000   --learning_rate=1e-05   --max_grad_norm=1   --output_dir="sd-pokemon-model"   --push_to_hub
```


要微调您自己的数据集，请根据🤗所需的格式准备数据集
 [数据集](https://huggingface.co/docs/datasets/index)
 。你可以
 [将数据集上传到中心](https://huggingface.co/docs/datasets/image_dataset#upload-dataset-to-the-hub)
 ，或者你可以
 [准备一个包含文件的本地文件夹](https://huggingface.co/docs/datasets/image_dataset#imagefolder)
 。


如果您想使用自定义加载逻辑，请修改脚本。我们在代码中的适当位置留下了指针来帮助您。 🤗 下面的示例脚本展示了如何在本地数据集上进行微调
 `TRAIN_DIR`
 ：



```
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export TRAIN_DIR="path\_to\_your\_dataset"

python train_text_to_image_flax.py   --pretrained_model_name_or_path=$MODEL\_NAME   --train_data_dir=$TRAIN\_DIR   --resolution=512 --center_crop --random_flip   --train_batch_size=1   --mixed_precision="fp16"   --max_train_steps=15000   --learning_rate=1e-05   --max_grad_norm=1   --output_dir="sd-pokemon-model"   --push_to_hub
```


## 使用最小 SNR 加权进行训练



我们支持使用 Min-SNR 加权策略进行训练
 [通过最小SNR加权策略进行高效扩散训练](https://arxiv.org/abs/2303.09556)
 这有助于实现更快的收敛
通过重新平衡损失。为了使用它，需要设置
 `--snr_gamma`
 争论。推荐的
使用时的值为5.0。


你可以找到
 [这个关于权重和偏差的项目](https://wandb.ai/sayakpaul/text2image-finetune-minsnr)
 比较以下设置的损耗面：


* 没有Min-SNR加权策略的训练
* 使用 Min-SNR 加权策略进行训练（
 `snr_gamma`
 设置为 5.0)
* 使用 Min-SNR 加权策略进行训练（
 `snr_gamma`
 设置为 1.0)


对于我们的小型 Pokemons 数据集，Min-SNR 加权策略的效果可能看起来并不明显，但对于较大的数据集，我们相信效果会更加明显。


另请注意，在本例中，我们要么预测
 `epsilon`
 （即噪音）或
 `v_预测`
 。对于这两种情况，我们使用的 Min-SNR 加权策略的公式都成立。


仅 PyTorch 支持使用 Min-SNR 加权策略进行训练。


## 洛拉



您还可以使用大型语言模型的低秩适应 (LoRA)，这是一种用于加速训练大型模型的微调技术，用于微调文本到图像模型。欲了解更多详细信息，请查看
 [LoRA训练](lora#text-to-image)
 指导。


## 推理



现在，您可以通过将 Hub 上的模型路径或模型名称传递到
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 ：


火炬


隐藏 Pytorch 内容



```
from diffusers import StableDiffusionPipeline

model_path = "path\_to\_saved\_model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True)
pipe.to("cuda")

image = pipe(prompt="yoda").images[0]
image.save("yoda-pokemon.png")
```


.J{
 中风：#dce0df；
 }
.K {
 笔划线连接：圆形；
 }


贾克斯


隐藏 JAX 内容



```
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline

model_path = "path\_to\_saved\_model"
pipe, params = FlaxStableDiffusionPipeline.from_pretrained(model_path, dtype=jax.numpy.bfloat16)

prompt = "yoda pokemon"
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
image.save("yoda-pokemon.png")
```


## 稳定扩散XL



* 我们支持对出厂的UNet进行微调
 [稳定扩散XL](https://huggingface.co/papers/2307.01952)
 通过
 `train_text_to_image_sdxl.py`
 脚本。请参考文档
 [此处](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/README_sdxl.md)
 。
* 我们还支持对 UNet 和文本编码器进行微调
 [稳定扩散XL](https://huggingface.co/papers/2307.01952)
 通过 LoRA
 `train_text_to_image_lora_sdxl.py`
 脚本。请参考文档
 [此处](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/README_sdxl.md)
 。


## 康定斯基2.2



* 我们支持对 Kandinsky2.2 中的解码器和先验进行微调
 `train_text_to_image_prior.py`
 和
 `train_text_to_image_decoder.py`
 脚本。还包括 LoRA 支持。请参考文档
 [此处](https://github.com/huggingface/diffusers/blob/main/examples/kandinsky2_2/text_to_image/README_sdxl.md)
 。