# 梦想展位

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/dreambooth>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/dreambooth>


[DreamBooth](https://arxiv.org/abs/2208.12242)
 是一种个性化文本到图像模型的方法，例如仅给定主题的几张（3-5）图像的稳定扩散。它允许模型在不同场景、姿势和视图中生成主体的上下文图像。


![项目博客中的 Dreamambooth 示例](https://dreambooth.github.io/DreamBooth_files/teaser_static.jpg)


Dreambooth 示例来自
 [项目的博客。](https://dreambooth.github.io)


 本指南将向您展示如何使用 DreamBooth 进行微调
 [`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4)
 适用于各种 GPU 大小和 Flax 的模型。本指南中使用的所有 DreamBooth 培训脚本均可在此处找到
 [此处](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)
 如果您有兴趣深入挖掘并了解事情是如何运作的。


在运行脚本之前，请确保安装库的训练依赖项。我们还建议安装 🧨 扩散器
 `主要`
 GitHub 分支：



```
pip install git+https://github.com/huggingface/diffusers
pip install -U -r diffusers/examples/dreambooth/requirements.txt
```


xFormers 不是培训要求的一部分，但我们建议您
 [安装](../optimization/xformers)
 如果可以的话，因为它可以让你的训练更快并且更少的内存消耗。


设置完所有依赖项后，初始化一个
 [🤗 加速](https://github.com/huggingface/accelerate/)
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


最后，下载一个
 [几张狗的图片](https://huggingface.co/datasets/diffusers/dog-example)
 前往 DreamBooth：



```
from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir,
    repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```


要使用您自己的数据集，请查看
 [创建训练数据集](create_dataset)
 指导。


## 微调



DreamBooth微调对超参数非常敏感，容易过拟合。我们建议您看看我们的
 [深入分析](https://huggingface.co/blog/dreambooth)
 具有针对不同主题的推荐设置，帮助您选择合适的超参数。


 torch


隐藏 Pytorch 内容


设置
 `INSTANCE_DIR`
 环境变量到包含狗图像的目录路径。


指定
 `型号_名称`
 环境变量（集线器模型存储库 ID 或包含模型权重的目录的路径）并将其传递给
 `预训练模型名称或路径`
 争论。这
 `实例提示`
 参数是包含唯一标识符的文本提示，例如
 `sks`
 ，以及图像所属的类，在本例中是
 “一张 sks 狗的照片”
 。



```
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="path\_to\_saved\_model"
```


然后您可以启动训练脚本（您可以找到完整的训练脚本
 [此处](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)
 ）使用以下命令：



```
accelerate launch train_dreambooth.py   --pretrained_model_name_or_path=$MODEL\_NAME    --instance_data_dir=$INSTANCE\_DIR   --output_dir=$OUTPUT\_DIR   --instance_prompt="a photo of sks dog"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=1   --learning_rate=5e-6   --lr_scheduler="constant"   --lr_warmup_steps=0   --max_train_steps=400   --push_to_hub
```


.J{
 中风：#dce0df；
 }
.K {
 笔划线连接：圆形；
 }


贾克斯


隐藏 JAX 内容


如果您可以使用 TPU 或想要更快地训练，您可以尝试
 [Flax训练脚本](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_flax.py)
 。 Flax 训练脚本不支持梯度检查点或梯度累积，因此您需要具有至少 30GB 内存的 GPU。


在运行脚本之前，请确保您已安装以下要求：



```
pip install -U -r requirements.txt
```


指定
 `型号_名称`
 环境变量（集线器模型存储库 ID 或包含模型权重的目录的路径）并将其传递给
 `预训练模型名称或路径`
 争论。这
 `实例提示`
 参数是包含唯一标识符的文本提示，例如
 `sks`
 ，以及图像所属的类，在本例中是
 “一张 sks 狗的照片”
 。


现在您可以使用以下命令启动训练脚本：



```
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py   --pretrained_model_name_or_path=$MODEL\_NAME    --instance_data_dir=$INSTANCE\_DIR   --output_dir=$OUTPUT\_DIR   --instance_prompt="a photo of sks dog"   --resolution=512   --train_batch_size=1   --learning_rate=5e-6   --max_train_steps=400   --push_to_hub
```


## 通过保留先验损失进行微调



预先保存用于避免过度拟合和语言漂移（查看
 [论文](https://arxiv.org/abs/2208.12242)
 如果您有兴趣，可以了解更多信息）。为了预先保存，您可以使用同一类的其他图像作为训练过程的一部分。好处是您可以使用稳定扩散模型本身生成这些图像！训练脚本会将生成的图像保存到您指定的本地路径。


作者建议生成
 `num_epochs * num_samples`
 预先保存的图像。在大多数情况下，200-300 张图像效果很好。


torch


隐藏 Pytorch 内容



```
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./dog"
export CLASS_DIR="path\_to\_class\_images"
export OUTPUT_DIR="path\_to\_saved\_model"

accelerate launch train_dreambooth.py   --pretrained_model_name_or_path=$MODEL\_NAME    --instance_data_dir=$INSTANCE\_DIR   --class_data_dir=$CLASS\_DIR   --output_dir=$OUTPUT\_DIR   --with_prior_preservation --prior_loss_weight=1.0   --instance_prompt="a photo of sks dog"   --class_prompt="a photo of dog"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=1   --learning_rate=5e-6   --lr_scheduler="constant"   --lr_warmup_steps=0   --num_class_images=200   --max_train_steps=800   --push_to_hub
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
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="./dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py   --pretrained_model_name_or_path=$MODEL\_NAME    --instance_data_dir=$INSTANCE\_DIR   --class_data_dir=$CLASS\_DIR   --output_dir=$OUTPUT\_DIR   --with_prior_preservation --prior_loss_weight=1.0   --instance_prompt="a photo of sks dog"   --class_prompt="a photo of dog"   --resolution=512   --train_batch_size=1   --learning_rate=5e-6   --num_class_images=200   --max_train_steps=800   --push_to_hub
```


## 微调文本编码器和 UNet



该脚本还允许您微调
 `文本编码器`
 随着
 `乌内特`
 。在我们的实验中（查看
 [使用 🧨 扩散器通过 DreamBooth 训练稳定扩散](https://huggingface.co/blog/dreambooth)
 发布更多详细信息），这会产生更好的结果，特别是在生成面部图像时。


训练文本编码器需要额外的内存，并且不适合 16GB GPU。您需要至少 24GB VRAM 才能使用此选项。


通过
 `--train_text_encoder`
 训练脚本的参数以启用微调
 `文本编码器`
 和
 `乌内特`
 ：


torch


隐藏 Pytorch 内容



```
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./dog"
export CLASS_DIR="path\_to\_class\_images"
export OUTPUT_DIR="path\_to\_saved\_model"

accelerate launch train_dreambooth.py   --pretrained_model_name_or_path=$MODEL\_NAME    --train_text_encoder   --instance_data_dir=$INSTANCE\_DIR   --class_data_dir=$CLASS\_DIR   --output_dir=$OUTPUT\_DIR   --with_prior_preservation --prior_loss_weight=1.0   --instance_prompt="a photo of sks dog"   --class_prompt="a photo of dog"   --resolution=512   --train_batch_size=1   --use_8bit_adam   --gradient_checkpointing   --learning_rate=2e-6   --lr_scheduler="constant"   --lr_warmup_steps=0   --num_class_images=200   --max_train_steps=800   --push_to_hub
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
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="./dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py   --pretrained_model_name_or_path=$MODEL\_NAME    --train_text_encoder   --instance_data_dir=$INSTANCE\_DIR   --class_data_dir=$CLASS\_DIR   --output_dir=$OUTPUT\_DIR   --with_prior_preservation --prior_loss_weight=1.0   --instance_prompt="a photo of sks dog"   --class_prompt="a photo of dog"   --resolution=512   --train_batch_size=1   --learning_rate=2e-6   --num_class_images=200   --max_train_steps=800   --push_to_hub
```


## 使用 LoRA 进行微调



您还可以在 DreamBooth 上使用大型语言模型的低秩适应 (LoRA)，这是一种用于加速训练大型模型的微调技术。欲了解更多详细信息，请查看
 [LoRA 培训](./lora#dreambooth)
 指导。


## 训练时保存检查点



使用 Dreambooth 进行训练时很容易出现过拟合，因此有时在训练过程中保存定期检查点很有用。中间检查点之一实际上可能比最终模型效果更好！将以下参数传递给训练脚本以启用保存检查点：



```
  --checkpointing_steps=500
```


这会将完整的训练状态保存在您的子文件夹中
 `输出目录`
 。子文件夹名称以前缀开头
 `检查点-`
 ，后面是迄今为止执行的步骤数；例如，
 `检查点-1500`
 将是 1500 个训练步骤后保存的检查点。


### 


 从保存的检查点恢复训练


如果你想从任何保存的检查点恢复训练，你可以传递参数
 `--resume_from_checkpoint`
 到脚本并指定要使用的检查点的名称。您还可以使用特殊字符串
 `“最新”`
 从最后保存的检查点（步数最多的检查点）恢复。例如，以下将从 1500 步后保存的检查点恢复训练：



```
  --resume_from_checkpoint="checkpoint-1500"
```


如果您愿意，这是调整一些超参数的好机会。


### 


 从保存的检查点推断


保存的检查点以适合恢复训练的格式存储。它们不仅包括模型权重，还包括优化器、数据加载器和学习率的状态。


如果你有
 **`“加速>=0.16.0”`**
 安装完毕，使用以下代码运行
从中间检查点推断。



```
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = "CompVis/stable-diffusion-v1-4"

unet = UNet2DConditionModel.from_pretrained("/sddata/dreambooth/daruma-v2-1/checkpoint-100/unet")

# if you have trained with `--args.train\_text\_encoder` make sure to also load the text encoder
text_encoder = CLIPTextModel.from_pretrained("/sddata/dreambooth/daruma-v2-1/checkpoint-100/text\_encoder")

pipeline = DiffusionPipeline.from_pretrained(
    model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16, use_safetensors=True
)
pipeline.to("cuda")

# Perform inference, or save, or push to the hub
pipeline.save_pretrained("dreambooth-pipeline")
```


如果你有
 **`“加速<0.16.0”`**
 安装完毕后，需要先将其转换为推理管道：



```
from accelerate import Accelerator
from diffusers import DiffusionPipeline

# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = DiffusionPipeline.from_pretrained(model_id, use_safetensors=True)

accelerator = Accelerator()

# Use text\_encoder if `--train\_text\_encoder` was used for the initial training
unet, text_encoder = accelerator.prepare(pipeline.unet, pipeline.text_encoder)

# Restore state from a checkpoint path. You have to use the absolute path here.
accelerator.load_state("/sddata/dreambooth/daruma-v2-1/checkpoint-100")

# Rebuild the pipeline with the unwrapped models (assignment to .unet and .text\_encoder should work too)
pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    unet=accelerator.unwrap_model(unet),
    text_encoder=accelerator.unwrap_model(text_encoder),
    use_safetensors=True,
)

# Perform inference, or save, or push to the hub
pipeline.save_pretrained("dreambooth-pipeline")
```


## 针对不同 GPU 大小的优化



根据您的硬件，有几种不同的方法可以在 GPU 上优化 DreamBooth（从 16GB 到 8GB）！


### 


 xFormers


[xFormers](https://github.com/facebookresearch/xformers)
 是一个用于优化 Transformers 的工具箱，它包括
 [内存效率注意力](https://facebookresearch.github.io/xformers/components/ops.html#module-xformers.ops)
 🧨 扩散器中使用的机制。你需要
 [安装 xFormers](./optimization/xformers)
 然后将以下参数添加到您的训练脚本中：



```
  --enable_xformers_memory_efficient_attention
```


xFormers 在 Flax 中不可用。


### 


 将渐变设置为无


降低内存占用的另一种方法是
 [设置梯度](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html)
 到
 `无`
 而不是零。但是，这可能会改变某些行为，因此如果您遇到任何问题，请尝试删除此参数。将以下参数添加到您的训练脚本中以将梯度设置为
 `无`
 ：



```
  --set_grads_to_none
```


### 


 16GB GPU


借助梯度检查点和
 [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
 8 位优化器，可以在 16GB GPU 上训练 DreamBooth。确保你已经安装了bitsandbytes：



```
pip install bitsandbytes
```


然后通过
 `--use_8bit_adam`
 训练脚本的选项：



```
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./dog"
export CLASS_DIR="path\_to\_class\_images"
export OUTPUT_DIR="path\_to\_saved\_model"

accelerate launch train_dreambooth.py   --pretrained_model_name_or_path=$MODEL\_NAME    --instance_data_dir=$INSTANCE\_DIR   --class_data_dir=$CLASS\_DIR   --output_dir=$OUTPUT\_DIR   --with_prior_preservation --prior_loss_weight=1.0   --instance_prompt="a photo of sks dog"   --class_prompt="a photo of dog"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=2 --gradient_checkpointing   --use_8bit_adam   --learning_rate=5e-6   --lr_scheduler="constant"   --lr_warmup_steps=0   --num_class_images=200   --max_train_steps=800   --push_to_hub
```


### 


 12GB GPU


要在 12GB GPU 上运行 DreamBooth，您需要启用梯度检查点、8 位优化器、xFormers，并将梯度设置为
 `无`
 ：



```
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py   --pretrained_model_name_or_path=$MODEL\_NAME    --instance_data_dir=$INSTANCE\_DIR   --class_data_dir=$CLASS\_DIR   --output_dir=$OUTPUT\_DIR   --with_prior_preservation --prior_loss_weight=1.0   --instance_prompt="a photo of sks dog"   --class_prompt="a photo of dog"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=1 --gradient_checkpointing   --use_8bit_adam   --enable_xformers_memory_efficient_attention   --set_grads_to_none   --learning_rate=2e-6   --lr_scheduler="constant"   --lr_warmup_steps=0   --num_class_images=200   --max_train_steps=800   --push_to_hub
```


### 


 8GB GPU


对于 8GB GPU，您需要以下帮助
 [DeepSpeed](https://www.deepspeed.ai/)
 卸载一些
张量从 VRAM 传输到 CPU 或 NVME，从而可以使用更少的 GPU 内存进行训练。


运行以下命令来配置您的 🤗 Accelerate 环境：



```
accelerate config
```


在配置过程中，确认您要使用 DeepSpeed。现在，可以通过结合 DeepSpeed stage 2、fp16 混合精度，并将模型参数和优化器状态卸载到 CPU，在 8GB VRAM 以下进行训练。缺点是这需要更多的系统 RAM，大约 25 GB。看
 [DeepSpeed 文档](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
 了解更多配置选项。


您还应该将默认的 Adam 优化器更改为 DeepSpeed 的 Adam 优化版本
 [`deepspeed.ops.adam.DeepSpeedCPUAdam`](https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu)
 大幅加速。启用
 `DeepSpeedCPUAdam`
 要求您系统的 CUDA 工具链版本与随 PyTorch 安装的版本相同。


8 位优化器目前似乎与 DeepSpeed 不兼容。


使用以下命令启动训练：



```
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./dog"
export CLASS_DIR="path\_to\_class\_images"
export OUTPUT_DIR="path\_to\_saved\_model"

accelerate launch train_dreambooth.py   --pretrained_model_name_or_path=$MODEL\_NAME   --instance_data_dir=$INSTANCE\_DIR   --class_data_dir=$CLASS\_DIR   --output_dir=$OUTPUT\_DIR   --with_prior_preservation --prior_loss_weight=1.0   --instance_prompt="a photo of sks dog"   --class_prompt="a photo of dog"   --resolution=512   --train_batch_size=1   --sample_batch_size=1   --gradient_accumulation_steps=1 --gradient_checkpointing   --learning_rate=5e-6   --lr_scheduler="constant"   --lr_warmup_steps=0   --num_class_images=200   --max_train_steps=800   --mixed_precision=fp16   --push_to_hub
```


## 推理



训练完模型后，指定模型的保存路径，并在模型中使用它进行推理
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 。确保您的提示包含特殊内容
 `标识符`
 训练期间使用（
 `sks`
 在前面的例子中）。


如果你有
 **`“加速>=0.16.0”`**
 安装完毕后，可以使用以下代码运行
从中间检查点推断：



```
from diffusers import DiffusionPipeline
import torch

model_id = "path\_to\_saved\_model"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True).to("cuda")

prompt = "A photo of sks dog in a bucket"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("dog-bucket.png")
```


您还可以从任何一个进行推断
 [保存的训练检查点](#inference-from-a-saved-checkpoint)
 。


## 如果



您可以使用 lora 和 full dreambooth 脚本将文本训练为图像
 [IF模型](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)
 和第二阶段升级器
 [IF模型](https://huggingface.co/DeepFloyd/IF-II-L-v1.0)
 。


请注意，IF 具有预测方差，而我们的微调脚本仅训练模型预测误差，因此对于微调 IF 模型，我们切换到固定的
差异表。完整的微调脚本将更新完整保存模型的调度程序配置。然而，当加载保存的 LoRA 权重时，您
还必须更新管道的调度程序配置。



```
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", use_safetensors=True)

pipe.load_lora_weights("<lora weights path>")

# Update scheduler config to fixed variance schedule
pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, variance_type="fixed\_small")
```


此外，IF 还需要一些替代的 cli 标志。


`--分辨率=64`
 ：IF是像素空间扩散模型。为了对未压缩的像素进行操作，输入图像的分辨率要小得多。


`--pre_compute_text_embeddings`
 : 如果使用
 [T5](https://huggingface.co/docs/transformers/model_doc/t5)
 对于它的文本编码器。为了节省 GPU 内存，我们预先计算所有文本嵌入，然后取消分配
T5。


`--tokenizer_max_length=77`
 ：T5 的默认文本长度较长，但默认 IF 编码过程使用较小的数字。


`--text_encoder_use_attention_mask`
 ：T5 将注意力掩码传递给文本编码器。


### 


 技巧和窍门


我们发现 LoRA 足以微调第一阶段模型，因为模型的低分辨率使得表示细粒度细节变得困难。


对于常见和/或视觉上不复杂的对象概念，您可以不微调升级器。请务必调整传递给的提示
upscaler 从实例提示中删除新令牌。 IE。如果您的第一阶段提示是“一只 sks 狗”，请使用“一只狗”作为您的第二阶段提示。


对于原始训练集中不存在的细粒度细节（例如人脸），我们发现第二阶段升级器的全面微调优于
LoRA 微调阶段 II。


对于像人脸这样的细粒度细节，我们发现较低的学习率和较大的批量大小效果最好。


对于第二阶段，我们发现还需要较低的学习率。


我们通过实验发现，具有默认较多去噪步骤的 DDPM 调度程序有时比 DPM Solver 调度程序工作得更好
在训练脚本中使用。


### 


 第二阶段附加验证图像


第二阶段验证需要对图像进行放大，我们可以下载训练集的缩小版本：



```
from huggingface_hub import snapshot_download

local_dir = "./dog\_downsized"
snapshot_download(
    "diffusers/dog-example-downsized",
    local_dir=local_dir,
    repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```


### 


 IF 第一阶段 LoRA Dreambooth


此训练配置需要 ~28 GB VRAM。



```
export MODEL_NAME="DeepFloyd/IF-I-XL-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth\_dog\_lora"

accelerate launch train_dreambooth_lora.py   --report_to wandb   --pretrained_model_name_or_path=$MODEL\_NAME    --instance_data_dir=$INSTANCE\_DIR   --output_dir=$OUTPUT\_DIR   --instance_prompt="a sks dog"   --resolution=64   --train_batch_size=4   --gradient_accumulation_steps=1   --learning_rate=5e-6   --scale_lr   --max_train_steps=1200   --validation_prompt="a sks dog"   --validation_epochs=25   --checkpointing_steps=100   --pre_compute_text_embeddings   --tokenizer_max_length=77   --text_encoder_use_attention_mask
```


### 


 IF 第二阶段 LoRA Dreambooth


`--validation_images`
 ：这些图像在验证步骤期间被放大。


`--class_labels_conditioning=时间步长`
 ：将第二阶段所需的额外调节传递给 UNet。


`--learning_rate=1e-6`
 ：学习率低于第一阶段。


`--分辨率=256`
 ：升级者期望更高分辨率的输入



```
export MODEL_NAME="DeepFloyd/IF-II-L-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth\_dog\_upscale"
export VALIDATION_IMAGES="dog\_downsized/image\_1.png dog\_downsized/image\_2.png dog\_downsized/image\_3.png dog\_downsized/image\_4.png"

python train_dreambooth_lora.py     --report_to wandb     --pretrained_model_name_or_path=$MODEL\_NAME     --instance_data_dir=$INSTANCE\_DIR     --output_dir=$OUTPUT\_DIR     --instance_prompt="a sks dog"     --resolution=256     --train_batch_size=4     --gradient_accumulation_steps=1     --learning_rate=1e-6 \ 
    --max_train_steps=2000     --validation_prompt="a sks dog"     --validation_epochs=100     --checkpointing_steps=500     --pre_compute_text_embeddings     --tokenizer_max_length=77     --text_encoder_use_attention_mask     --validation_images $VALIDATION\_IMAGES     --class_labels_conditioning=timesteps
```


### 


 IF 第一阶段 Full Dreambooth


`--skip_save_text_encoder`
 ：训练完整模型时，这将跳过使用微调模型保存整个 T5。您仍然可以加载管道
带有从原始模型加载的 T5。


`use_8bit_adam`
 ：由于优化器状态的大小，我们建议使用 8 位 adam 训练完整的 XL IF 模型。


`--learning_rate=1e-7`
 ：对于完整的 Dreambooth，IF 需要非常低的学习率。学习率越高，模型质量就会下降。请注意，它是
学习率可能会随着批量大小的增加而增加。


使用 8 位 adam 和批量大小 4，可以在 ~48 GB VRAM 中训练模型。



```
export MODEL_NAME="DeepFloyd/IF-I-XL-v1.0"

export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth\_if"

accelerate launch train_dreambooth.py   --pretrained_model_name_or_path=$MODEL\_NAME    --instance_data_dir=$INSTANCE\_DIR   --output_dir=$OUTPUT\_DIR   --instance_prompt="a photo of sks dog"   --resolution=64   --train_batch_size=4   --gradient_accumulation_steps=1   --learning_rate=1e-7   --max_train_steps=150   --validation_prompt "a photo of sks dog"   --validation_steps 25   --text_encoder_use_attention_mask   --tokenizer_max_length 77   --pre_compute_text_embeddings   --use_8bit_adam   --set_grads_to_none   --skip_save_text_encoder   --push_to_hub
```


### 


 IF Stage II 完整梦想展台


`--learning_rate=5e-6`
 ：当有效批量大小为 4 时，我们发现我们需要的学习率低至
1e-8。


`--分辨率=256`
 ：升级者期望更高分辨率的输入


`--train_batch_size=2`
 和
 `--gradient_accumulation_steps=6`
 ：我们发现第二阶段的全面训练尤其是
面需要大的有效批量大小。



```
export MODEL_NAME="DeepFloyd/IF-II-L-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth\_dog\_upscale"
export VALIDATION_IMAGES="dog\_downsized/image\_1.png dog\_downsized/image\_2.png dog\_downsized/image\_3.png dog\_downsized/image\_4.png"

accelerate launch train_dreambooth.py   --report_to wandb   --pretrained_model_name_or_path=$MODEL\_NAME   --instance_data_dir=$INSTANCE\_DIR   --output_dir=$OUTPUT\_DIR   --instance_prompt="a sks dog"   --resolution=256   --train_batch_size=2   --gradient_accumulation_steps=6   --learning_rate=5e-6   --max_train_steps=2000   --validation_prompt="a sks dog"   --validation_steps=150   --checkpointing_steps=500   --pre_compute_text_embeddings   --tokenizer_max_length=77   --text_encoder_use_attention_mask   --validation_images $VALIDATION\_IMAGES   --class_labels_conditioning timesteps   --push_to_hub
```


## 稳定扩散XL



我们支持对 UNet 和文本编码器进行微调
 [稳定扩散XL](https://huggingface.co/papers/2307.01952)
 通过 DreamBooth 和 LoRA
 `train_dreambooth_lora_sdxl.py`
 脚本。请参考文档
 [此处](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md)
 。