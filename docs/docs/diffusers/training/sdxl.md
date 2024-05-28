# [](#stable-diffusion-xl)Stable Diffusion XL

> 译者：[疾风兔X](https://github.com/jifnegtu)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/sdxl>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/sdxl>


>这个脚本是实验性的，很容易过度拟合并遇到灾难性遗忘等问题。尝试探索不同的超参数，以获得数据集的最佳结果。

[Stable Diffusion XL （SDXL）](https://hf.co/papers/2307.01952) 是 Stable Diffusion 模型的更大、更强的迭代版本，能够生成更高分辨率的图像。

SDXL 的 UNet 大了 3 倍，该模型在架构中添加了第二个文本编码器。根据您所拥有的硬件条件，这可能非常耗费计算量，并且可能无法在像 Tesla T4 这样的消费类 GPU 上运行。为了帮助将这个更大的模型放入内存并加快训练速度，请尝试启用`gradient_checkpointing`、`mixed_precision`和`gradient_accumulation_steps`。通过使用 [xFormers](../optimization/xformers) 的记忆高效注意（memory-efficient attention）和 [bitsandbytes ](https://github.com/TimDettmers/bitsandbytes) 的8位优化器（8-bit optimizer），您可以进一步减少内存使用量。

本指南将探讨[train\_text\_to\_image\_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py)训练脚本，以帮助您更加熟悉它，以及如何根据自己的用例调整它。

在运行脚本之前，请确保从源代码安装库：

```py
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```
然后导航到包含训练脚本的示例文件夹，并安装正在使用的脚本所需的依赖项：

```py
cd examples/text_to_image
pip install -r requirements_sdxl.txt
```
>🤗 Accelerate 是一个库，可帮助您在多个 GPU/TPU 上或以混合精度进行训练。它将根据您的硬件和环境自动配置您的训练设置。请查看 🤗 Accelerate [Quick 教程](https://huggingface.co/docs/accelerate/quicktour)以了解更多信息。

初始化 🤗 Accelerate 环境：

```py
accelerate config
```

要在不选择任何配置的情况下设置默认🤗的 Accelerate 环境，请执行以下操作：

```py
accelerate config default
```

或者，如果您的环境不支持交互式 shell（如笔记本），则可以使用：

```py
from accelerate.utils import write_basic_config

write_basic_config()
```
最后，如果要在自己的数据集上训练模型，请查看[创建用于训练的数据集](create_dataset)指南，了解如何创建适用于训练脚本的数据集。

## [](#script-parameters)脚本参数(Script parameters)

>以下部分重点介绍了训练脚本中对于了解如何修改它很重要的部分，但并未详细介绍脚本的各个方面。如果您有兴趣了解更多信息，请随时通读[脚本](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py)，如果您有任何问题或疑虑，请告诉我们。

训练脚本提供了许多参数来帮助您自定义训练运行。所有参数及其描述都可以在 [`parse_args（）`](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L129) 函数中找到。此函数为每个参数提供默认值，例如训练批次大小和学习率，但您也可以根据需要在训练命令中设置自己的值。

例如，要使用 bf16 格式以混合精度加速训练，请将参数`--mixed_precision`添加到训练命令中：

```py
accelerate launch train_text_to_image_sdxl.py \
  --mixed_precision="bf16"
```
大多数参数与[文生图（ Text-to-image ）](text2image#script-parameters)训练指南中的参数相同，因此本指南将重点介绍与训练SDXL相关的参数。

+   `--pretrained_vae_model_name_or_path`：通往预训练 VAE 的路径;已知 SDXL VAE 存在数值不稳定性，因此此参数允许您指定更好的 [VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
+   `--proportion_empty_prompts`：图片提示替换为空字符串的比例
+   `--timestep_bias_strategy`：其中（较早与较晚）在时间步长中应用偏差，这可以鼓励模型学习低频或高频细节
+   `--timestep_bias_multiplier`：应用于时间步长的偏置权重
+   `--timestep_bias_begin`：开始应用偏差的时间步长
+   `--timestep_bias_end`：结束应用偏差的时间步长
+   `--timestep_bias_portion`：应用偏差的时间步长比例

### [](#min-snr-weighting)最小信噪比加权（Min-SNR weighting）

[Min-SNR](https://huggingface.co/papers/2303.09556) 加权策略可以通过重新平衡损失来实现更快的收敛来帮助训练。训练脚本支持预测 `epsilon`（噪声） 或 `v_prediction` ，但 Min-SNR 与这两种预测类型兼容。此加权策略仅受 PyTorch 支持，在 Flax 训练脚本中不可用。

添加`--snr_gamma`参数并将其设置为建议的值 5.0：

```py
accelerate launch train_text_to_image_sdxl.py \
  --snr_gamma=5.0
```
## [](#training-script)训练脚本（Training script）

训练脚本也类似于[文生图](text2image#training-script)训练指南，但已修改为支持 SDXL 训练。本指南将重点介绍 SDXL 训练脚本独有的代码。

它首先创建函数来[标记提示（ tokenize the prompts）](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L478)以计算提示嵌入，并使用 [VAE](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L519) 计算图像嵌入。接下来，您将使用一个函数来生成[时间步长权重（ generate the timesteps weights）](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L531)，具体取决于要应用的时间步长偏差策略。

在 [`main（）`](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L572) 函数中，除了加载分词器外，脚本还会加载第二个分词器和文本编码器，因为 SDXL 架构分别使用其中两个：

```py
tokenizer_one = AutoTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
)
tokenizer_two = AutoTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False
)

text_encoder_cls_one = import_model_class_from_model_name_or_path(
    args.pretrained_model_name_or_path, args.revision
)
text_encoder_cls_two = import_model_class_from_model_name_or_path(
    args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
)
```
[首先计算提示和图像嵌入（ prompt and image embeddings ）](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L857)并保留在内存中，这对于较小的数据集通常不是问题，但对于较大的数据集，它可能会导致内存问题。如果是这种情况，则应将预先计算的嵌入单独保存到磁盘，并在训练过程中将它们加载到内存中（有关此主题的更多讨论，请参阅此 [PR](https://github.com/huggingface/diffusers/pull/4505)）。

```py
text_encoders = [text_encoder_one, text_encoder_two]
tokenizers = [tokenizer_one, tokenizer_two]
compute_embeddings_fn = functools.partial(
    encode_prompt,
    text_encoders=text_encoders,
    tokenizers=tokenizers,
    proportion_empty_prompts=args.proportion_empty_prompts,
    caption_column=args.caption_column,
)

train_dataset = train_dataset.map(compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint)
train_dataset = train_dataset.map(
    compute_vae_encodings_fn,
    batched=True,
    batch_size=args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
    new_fingerprint=new_fingerprint_for_vae,
)
```
计算嵌入后，删除文本编码器、VAE 和分词器以释放一些内存：

```py
del text_encoders, tokenizers, vae
gc.collect()
torch.cuda.empty_cache()
```
最后，[训练循环（training loop）](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/text_to_image/train_text_to_image_sdxl.py#L943)负责剩下的工作。如果选择应用时间步长偏置策略，则会看到时间步长权重被计算并添加为噪声：

```py
weights = generate_timestep_weights(args, noise_scheduler.config.num_train_timesteps).to(
        model_input.device
    )
    timesteps = torch.multinomial(weights, bsz, replacement=True).long()

noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
```
如果您想了解有关训练循环工作原理的更多信息，请查看[了解管道、模型和调度程序（ Understanding pipelines, models and schedulers）](../using-diffusers/write_own_pipeline)教程，该教程分解了去噪过程的基本模式。

## [](#launch-the-script)启动脚本（Training script）
完成所有更改或对默认配置感到满意后，就可以启动训练脚本了！🚀

让我们在 [ Pokémon BLIP captions ](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)数据集上进行训练，以生成您自己的 Pokémon。设置环境变量`MODEL_NAME`以及`DATASET_NAME`模型和数据集（从中心或本地路径）。您还应该指定 SDXL VAE 以外的 VAE（来自 Hub 或本地路径），并用`VAE_NAME`来避免避免数字不稳定。

>要使用权重和偏差监控训练进度，请将参数`--report_to=wandb`添加到训练命令中。您还需要将 `--validation_prompt` 和 `--validation_epochs` 添加到训练命令中以跟踪结果。这对于调试模型和查看中间结果非常有用。

```py
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --enable_xformers_memory_efficient_attention \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="a cute Sundar Pichai creature" \
  --validation_epochs 5 \
  --checkpointing_steps=5000 \
  --output_dir="sdxl-pokemon-model" \
  --push_to_hub
```
完成训练后，您可以使用新训练的 SDXL 模型进行推理！

### PyTorch

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("path/to/your/model", torch_dtype=torch.float16).to("cuda")

prompt = "A pokemon with green eyes and red legs."
image = pipeline(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon.png")
```
### PyTorch XLA

[PyTorch XLA](https://pytorch.org/xla) 允许您在 XLA 设备（如 TPU）上运行 PyTorch，这可能会更快。初始预热步骤需要更长的时间，因为需要编译和优化模型。但是，在与原始提示**相同长度**的输入上对管道的后续调用要快得多，因为它可以重用优化的图形。

```py
from diffusers import DiffusionPipeline
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0").to(device)

prompt = "A pokemon with green eyes and red legs."
start = time()
image = pipeline(prompt, num_inference_steps=inference_steps).images[0]
print(f'Compilation time is {time()-start} sec')
image.save("pokemon.png")

start = time()
image = pipeline(prompt, num_inference_steps=inference_steps).images[0]
print(f'Inference time is {time()-start} sec after compilation')
```
## [](#next-steps)后续步骤（Next steps）

恭喜您训练了 SDXL 模型！若要详细了解如何使用新模型，以下指南可能会有所帮助：

+   阅读 [Stable Diffusion XL](../using-diffusers/sdxl) 指南，了解如何将其用于各种不同的任务（文生图、图生图、修复）、如何使用它的精炼器模型以及不同类型的微调。
+   查看 [DreamBooth](dreambooth) 和 [LoRA](lora) 培训指南，了解如何使用几个示例图像来训练个性化的 SDXL 模型。这两种训练技巧甚至可以结合使用！