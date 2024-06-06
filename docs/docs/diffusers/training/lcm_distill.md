# [](#latent-consistency-distillation)潜在一致性蒸馏（Latent Consistency Distillation）

> 译者：[疾风兔X](https://github.com/jifnegtu)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/lcm_distill>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/lcm_distill>



[潜在一致性模型 （LCMs）](https://hf.co/papers/2310.04378) 只需几个步骤即可生成高质量的图像，这代表了一个巨大的飞跃，因为许多管道至少需要 25+ 个步骤。LCM 是通过将潜在一致性蒸馏方法应用于任何稳定扩散模型来生产的。该方法的工作原理是将 *单阶段引导蒸馏（one-stage guided distillation）* 应用于潜在空间，并结合 *跳过步骤（skipping-step）* 方法以一致跳过时间步长以加速蒸馏过程（有关详细信息，请参阅论文的第 4.1、4.2 和 4.3 节）。

如果在 vRAM 有限的 GPU 上进行训练，请尝试启用 `gradient_checkpointing` 、`gradient_accumulation_steps` 和 `mixed_precision` 以减少内存使用并加快训练速度。通过使用 [xFormers](../optimization/xformers) 和 [bitsandbytes ](https://github.com/TimDettmers/bitsandbytes) 的8 位优化器实现内存效率高的注意力，您可以进一步减少内存使用量。

本指南将探讨 [train\_lcm\_distill\_sd\_wds.py](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_sd_wds.py) 脚本，以帮助您更加熟悉它，以及如何根据自己的用例调整它。

在运行脚本之前，请确保从源代码安装库：

```py
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```
然后导航到包含训练脚本的示例文件夹，并安装正在使用的脚本所需的依赖项：

```py
cd examples/consistency_distillation
pip install -r requirements.txt
```
>🤗 Accelerate 是一个库，可帮助您在多个 GPU/TPU 上或以混合精度进行训练。它将根据您的硬件和环境自动配置您的训练设置。请查看 🤗 Accelerate [Quick 教程](https://huggingface.co/docs/accelerate/quicktour)以了解更多信息。

初始化 🤗 Accelerate 环境（尝试启用`torch.compile`以显著加快训练速度）：

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
最后，如果要在自己的数据集上训练模型，请查看[创建用于训练的数据集（ Create a dataset for training ）](create_dataset)指南，了解如何创建适用于训练脚本的数据集。

## [](#script-parameters)脚本参数（Script parameters）

>以下部分重点介绍了训练脚本中对于了解如何修改它很重要的部分，但并未详细介绍脚本的各个方面。如果您有兴趣了解更多信息，请随时通读[脚本](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_sd_wds.py)，如果您有任何问题或疑虑，请告诉我们。

训练脚本提供了许多参数来帮助您自定义训练运行。所有参数及其描述都可以在 [`parse_args（）`](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L419) 函数中找到。此函数为每个参数提供默认值，例如训练批次大小（training batch size）和学习率（ learning rate），但您也可以根据需要在训练命令中设置自己的值。

例如，要使用 fp16 格式以混合精度加速训练，请将参数`--mixed_precision`添加到训练命令中：

```py
accelerate launch train_lcm_distill_sd_wds.py \
  --mixed_precision="fp16"
```
大多数参数与[文生图](text2image#script-parameters)训练指南中的参数相同，因此在本指南中，您将重点介绍与潜在一致性蒸馏相关的参数。

+   `--pretrained_teacher_model`：用作教师模型的预训练潜在扩散模型的路径
+   `--pretrained_vae_model_name_or_path`：通往预训练 VAE 的路径;已知 SDXL VAE 存在数值不稳定性，因此此参数允许您指定替代 VAE（例如 madebyollin 的 [VAE]((https://huggingface.co/madebyollin/sdxl-vae-fp16-fix))，它在 fp16 中工作）
+   `--w_min`以及`--w_max`：指导标度抽样的最小和最大指导标度值
+   `--num_ddim_timesteps`：DDIM 采样的时间步长数
+   `--loss_type`：要计算潜在一致性度蒸馏的损失类型（L2或Huber）;Huber 损失通常是首选，因为它对异常值更稳健
+   `--huber_c`：Huber 损失参数

## [](#training-script)训练脚本（Training script）

训练脚本首先创建一个数据集类 [`Text2ImageDataset`](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L141)，用于预处理图像并创建训练数据集。

```py
def transform(example):
    image = example["image"]
    image = TF.resize(image, resolution, interpolation=transforms.InterpolationMode.BILINEAR)

    c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(resolution, resolution))
    image = TF.crop(image, c_top, c_left, resolution, resolution)
    image = TF.to_tensor(image)
    image = TF.normalize(image, [0.5], [0.5])

    example["image"] = image
    return example
```
为了提高读取和写入存储在云中的大型数据集的性能，此脚本使用 [WebDataset](https://github.com/webdataset/webdataset) 格式创建预处理管道以应用转换并创建用于训练的数据集和数据加载器。图像经过处理并馈送到训练循环，无需先下载完整的数据集。

```py
processing_pipeline = [
    wds.decode("pil", handler=wds.ignore_and_continue),
    wds.rename(image="jpg;png;jpeg;webp", text="text;txt;caption", handler=wds.warn_and_continue),
    wds.map(filter_keys({"image", "text"})),
    wds.map(transform),
    wds.to_tuple("image", "text"),
]
```
在 [`main（）`](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L768) 函数中，加载了所有必要的组件，如噪声调度器（noise scheduler）、分词器（tokenizers）、文本编码器（text encoders）和 VAE。教师 UNet 也加载到此处，然后您可以从教师 UNet 创建一个学生 UNet。在训练期间，优化器（optimizer）会更新学生 UNet。

```py
teacher_unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_teacher_model, subfolder="unet", revision=args.teacher_revision
)

unet = UNet2DConditionModel(**teacher_unet.config)
unet.load_state_dict(teacher_unet.state_dict(), strict=False)
unet.train()
```
现在，您可以创建[优化器（optimizer）](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L979)来更新 UNet 参数：

```py
optimizer = optimizer_class(
    unet.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```
创建[数据集](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L994)：

```py
dataset = Text2ImageDataset(
    train_shards_path_or_url=args.train_shards_path_or_url,
    num_train_examples=args.max_train_samples,
    per_gpu_batch_size=args.train_batch_size,
    global_batch_size=args.train_batch_size * accelerator.num_processes,
    num_workers=args.dataloader_num_workers,
    resolution=args.resolution,
    shuffle_buffer_size=1000,
    pin_memory=True,
    persistent_workers=True,
)
train_dataloader = dataset.train_dataloader
```
接下来，您已准备好设置[训练循环](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L1049)并实现潜在一致性蒸馏方法（有关详细信息，请参阅本文中的算法 1）。脚本的这一部分负责向潜伏添加噪声、采样和创建引导标度嵌入，以及从噪声中预测原始图像。

```py
pred_x_0 = predicted_origin(
    noise_pred,
    start_timesteps,
    noisy_model_input,
    noise_scheduler.config.prediction_type,
    alpha_schedule,
    sigma_schedule,
)

model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0
```
它获取[教师模型预测（teacher model predictions ）](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L1172)和接下来的 [LCM 预测（ LCM predictions）](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L1209)，计算损失，然后将其反向传播到 LCM。

```py
if args.loss_type == "l2":
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
elif args.loss_type == "huber":
    loss = torch.mean(
        torch.sqrt((model_pred.float() - target.float()) ** 2 + args.huber_c**2) - args.huber_c
    )
```
如果您想了解有关训练循环工作原理的更多信息，请查看[了解管道、模型和调度程序教程（Understanding pipelines, models and schedulers tutorial ）](../using-diffusers/write_own_pipeline)，该教程分解了去噪过程的基本模式。

## [](#launch-the-script)启动脚本（Launch the script）

现在，您可以启动训练脚本并开始蒸馏了！

在本指南中，你将使用 `--train_shards_path_or_url` 指定存储在 Hub 上的 [Conceptual Captions 12M](https://github.com/google-research-datasets/conceptual-12m) 数据集的[路径。](https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset)将环境变量`MODEL_DIR`设置为教师模型的名称，并将`OUTPUT_DIR`设置为要保存模型的位置。

```py
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path/to/saved/model"

accelerate launch train_lcm_distill_sd_wds.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=512 \
    --learning_rate=1e-6 --loss_type="huber" --ema_decay=0.95 --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url="pipe:curl -L -s https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/{00000..01099}.tar?download=true" \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=10 \
    --train_batch_size=12 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=wandb \
    --seed=453645634 \
    --push_to_hub
```
训练完成后，您可以使用新的 LCM 进行推理。

```py
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
import torch

unet = UNet2DConditionModel.from_pretrained("your-username/your-model", torch_dtype=torch.float16, variant="fp16")
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", unet=unet, torch_dtype=torch.float16, variant="fp16")

pipeline.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipeline.to("cuda")

prompt = "sushi rolls in the form of panda heads, sushi platter"

image = pipeline(prompt, num_inference_steps=4, guidance_scale=1.0).images[0]
```
## [](#lora)LoRA

LoRA 是一种训练技术，用于显著减少可训练参数的数量。因此，训练速度更快，并且更容易存储生成的权重，因为它们要小得多（~100MB）。使用 [train\_lcm\_distill\_lora\_sd\_wds.py](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_lora_sd_wds.py) 或 [train\_lcm\_distill\_lora\_sdxl.wds.py](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_lora_sdxl_wds.py) 脚本使用 LoRA 进行训练。

[LoRA 培训](lora)指南中更详细地讨论了 LoRA 培训脚本。

## [](#stable-diffusion-xl)Stable Diffusion XL

Stable Diffusion XL （SDXL） 是一个功能强大的文生图模型，可生成高分辨率图像，并在其架构中添加了第二个文本编码器。使用 [train\_lcm\_distill\_sdxl\_wds.py](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_sdxl_wds.py) 脚本通过 LoRA 训练 SDXL 模型。

[SDXL 培训](sdxl)指南中更详细地讨论了 SDXL 培训脚本。

## [](#next-steps)后续步骤

恭喜您蒸馏出 LCM 模型！要了解有关 LCM 的更多信息，以下内容可能会有所帮助：

+   了解如何使用 [LCM 对](../using-diffusers/lcm)文生图、图生图以及 LoRA 检查点进行推理。
+   阅读 [SDXL in 4 Steps with Latent Consistency LoRAs](https://huggingface.co/blog/lcm_lora) 博客文章，了解有关 SDXL LCM-LoRA 的更多信息，以实现超快速推理、质量比较、基准测试等。