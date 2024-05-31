# [](#wuerstchen) Wuerstchen

> 译者：译者：[疾风兔X](https://github.com/jifnegtu)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/wuerstchen>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/wuerstchen>


[Wuerstchen](https://hf.co/papers/2306.00637) 模型通过将潜在空间压缩 42 倍，在不影响图像质量和加速推理的情况下，大幅降低了计算成本。在训练过程中，Wuerstchen 使用两个模型（VQGAN + 自动编码器）来压缩潜伏，然后第三个模型（文本条件潜在扩散模型）以这个高度压缩的空间为条件来生成图像。

要将先前的模型拟合到 GPU 内存中并加快训练速度，请尝试分别启用`gradient_accumulation_steps` 、`gradient_checkpointing` 和 `mixed_precision`。

本指南探讨了 [train\_text\_to\_image\_prior.py](https://github.com/huggingface/diffusers/blob/main/examples/wuerstchen/text_to_image/train_text_to_image_prior.py) 脚本，以帮助您更加熟悉它，以及如何根据自己的用例调整它。

在运行脚本之前，请确保从源代码安装库：

```py
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```
然后导航到包含训练脚本的示例文件夹，并安装正在使用的脚本所需的依赖项：

```py
cd examples/wuerstchen/text\_to\_image
pip install -r requirements.txt
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
最后，如果要在自己的数据集上训练模型，请查看[创建用于训练的数据集（ Create a dataset for training ）](create_dataset)指南，了解如何创建适用于训练脚本的数据集。

>以下部分重点介绍了训练脚本中对于了解如何修改训练脚本很重要的部分，但并未详细介绍[脚本](https://github.com/huggingface/diffusers/blob/main/examples/wuerstchen/text_to_image/train_text_to_image_prior.py)的各个方面。如果您有兴趣了解更多信息，请随时阅读脚本，如果您有任何问题或疑虑，请告诉我们。

## [](#script-parameters)脚本参数（Script parameters）

训练脚本提供了许多参数来帮助您自定义训练运行。所有参数及其描述都可以在 [`parse_args（）`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L192) 函数中找到。它为每个参数提供默认值，例如训练批次大小和学习率，但您也可以根据需要在训练命令中设置自己的值。

例如，要使用 fp16 格式以混合精度加速训练，请将参数`--mixed_precision`添加到训练命令中：

```py
accelerate launch train_text_to_image_prior.py \
  --mixed_precision="fp16"
```
大多数参数与[文生图](text2image#script-parameters)训练指南中的参数相同，因此让我们直接进入 Wuerstchen 训练脚本！

## [](#training-script)训练脚本（Training script）

训练脚本也类似于[文生图](text2image#training-script)训练指南，但已修改以支持 Wuerstchen。本指南重点介绍 Wuerstchen 训练脚本独有的代码。

[`main（）`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L441) 函数首先初始化图像编码器（[EfficientNet](https://github.com/huggingface/diffusers/blob/main/examples/wuerstchen/text_to_image/modeling_efficient_net_encoder.py)），以及通常的调度程序和分词器。

```py
with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    pretrained_checkpoint_file = hf_hub_download("dome272/wuerstchen", filename="model_v2_stage_b.pt")
    state_dict = torch.load(pretrained_checkpoint_file, map_location="cpu")
    image_encoder = EfficientNetEncoder()
    image_encoder.load_state_dict(state_dict["effnet_state_dict"])
    image_encoder.eval()
```
您还将加载`WuerstchenPrior`模型进行优化。

```py
prior = WuerstchenPrior.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="prior")

optimizer = optimizer_cls(
    prior.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```
接下来，您将对图像应用一些[转换（transforms）](https://github.com/huggingface/diffusers/blob/65ef7a0c5c594b4f84092e328fbdd73183613b30/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L656)并[标记标题（tokenize）](https://github.com/huggingface/diffusers/blob/65ef7a0c5c594b4f84092e328fbdd73183613b30/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L637)：

```py
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["effnet_pixel_values"] = [effnet_transforms(image) for image in images]
    examples["text_input_ids"], examples["text_mask"] = tokenize_captions(examples)
    return examples
```
最后，[训练循环（training loop ）](https://github.com/huggingface/diffusers/blob/65ef7a0c5c594b4f84092e328fbdd73183613b30/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L656)处理使用`EfficientNetEncoder`将图像压缩到潜在空间，向潜在空间添加噪声，并使用`WuerstchenPrior`模型预测噪声残留。

```py
pred_noise = prior(noisy_latents, timesteps, prompt_embeds)
```
如果您想了解有关训练循环工作原理的更多信息，请查看[了解管道、模型和调度程序（ Understanding pipelines, models and schedulers）](../using-diffusers/write_own_pipeline)教程，该教程分解了去噪过程的基本模式。

## [](#launch-the-script)启动脚本（Launch the script）

完成所有更改或对默认配置感到满意后，就可以启动训练脚本了！🚀

将环境变量`DATASET_NAME`设置为中心中的数据集名称。本指南使用 [ Naruto BLIP captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions)数据集，但您也可以在自己的数据集上创建和训练（请参阅[创建用于训练的数据集（Create a dataset for training）](create_dataset)指南）。

要使用权重和偏差监控训练进度，请将参数`--report_to=wandb`添加到训练命令中。您还需要将`--validation_prompt`添加到训练命令中以跟踪结果。这对于调试模型和查看中间结果非常有用。

```py
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch  train_text_to_image_prior.py \
  --mixed_precision="fp16" \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --dataloader_num_workers=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_prompts="A robot naruto, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="wuerstchen-prior-naruto-model"
```
训练完成后，您可以使用新训练的模型进行推理！

```py
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

pipeline = AutoPipelineForText2Image.from_pretrained("path/to/saved/model", torch_dtype=torch.float16).to("cuda")

caption = "A cute bird naruto holding a shield"
images = pipeline(
    caption,
    width=1024,
    height=1536,
    prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    prior_guidance_scale=4.0,
    num_images_per_prompt=2,
).images
```
## [](#next-steps)后续步骤（Next steps）

恭喜您训练了 Wuerstchen 模型！要了解有关如何使用新模型的更多信息，以下内容可能会有所帮助：

+   请查看 [Wuerstchen](../api/pipelines/wuerstchen#text-to-image-generation) API 文档，详细了解如何使用管道生成文本到图像及其限制。