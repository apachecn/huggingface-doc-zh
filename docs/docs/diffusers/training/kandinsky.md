# [](#kandinsky-22)Kandinsky 2.2

> 译者：[疾风兔X](https://github.com/jifnegtu)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/kandinsky>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/kandinsky>


>这个脚本是实验性的，很容易过度拟合并遇到灾难性遗忘等问题。尝试探索不同的超参数，以获得数据集的最佳结果。

Kandinsky 2.2 是一个多语言文生图（text-to-image）模型，能够生成更逼真的图像。该模型包括一个图像先验模型（image prior model）用于从文本提示创建图像嵌入，以及一个解码器模型（decoder model），该模型基于先验模型的嵌入来生成图像。这就是为什么在Diffusers中你会发现用于训练Kandinsky 2.2的两个独立的脚本，一个用于训练先验模型，另一个用于训练解码器模型。您可以分别训练这两个模型，但要获得最佳结果，您应该同时训练先前模型和解码器模型。

根据您的 GPU，您可能需要启用 `gradient_checkpointing`（⚠️以前的模型不支持！），`mixed_precision` 和 `gradient_accumulation_steps` 来帮助将模型放入内存并加快训练速度。您可以通过使用 [xFormers](../optimization/xformers) 启用内存效率高的注意力来进一步减少内存使用量（版本 [v0.0.16](https://github.com/huggingface/diffusers/issues/2234#issuecomment-1416931212) 在某些 GPU 上无法进行训练，因此您可能需要安装开发版本）。

本指南探讨了[train\_text\_to\_image\_prior.py](https://github.com/huggingface/diffusers/blob/main/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py)和[train\_text\_to\_image\_decoder.py](https://github.com/huggingface/diffusers/blob/main/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py)脚本，以帮助您更加熟悉它，以及如何根据自己的用例调整它。

在运行脚本之前，请确保从源代码安装库：

```py
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```
然后导航到包含训练脚本的示例文件夹，并安装正在使用的脚本所需的依赖项：

```py
cd examples/kandinsky2_2/text_to_image
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
最后，如果要在自己的数据集上训练模型，请查看[创建用于训练的数据集](create_dataset)指南，了解如何创建适用于训练脚本的数据集。

以下部分重点介绍了训练脚本中对于了解如何修改训练脚本很重要的部分，但并未详细介绍脚本的各个方面。如果您有兴趣了解更多信息，请随时阅读脚本，如果您有任何问题或疑虑，请告诉我们。

## [](#script-parameters)脚本参数（Script parameters）

训练脚本提供了许多参数来帮助您自定义训练运行。所有参数及其描述都可以在 [`parse_args（）`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L190) 函数中找到。训练脚本为每个参数提供默认值，例如训练批大小和学习率，但您也可以根据需要在训练命令中设置自己的值。

例如，要使用 fp16 格式以混合精度加速训练，请将参数`--mixed_precision`添加到训练命令中：

```py
accelerate launch train_text_to_image_prior.py \
  --mixed_precision="fp16"
```
大多数参数与[文生图（Text-to-image）](text2image#script-parameters)训练指南中的参数相同，因此让我们直接进入Kandinsky训练脚本的演练！

### [](#min-snr-weighting)最小信噪比加权（Min-SNR weighting）

[Min-SNR](https://huggingface.co/papers/2303.09556) 加权策略可以通过重新平衡损失来实现更快的收敛来帮助训练。训练脚本支持预测 `epsilon`（噪声）或 `v_prediction` ，但 Min-SNR 与这两种预测类型兼容。此加权策略仅受 PyTorch 支持，在 Flax 训练脚本中不可用。

添加参数`--snr_gamma`并将其设置为建议的值 5.0：

```py
accelerate launch train_text_to_image_prior.py \
  --snr_gamma=5.0
```
## [](#training-script)训练脚本（Training script）

训练脚本也类似于[文生图（Text-to-image）](text2image#training-script)训练指南，但已对其进行修改，以支持训练先前模型和解码器模型。本指南重点介绍 Kandinsky 2.2 训练脚本所特有的代码。


### 先验模型（prior model）
[`main（）`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L441) 函数包含用于准备数据集和训练模型的代码。

您会立即注意到的主要区别之一是，训练脚本还加载了一个 [CLIPImageProcessor](https://huggingface.co/docs/transformers/v4.39.0/en/model_doc/clip#transformers.CLIPImageProcessor)（除了调度程序和分词器）用于预处理图像，以及一个用于对图像进行编码的 [CLIPVisionModelWithProjection](https://huggingface.co/docs/transformers/v4.39.0/en/model_doc/clip#transformers.CLIPVisionModelWithProjection) 模型：

```py
noise_scheduler = DDPMScheduler(beta_schedule="squaredcos_cap_v2", prediction_type="sample")
image_processor = CLIPImageProcessor.from_pretrained(
    args.pretrained_prior_model_name_or_path, subfolder="image_processor"
)
tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="tokenizer")

with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="image_encoder", torch_dtype=weight_dtype
    ).eval()
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
    ).eval()
```
Kandinsky 使用 [PriorTransformer](/docs/diffusers/v0.27.2/en/api/models/prior_transformer#diffusers.PriorTransformer) 来生成图像嵌入，因此您需要设置优化器来了解先验模式的参数。

```py
prior = PriorTransformer.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="prior")
prior.train()
optimizer = optimizer_cls(
    prior.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```
接下来，对输入字幕进行标记化，并由 [CLIPImageProcessor](https://huggingface.co/docs/transformers/v4.39.0/en/model_doc/clip#transformers.CLIPImageProcessor) 对图像进行[预处理](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L632)：

```py
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["clip_pixel_values"] = image_processor(images, return_tensors="pt").pixel_values
    examples["text_input_ids"], examples["text_mask"] = tokenize_captions(examples)
    return examples
```
最后，[训练循环](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L718)将输入图像转换为潜能图像，向图像嵌入添加噪声，并做出预测：

```py
model_pred = prior(
    noisy_latents,
    timestep=timesteps,
    proj_embedding=prompt_embeds,
    encoder_hidden_states=text_encoder_hidden_states,
    attention_mask=text_mask,
).predicted_image_embedding
```
如果您想了解有关训练循环工作原理的更多信息，请查看[了解管道、模型和调度程序（ Understanding pipelines, models and schedulers）](../using-diffusers/write_own_pipeline)教程，该教程分解了去噪过程的基本模式。

### 解码器模型（decoder model）
[`main（）`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py#L440) 函数包含用于准备数据集和训练模型的代码。

与之前的模型不同，解码器初始化 [VQModel](/docs/diffusers/v0.27.2/en/api/models/vq#diffusers.VQModel) 以将潜伏物解码为图像，并使用 [UNet2DConditionModel](/docs/diffusers/v0.27.2/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)：

```py
with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    vae = VQModel.from_pretrained(
        args.pretrained_decoder_model_name_or_path, subfolder="movq", torch_dtype=weight_dtype
    ).eval()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="image_encoder", torch_dtype=weight_dtype
    ).eval()
unet = UNet2DConditionModel.from_pretrained(args.pretrained_decoder_model_name_or_path, subfolder="unet")
```
接下来，该脚本包括多个图像转换和一个[预处理](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py#L622)函数，用于将转换应用于图像并返回像素值：

```py
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["clip_pixel_values"] = image_processor(images, return_tensors="pt").pixel_values
    return examples
```
最后，[训练循环](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py#L706)处理将图像转换为潜伏图像、添加噪声和预测噪声残差。

如果您想了解有关训练循环工作原理的更多信息，请查看[了解管道、模型和调度程序（Understanding pipelines, models and schedulers）](../using-diffusers/write_own_pipeline)教程，该教程分解了去噪过程的基本模式。

```py
model_pred = unet(noisy_latents, timesteps, None, added_cond_kwargs=added_cond_kwargs).sample[:, :4]
```
## [](#launch-the-script)启动脚本（Launch the script）

完成所有更改或对默认配置感到满意后，就可以启动训练脚本了！🚀

你将在 [Naruto BLIP captions ](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)数据集上进行训练以生成自己的 Naruto 角色，但您也可以按照[创建用于训练的数据集（ Create a dataset for training）](create_dataset)指南在自己的数据集上创建和训练。将环境变量`DATASET_NAME`设置为 Hub 上数据集的名称，或者，如果要对自己的文件进行训练，请将环境变量`TRAIN_DIR`设置为数据集的路径。

如果要在多个 GPU 上进行训练，请将参数`--multi_gpu``accelerate launch`添加到命令中。

要使用权重和偏差监控训练进度，请将参数`--report_to=wandb`添加到训练命令中。您还需要将 `--validation_prompt` 添加到训练命令中以跟踪结果。这对于调试模型和查看中间结果非常有用。

### 先验模型

```py
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_prior.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_prompts="A robot naruto, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="kandi2-prior-naruto-model"
```

### 解码器模型
```py
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_decoder.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_prompts="A robot naruto, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="kandi2-decoder-naruto-model"
```
训练完成后，您可以使用新训练的模型进行推理！

### 先验模型

```py
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import torch

prior_pipeline = DiffusionPipeline.from_pretrained(output_dir, torch_dtype=torch.float16)
prior_components = {"prior_" + k: v for k,v in prior_pipeline.components.items()}
pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", **prior_components, torch_dtype=torch.float16)

pipe.enable_model_cpu_offload()
prompt="A robot naruto, 4k photo"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt).images[0]
```
>请随意使用您自己训练有素的解码器检查点替换`kandinsky-community/kandinsky-2-2-decoder`!
### 解码器模型

```py
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import torch

prior_pipeline = DiffusionPipeline.from_pretrained(output_dir, torch_dtype=torch.float16)
prior_components = {"prior_" + k: v for k,v in prior_pipeline.components.items()}
pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", **prior_components, torch_dtype=torch.float16)

pipe.enable_model_cpu_offload()
prompt="A robot naruto, 4k photo"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt).images[0]
```
对于解码器模型，还可以从保存的检查点执行推断，这对于查看中间结果很有用。在这种情况下，将检查点加载到UNet中:
```py
from diffusers import AutoPipelineForText2Image, UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("path/to/saved/model" + "/checkpoint-<N>/unet")

pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", unet=unet, torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

image = pipeline(prompt="A robot naruto, 4k photo").images[0]
```
## [](#next-steps)后续步骤（Next steps）

恭喜您训练了Kandinsky 2.2 模型！若要详细了解如何使用新模型，以下指南可能会有所帮助：

+   阅读[Kandinsky](../using-diffusers/kandinsky)指南，了解如何将其用于各种不同的任务（文生图、图生图、修复、插值），以及如何将其与 ControlNet 结合使用。
+   查看 [DreamBooth](dreambooth) 和 [LoRA](lora) 培训指南，了解如何使用几张示例图像训练个性化的Kandinsky模型。这两种训练技巧甚至可以结合使用！