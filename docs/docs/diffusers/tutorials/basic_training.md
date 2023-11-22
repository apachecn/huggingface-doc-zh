![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


# 训练扩散模型

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/tutorials/basic_training>
>
> 原始地址：<https://huggingface.co/docs/diffusers/tutorials/basic_training>


无条件图像生成是扩散模型的一种流行应用，它生成的图像与用于训练的数据集中的图像相似。通常，最好的结果是通过在特定数据集上微调预训练模型来获得的。您可以在以下位置找到许多这样的检查点：
 [中心](https://huggingface.co/search/full-text?q=unconditional-image- Generation&type=model)
 ，但如果你找不到你喜欢的，你可以随时训练自己的！


本教程将教您如何训练
 [UNet2DModel](/docs/diffusers/v0.23.1/en/api/models/unet2d#diffusers.UNet2DModel)
 从头开始的一个子集
 [史密森蝴蝶](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset)
 数据集来生成您自己的🦋蝴蝶🦋。


💡 本培训教程基于
 [使用 🧨 扩散器进行训练](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb)
 笔记本。有关扩散模型的更多详细信息和背景（例如它们的工作原理），请查看笔记本！


在开始之前，请确保您安装了🤗数据集来加载和预处理图像数据集，并安装了🤗加速，以简化在任意数量的 GPU 上的训练。以下命令也将安装
 [TensorBoard](https://www.tensorflow.org/tensorboard)
 可视化训练指标（您也可以使用
 [权重和偏差](https://docs.wandb.ai/)
 跟踪您的训练）。



```
# uncomment to install the necessary libraries in Colab
#!pip install diffusers[training]
```


我们鼓励您与社区分享您的模型，为此，您需要登录您的 Hugging Face 帐户（创建一个
 [此处](https://hf.co/join)
 如果您还没有的话！）。您可以从笔记本登录并在出现提示时输入您的令牌。确保您的令牌具有写入角色。



```
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```


或者从终端登录：



```
huggingface-cli login
```


由于模型检查点相当大，因此安装
 [Git-LFS](https://git-lfs.com/)
 对这些大文件进行版本控制：



```
!sudo apt -qq install git-lfs
!git config --global credential.helper store
```


## 训练配置



为了方便起见，创建一个
 `训练配置`
 包含训练超参数的类（随意调整它们）：



```
>>> from dataclasses import dataclass

>>> @dataclass
... class TrainingConfig:
...     image_size = 128  # the generated image resolution
...     train_batch_size = 16
...     eval_batch_size = 16  # how many images to sample during evaluation
...     num_epochs = 50
...     gradient_accumulation_steps = 1
...     learning_rate = 1e-4
...     lr_warmup_steps = 500
...     save_image_epochs = 10
...     save_model_epochs = 30
...     mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
...     output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub

...     push_to_hub = True  # whether to upload the saved model to the HF Hub
...     hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
...     hub_private_repo = False
...     overwrite_output_dir = True  # overwrite the old model when re-running the notebook
...     seed = 0


>>> config = TrainingConfig()
```


## 加载数据集



您可以轻松加载
 [史密森蝴蝶](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset)
 带有🤗数据集库的数据集：



```
>>> from datasets import load_dataset

>>> config.dataset_name = "huggan/smithsonian\_butterflies\_subset"
>>> dataset = load_dataset(config.dataset_name, split="train")
```


💡 您可以从以下位置找到其他数据集
 [HugGan 社区活动](https://huggingface.co/huggan)
 或者您可以通过创建本地数据集来使用自己的数据集
 [`ImageFolder`](https://huggingface.co/docs/datasets/image_dataset#imagefolder)
 。放
 `config.dataset_name`
 如果数据集来自 HugGan 社区活动，则为数据集的存储库 ID，或者
 `图像文件夹`
 如果您使用自己的图像。


🤗 数据集使用
 [图片](https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes#datasets.Image)
 自动解码图像数据并将其加载为
 [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)
 我们可以想象：



```
>>> import matplotlib.pyplot as plt

>>> fig, axs = plt.subplots(1, 4, figsize=(16, 4))
>>> for i, image in enumerate(dataset[:4]["image"]):
...     axs[i].imshow(image)
...     axs[i].set_axis_off()
>>> fig.show()
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/butterflies_ds.png)


 不过，这些图像的尺寸各不相同，因此您需要先对它们进行预处理：


*“调整大小”
 将图像大小更改为中定义的大小
 `config.image_size`
 。
* `随机水平翻转`
 通过随机镜像图像来增强数据集。
*“标准化”
 将像素值重新缩放到 [-1, 1] 范围内非常重要，这正是模型所期望的。



```
>>> from torchvision import transforms

>>> preprocess = transforms.Compose(
...     [
...         transforms.Resize((config.image_size, config.image_size)),
...         transforms.RandomHorizontalFlip(),
...         transforms.ToTensor(),
...         transforms.Normalize([0.5], [0.5]),
...     ]
... )
```


使用🤗数据集’
 [set\_transform](https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes#datasets.Dataset.set_transform)
 方法来应用
 `预处理`
 训练期间动态运行的函数：



```
>>> def transform(examples):
...     images = [preprocess(image.convert("RGB")) for image in examples["image"]]
...     return {"images": images}


>>> dataset.set_transform(transform)
```


请随意再次可视化图像以确认它们已调整大小。现在您已准备好将数据集包装在
 [数据加载器](https://pytorch.org/docs/stable/data#torch.utils.data.DataLoader)
 为了训练！



```
>>> import torch

>>> train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
```


## 创建 UNet2D 模型



🧨 Diffusers 中的预训练模型可以使用您想要的参数轻松地从其模型类创建。例如，要创建一个
 [UNet2DModel](/docs/diffusers/v0.23.1/en/api/models/unet2d#diffusers.UNet2DModel)
 :



```
>>> from diffusers import UNet2DModel

>>> model = UNet2DModel(
...     sample_size=config.image_size,  # the target image resolution
...     in_channels=3,  # the number of input channels, 3 for RGB images
...     out_channels=3,  # the number of output channels
...     layers_per_block=2,  # how many ResNet layers to use per UNet block
...     block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
...     down_block_types=(
...         "DownBlock2D",  # a regular ResNet downsampling block
...         "DownBlock2D",
...         "DownBlock2D",
...         "DownBlock2D",
...         "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
...         "DownBlock2D",
...     ),
...     up_block_types=(
...         "UpBlock2D",  # a regular ResNet upsampling block
...         "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
...         "UpBlock2D",
...         "UpBlock2D",
...         "UpBlock2D",
...         "UpBlock2D",
...     ),
... )
```


快速检查样本图像形状与模型输出形状是否匹配通常是个好主意：



```
>>> sample_image = dataset[0]["images"].unsqueeze(0)
>>> print("Input shape:", sample_image.shape)
Input shape: torch.Size([1, 3, 128, 128])

>>> print("Output shape:", model(sample_image, timestep=0).sample.shape)
Output shape: torch.Size([1, 3, 128, 128])
```


伟大的！接下来，您需要一个调度程序来向图像添加一些噪声。


## 创建调度程序



根据您使用模型进行训练还是推理，调度程序的行为会有所不同。在推理过程中，调度器根据噪声生成图像。在训练期间，调度程序从扩散过程中的特定点获取模型输出（或样本），并根据
 *噪音时间表*
 和
 *更新规则*
 。


让我们看一下
 [DDPMScheduler](/docs/diffusers/v0.23.1/en/api/schedulers/ddpm#diffusers.DDPMScheduler)
 并使用
 `添加噪音`
 添加一些随机噪声的方法
 `样本图像`
 从以前：



```
>>> import torch
>>> from PIL import Image
>>> from diffusers import DDPMScheduler

>>> noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
>>> noise = torch.randn(sample_image.shape)
>>> timesteps = torch.LongTensor([50])
>>> noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

>>> Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/noisy_butterfly.png)


 模型的训练目标是预测添加到图像中的噪声。这一步的损失可以通过下式计算：



```
>>> import torch.nn.functional as F

>>> noise_pred = model(noisy_image, timesteps).sample
>>> loss = F.mse_loss(noise_pred, noise)
```


## 训练模型



到目前为止，您已经掌握了开始训练模型的大部分内容，剩下的就是将所有内容组合在一起。


首先，您需要一个优化器和一个学习率调度器：



```
>>> from diffusers.optimization import get_cosine_schedule_with_warmup

>>> optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
>>> lr_scheduler = get_cosine_schedule_with_warmup(
...     optimizer=optimizer,
...     num_warmup_steps=config.lr_warmup_steps,
...     num_training_steps=(len(train_dataloader) * config.num_epochs),
... )
```


然后，您需要一种评估模型的方法。对于评估，您可以使用
 [DDPMPipeline](/docs/diffusers/v0.23.1/en/api/pipelines/ddpm#diffusers.DDPMPipeline)
 生成一批样本图像并将其保存为网格：



```
>>> from diffusers import DDPMPipeline
>>> from diffusers.utils import make_image_grid
>>> import os

>>> def evaluate(config, epoch, pipeline):
...     # Sample some images from random noise (this is the backward diffusion process).
...     # The default pipeline output type is `List[PIL.Image]`
...     images = pipeline(
...         batch_size=config.eval_batch_size,
...         generator=torch.manual_seed(config.seed),
...     ).images

...     # Make a grid out of the images
...     image_grid = make_image_grid(images, rows=4, cols=4)

...     # Save the images
...     test_dir = os.path.join(config.output_dir, "samples")
...     os.makedirs(test_dir, exist_ok=True)
...     image_grid.save(f"{test\_dir}/{epoch:04d}.png")
```


现在，您可以使用 🤗 Accelerate 将所有这些组件包装在一个训练循环中，以轻松进行 TensorBoard 日志记录、梯度累积和混合精度训练。要将模型上传到 Hub，请编写一个函数来获取存储库名称和信息，然后将其推送到 Hub。


💡 下面的训练循环可能看起来令人生畏且漫长，但是当您仅用一行代码启动训练时，这一切都是值得的！如果您迫不及待地想要开始生成图像，请随意复制并运行下面的代码。您可以随时返回并更仔细地检查训练循环，例如当您等待模型完成训练时。 🤗



```
>>> from accelerate import Accelerator
>>> from huggingface_hub import create_repo, upload_folder
>>> from tqdm.auto import tqdm
>>> from pathlib import Path
>>> import os

>>> def train\_loop(config, model, noise\_scheduler, optimizer, train\_dataloader, lr\_scheduler):
...     # Initialize accelerator and tensorboard logging
...     accelerator = Accelerator(
...         mixed_precision=config.mixed_precision,
...         gradient_accumulation_steps=config.gradient_accumulation_steps,
...         log_with="tensorboard",
...         project_dir=os.path.join(config.output_dir, "logs"),
...     )
...     if accelerator.is_main_process:
...         if config.output_dir is not None:
...             os.makedirs(config.output_dir, exist_ok=True)
...         if config.push_to_hub:
...             repo_id = create_repo(
...                 repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
...             ).repo_id
...         accelerator.init_trackers("train\_example")

...     # Prepare everything
...     # There is no specific order to remember, you just need to unpack the
...     # objects in the same order you gave them to the prepare method.
...     model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
...         model, optimizer, train_dataloader, lr_scheduler
...     )

...     global_step = 0

...     # Now you train the model
...     for epoch in range(config.num_epochs):
...         progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
...         progress_bar.set_description(f"Epoch {epoch}")

...         for step, batch in enumerate(train_dataloader):
...             clean_images = batch["images"]
...             # Sample noise to add to the images
...             noise = torch.randn(clean_images.shape).to(clean_images.device)
...             bs = clean_images.shape[0]

...             # Sample a random timestep for each image
...             timesteps = torch.randint(
...                 0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
...             ).long()

...             # Add noise to the clean images according to the noise magnitude at each timestep
...             # (this is the forward diffusion process)
...             noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

...             with accelerator.accumulate(model):
...                 # Predict the noise residual
...                 noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
...                 loss = F.mse_loss(noise_pred, noise)
...                 accelerator.backward(loss)

...                 accelerator.clip_grad_norm_(model.parameters(), 1.0)
...                 optimizer.step()
...                 lr_scheduler.step()
...                 optimizer.zero_grad()

...             progress_bar.update(1)
...             logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
...             progress_bar.set_postfix(**logs)
...             accelerator.log(logs, step=global_step)
...             global_step += 1

...         # After each epoch you optionally sample some demo images with evaluate() and save the model
...         if accelerator.is_main_process:
...             pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

...             if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
...                 evaluate(config, epoch, pipeline)

...             if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
...                 if config.push_to_hub:
...                     upload_folder(
...                         repo_id=repo_id,
...                         folder_path=config.output_dir,
...                         commit_message=f"Epoch {epoch}",
...                         ignore_patterns=["step\_\*", "epoch\_\*"],
...                     )
...                 else:
...                     pipeline.save_pretrained(config.output_dir)
```


唷，那是相当多的代码！但您终于准备好通过 🤗 Accelerate 启动培训了
 `笔记本启动器`
 功能。向函数传递训练循环、所有训练参数和进程数（您可以将此值更改为可用的 GPU 数量）以用于训练：



```
>>> from accelerate import notebook_launcher

>>> args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

>>> notebook_launcher(train_loop, args, num_processes=1)
```


训练完成后，查看扩散模型生成的最终 🦋 图像 🦋！



```
>>> import glob

>>> sample_images = sorted(glob.glob(f"{config.output\_dir}/samples/\*.png"))
>>> Image.open(sample_images[-1])
```


![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/butterflies_final.png)


## 下一步



无条件图像生成是可训练任务的一个示例。您可以通过访问来探索其他任务和培训技术
 [🧨 扩散器培训示例](../培训/概述)
 页。以下是您可以学到的一些示例：


* [文本倒置](../training/text_inversion)
 ，一种向模型传授特定视觉概念并将其集成到生成图像中的算法。
* [DreamBooth](../培训/dreambooth)
 ，一种在给定对象的多个输入图像的情况下生成对象的个性化图像的技术。
* [指南](../training/text2image)
 在您自己的数据集上微调稳定扩散模型。
* [指南](../培训/lora)
 使用 LoRA，这是一种内存高效技术，可以更快地微调大型模型。