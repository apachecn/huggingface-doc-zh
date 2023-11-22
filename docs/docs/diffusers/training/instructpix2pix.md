# 指导Pix2Pix

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/instructpix2pix>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/instructpix2pix>


[InstructPix2Pix](https://arxiv.org/abs/2211.09800)
 是一种微调文本条件扩散模型的方法，以便它们可以遵循输入图像的编辑指令。使用此方法微调的模型将以下内容作为输入：


![instructpix2pix-inputs](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/edit-instruction.png)


输出是一个“编辑”图像，反映了应用于输入图像的编辑指令：


![instructpix2pix-output](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/output-gs%407-igs%401-steps%4050.png)


这
 `train_instruct_pix2pix.py`
 脚本（你可以找到它
 [此处](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py)
 ）展示了如何实施训练程序并使其适应稳定扩散。


***免责声明：尽管
 `train_instruct_pix2pix.py`
 实现 InstructPix2Pix
培训程序，同时忠实于
 [原始实现](https://github.com/timothybrooks/instruct-pix2pix)
 我们只在一个上测试过它
 [小规模数据集](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples)
 。这可能会影响最终结果。为了获得更好的结果，我们建议使用更大的数据集进行更长的训练。
 [此处](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered)
 您可以找到用于 InstructPix2Pix 训练的大型数据集。***


## 使用 PyTorch 在本地运行



### 


 安装依赖项


在运行脚本之前，请确保安装库的训练依赖项：


**重要的**


为了确保您可以成功运行示例脚本的最新版本，我们强烈建议
 **从源安装**
 并保持安装最新，因为我们经常更新示例脚本并安装一些特定于示例的要求。为此，请在新的虚拟环境中执行以下步骤：



```
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```


然后cd到示例文件夹中



```
cd examples/instruct_pix2pix
```


现在运行



```
pip install -r requirements.txt
```


并初始化一个
 [🤗加速](https://github.com/huggingface/accelerate/)
 环境：



```
accelerate config
```


或者使用默认加速配置而不回答有关您的环境的问题



```
accelerate config default
```


或者，如果您的环境不支持交互式 shell，例如笔记本



```
from accelerate.utils import write_basic_config

write_basic_config()
```


### 


 玩具示例


如前所述，我们将使用
 [小玩具数据集](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples)
 为了训练。数据集
是一个较小的版本
 [原始数据集](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered)
 InstructPix2Pix 论文中使用。要使用您自己的数据集，请查看
 [创建训练数据集](create_dataset)
 指导。


指定
 `型号_名称`
 环境变量（集线器模型存储库 ID 或包含模型权重的目录的路径）并将其传递给
 [`pretrained_model_name_or_path`](https://huggingface.co/docs/diffusers/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained.pretrained_model_name_or_path)
 争论。您还需要在中指定数据集名称
 `数据集_ID`
 ：



```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_ID="fusing/instructpix2pix-1000-samples"
```


现在，我们可以开始训练了。该脚本保存所有组件（
 `特征提取器`
 ,
 `调度程序`
 ,
 `文本编码器`
 ,
 `乌内特`
 等）在存储库的子文件夹中。



```
accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py     --pretrained_model_name_or_path=$MODEL\_NAME     --dataset_name=$DATASET\_ID     --enable_xformers_memory_efficient_attention     --resolution=256 --random_flip     --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing     --max_train_steps=15000     --checkpointing_steps=5000 --checkpoints_total_limit=1     --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0     --conditioning_dropout_prob=0.05     --mixed_precision=fp16     --seed=42     --push_to_hub
```


此外，我们支持执行验证推理来监控训练进度
具有权重和偏差。您可以通过以下方式启用此功能
 `report_to="wandb"`
 ：



```
accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py     --pretrained_model_name_or_path=$MODEL\_NAME     --dataset_name=$DATASET\_ID     --enable_xformers_memory_efficient_attention     --resolution=256 --random_flip     --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing     --max_train_steps=15000     --checkpointing_steps=5000 --checkpoints_total_limit=1     --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0     --conditioning_dropout_prob=0.05     --mixed_precision=fp16     --val_image_url="https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"     --validation_prompt="make the mountains snowy"     --seed=42     --report_to=wandb     --push_to_hub
```


我们推荐这种类型的验证，因为它对于模型调试很有用。请注意，您需要
 `万德布`
 安装使用这个。您可以安装
 `万德布`
 通过跑步
 `pip 安装wandb`
 。


[此处](https://wandb.ai/sayakpaul/instruct-pix2pix/runs/ctr3kovq)
 ，您可以找到一个示例训练运行，其中包括一些验证样本和训练超参数。


***注意：在原始论文中，作者观察到，即使使用 256x256 的图像分辨率训练模型，它也可以很好地推广到更大的分辨率，例如 512x512。这可能是因为他们在训练期间使用了更大的数据集。***


## 使用多个 GPU 进行训练



‘加速’
 允许无缝的多 GPU 训练。按照说明操作
 [此处](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
 用于运行分布式训练
 ‘加速’
 。这是一个示例命令：



```
accelerate launch --mixed_precision="fp16" --multi_gpu train_instruct_pix2pix.py  --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5  --dataset_name=sayakpaul/instructpix2pix-1000-samples  --use_ema  --enable_xformers_memory_efficient_attention  --resolution=512 --random_flip  --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing  --max_train_steps=15000  --checkpointing_steps=5000 --checkpoints_total_limit=1  --learning_rate=5e-05 --lr_warmup_steps=0  --conditioning_dropout_prob=0.05  --mixed_precision=fp16  --seed=42  --push_to_hub
```


## 推理



训练完成后，我们就可以进行推理：



```
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

model_id = "your\_model\_id"  # <- replace this
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
   model_id, torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test\_pix2pix\_4.png"


def download\_image(url):
   图像 = PIL.Image.open(requests.get(url,stream=True).raw)
   图像 = PIL.ImageOps.exif_t​​ranspose(图像)
   图像 = image.convert("RGB")
   返回图像


image = download_image(url)
prompt = "wipe out the lake"
num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

edited_image = pipe(
   prompt,
   image=image,
   num_inference_steps=num_inference_steps,
   image_guidance_scale=image_guidance_scale,
   guidance_scale=guidance_scale,
   generator=generator,
).images[0]
edited_image.save("edited\_image.png")
```


可以找到使用此训练脚本获得的示例模型存储库
这里 -
 [sayakpaul/instruct-pix2pix](https://huggingface.co/sayakpaul/instruct-pix2pix)
 。


我们鼓励您使用以下三个参数来控制
表演时的速度和质量：


* `num_inference_steps`
* `image_guidance_scale`
* `指导规模`


特别，
 `image_guidance_scale`
 和
 `指导规模`
 可以产生深远的影响
在生成的（“编辑的”）图像上（参见
 [此处](https://twitter.com/RisingSayak/status/1628392199196151808?s=20)
 举个例子）。


如果您正在寻找一些有趣的方法来使用 InstructPix2Pix 训练方法，我们欢迎您查看此博客文章：
 [使用 InstructPix2Pix 进行指令调整稳定扩散](https://huggingface.co/blog/instruction-tuning-sd)
 。


## 稳定扩散XL



训练与
 [稳定扩散XL](https://huggingface.co/papers/2307.01952)
 还通过以下方式支持
 `train_instruct_pix2pix_sdxl.py`
 脚本。请参考文档
 [此处](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/README_sdxl.md)
 。