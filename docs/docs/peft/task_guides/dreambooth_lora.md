# DreamBooth 使用 LoRA 进行微调

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/task_guides/dreambooth_lora>
>
> 原始地址：<https://huggingface.co/docs/peft/task_guides/dreambooth_lora>


本指南演示了如何使用 LoRA（一种低秩逼近技术）对 DreamBooth 进行微调
 `CompVis/stable-diffusion-v1-4`
 模型。


尽管LoRA最初被设计为一种减少可训练参数数量的技术
大语言模型，该技术也可以应用于扩散模型。执行完整的模型微调
扩散模型的建立是一项耗时的任务，这就是为什么像 DreamBooth 或 Textual Inversion 这样的轻量级技术
获得了人气。随着 LoRA 的引入，在特定数据集上定制和微调模型已成为
甚至更快。


在本指南中，我们将使用 DreamBooth 微调脚本，该脚本可在
 [PEFT 的 GitHub 存储库](https://github.com/huggingface/peft/tree/main/examples/lora_dreambooth)
 。随意探索它并
了解事情是如何运作的。


## 设置您的环境



首先克隆 PEFT 存储库：



```
git clone https://github.com/huggingface/peft
```


导航到包含用于使用 LoRA 微调 Dreambooth 的训练脚本的目录：



```
cd peft/examples/lora_dreambooth
```


设置您的环境：安装 PEFT 和所有必需的库。在撰写本指南时，我们建议
从源安装 PEFT。



```
pip install -r requirements.txt
pip install git+https://github.com/huggingface/peft
```


## 微调 DreamBooth



准备用于微调模型的图像。设置几个环境变量：



```
export MODEL_NAME="CompVis/stable-diffusion-v1-4" 
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"
```


这里：


* `实例目录`
 ：包含您打算用于训练模型的图像的目录。
* `CLASS_DIR`
 ：包含特定于类的图像的目录。在这个例子中，我们使用预先保存来避免过度拟合和语言漂移。为了预先保存，您需要同一类的其他图像作为训练过程的一部分。但是，可以生成这些图像，并且训练脚本会将它们保存到您在此处指定的本地路径。
* `输出_目录`
 ：用于存储训练模型权重的目标文件夹。


要了解有关 DreamBooth 通过保留先验损失进行微调的更多信息，请查看
 [Diffusers 文档](https://huggingface.co/docs/diffusers/training/dreambooth#finetuning-with-priorpreserving-loss)
 。


启动训练脚本
 ‘加速’
 并向其传递超参数以及 LoRa 特定的参数，例如：


* `use_lora`
 ：在训练脚本中启用 LoRa。
* `洛拉_r`
 ：LoRA 更新矩阵使用的维度。
* `lora_alpha`
 ：比例因子。
* `lora_text_encoder_r`
 ：文本编码器的 LoRA 排名。
* `lora_text_encoder_alpha`
 ：文本编码器的 LoRA alpha（缩放因子）。


完整的脚本参数集可能如下所示：



```
accelerate launch train_dreambooth.py   --pretrained_model_name_or_path=$MODEL\_NAME    --instance_data_dir=$INSTANCE\_DIR   --class_data_dir=$CLASS\_DIR   --output_dir=$OUTPUT\_DIR   --train_text_encoder   --with_prior_preservation --prior_loss_weight=1.0   --num_dataloader_workers=1   --instance_prompt="a photo of sks dog"   --class_prompt="a photo of dog"   --resolution=512   --train_batch_size=1   --lr_scheduler="constant"   --lr_warmup_steps=0   --num_class_images=200   --use_lora   --lora_r 16   --lora_alpha 27   --lora_text_encoder_r 16   --lora_text_encoder_alpha 17   --learning_rate=1e-4   --gradient_accumulation_steps=1   --gradient_checkpointing   --max_train_steps=800
```


如果您在 Windows 上运行此脚本，您可能需要设置
 `--num_dataloader_workers`
 至 0。


## 使用单个适配器进行推理



要使用微调模型运行推理，首先指定将与微调 LoRA 权重相结合的基本模型：



```
import os
import torch

from diffusers import StableDiffusionPipeline
from peft import PeftModel, LoraConfig

MODEL_NAME = "CompVis/stable-diffusion-v1-4"
```


接下来，添加一个函数来创建用于图像生成的稳定扩散管道。它将结合以下的权重
使用微调 LoRA 权重的基本模型
 `LoraConfig`
 。



```
def get\_lora\_sd\_pipeline(
 ckpt\_dir, base\_model\_name\_or\_path=None, dtype=torch.float16, device="cuda", adapter\_name="default"
):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text\_encoder")
    if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path

    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")

    pipe = StableDiffusionPipeline.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
        )

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe
```


现在，您可以使用上面的函数，使用您在微调步骤中创建的 LoRA 权重来创建稳定扩​​散管道。
   

 请注意，如果您在同一台计算机上运行推理，则此处指定的路径将与
 `输出目录`
 。



```
pipe = get_lora_sd_pipeline(Path("path-to-saved-model"), adapter_name="dog")
```


一旦有了带有微调模型的管道，您就可以使用它来生成图像：



```
prompt = "sks dog playing fetch in the park"
negative_prompt = "low quality, blurry, unfinished"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7, negative_prompt=negative_prompt).images[0]
image.save("DESTINATION\_PATH\_FOR\_THE\_IMAGE")
```


![生成的公园里的狗的图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_dreambooth_dog_park.png)


## 多适配器推理



使用 PEFT，您可以组合多个适配器进行推理。在前面的示例中，您对稳定扩散进行了微调
一些狗的图像。基于这些权重创建的管道得到了一个名字 -
 `adapter_name="狗"`
 。现在，假设您还进行了微调
这个基础模型基于钩针玩具的图像。让我们看看如何使用这两个适配器。


首先，您需要执行单个适配器推理示例中的所有步骤：


1. 指定基础型号。
2. 添加一个函数，使用 LoRA 权重创建用于图像生成的稳定扩散管道。
3. 创建一个
 `管道`
 和
 `adapter_name="狗"`
 基于对狗图像进行微调的模型。


接下来，您将需要更多辅助函数。
要加载另一个适配器，请创建一个
 `load_adapter()`
 杠杆作用的函数
 `load_adapter()`
 的方法
 `佩夫特模型`
 （例如。
 `pipe.unet.load_adapter(peft_model_path, 适配器名称)`
 ）：



```
def load\_adapter(pipe, ckpt\_dir, adapter\_name):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text\_encoder")
    pipe.unet.load_adapter(unet_sub_dir, adapter_name=adapter_name)
    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder.load_adapter(text_encoder_sub_dir, adapter_name=adapter_name)
```


要在适配器之间切换，请编写一个使用以下函数的函数
 `set_adapter()`
 的方法
 `佩夫特模型`
 （看
 `pipe.unet.set_adapter（适配器名称）`
 ）



```
def set\_adapter(pipe, adapter\_name):
    pipe.unet.set_adapter(adapter_name)
    if isinstance(pipe.text_encoder, PeftModel):
        pipe.text_encoder.set_adapter(adapter_name)
```


最后，添加一个函数来创建加权 LoRA 适配器。



```
def create\_weighted\_lora\_adapter(pipe, adapters, weights, adapter\_name="default"):
    pipe.unet.add_weighted_adapter(adapters, weights, adapter_name)
    if isinstance(pipe.text_encoder, PeftModel):
        pipe.text_encoder.add_weighted_adapter(adapters, weights, adapter_name)

    return pipe
```


让我们加载在钩针玩具图像上微调的模型中的第二个适配器，并为其指定一个唯一的名称：



```
load_adapter(pipe, Path("path-to-the-second-saved-model"), adapter_name="crochet")
```


使用加权适配器创建管道：



```
pipe = create_weighted_lora_adapter(pipe, ["crochet", "dog"], [1.0, 1.05], adapter_name="crochet\_dog")
```


现在您可以在适配器之间切换。如果您想生成更多狗图像，请将适配器设置为
 「狗」
 ：



```
set_adapter(pipe, adapter_name="dog")
prompt = "sks dog in a supermarket isle"
negative_prompt = "low quality, blurry, unfinished"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7, negative_prompt=negative_prompt).images[0]
image
```


![生成的超市里的狗的图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_dreambooth_dog_supermarket.png)


 以同样的方式，您可以切换到第二个适配器：



```
set_adapter(pipe, adapter_name="crochet")
prompt = "a fish rendered in the style of <1>"
negative_prompt = "low quality, blurry, unfinished"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7, negative_prompt=negative_prompt).images[0]
image
```


![生成的钩针鱼图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_dreambooth_fish.png)


 最后，您可以使用组合加权适配器：



```
set_adapter(pipe, adapter_name="crochet\_dog")
prompt = "sks dog rendered in the style of <1>, close up portrait, 4K HD"
negative_prompt = "low quality, blurry, unfinished"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7, negative_prompt=negative_prompt).images[0]
image
```


![生成的钩针狗图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_dreambooth_crochet_dog.png)