# 用于稳定扩散 XL (SDXL) 的 T2I 适配器

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/t2i_adapters>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/t2i_adapters>


这
 `train_t2i_adapter_sdxl.py`
 脚本（如下所示）展示了如何实现
 [T2I-Adapter 培训流程](https://hf.co/papers/2302.08453)
 为了
 [稳定扩散XL](https://huggingface.co/papers/2307.01952)
 。


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


然后cd到
 `示例/t2i_adapter`
 文件夹并运行



```
pip install -r requirements_sdxl.txt
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


或者，如果您的环境不支持交互式 shell（例如笔记本）



```
from accelerate.utils import write_basic_config
write_basic_config()
```


跑步时
 `加速配置`
 ，如果我们将 torch 编译模式指定为 True，则可以显着提高速度。


## 圆形填充数据集



原始数据集托管在
 [ControlNet 存储库](https://huggingface.co/lllyasviel/ControlNet/blob/main/training/fill50k.zip)
 。我们重新上传它以兼容
 `数据集`
[此处](https://huggingface.co/datasets/fusing/fill50k)
 。注意
 `数据集`
 处理训练脚本中的数据加载。


## 训练



我们的训练示例使用两个测试条件图像。可以通过运行来下载它们



```
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png

wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```


然后运行
 `huggingface-cli 登录`
 登录您的 Hugging Face 帐户。这是为了能够将经过训练的 T2IAdapter 参数推送到 Hugging Face Hub 所必需的。



```
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="path to save model"

accelerate launch train_t2i_adapter_sdxl.py  --pretrained_model_name_or_path=$MODEL\_DIR  --output_dir=$OUTPUT\_DIR  --dataset_name=fusing/fill50k  --mixed_precision="fp16"  --resolution=1024  --learning_rate=1e-5  --max_train_steps=15000  --validation_image "./conditioning\_image\_1.png" "./conditioning\_image\_2.png"  --validation_prompt "red circle with blue background" "cyan circle with brown floral background"  --validation_steps=100  --train_batch_size=1  --gradient_accumulation_steps=4  --report_to="wandb"  --seed=42  --push_to_hub
```


为了更好地跟踪我们的训练实验，我们在上面的命令中使用以下标志：


* `report_to="wandb`
 将确保跟踪训练运行的权重和偏差。要使用它，请务必安装
 `万德布`
 和
 `pip 安装wandb`
 。
* `验证图像`
 ,
 `验证提示`
 ， 和
 `验证步骤`
 允许脚本执行一些验证推理运行。这使我们能够定性地检查培训是否按预期进行。


我们的实验是在单个 40GB A100 GPU 上进行的。


### 


 推理


训练完成后，我们可以像这样进行推理：



```
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteSchedulerTest
from diffusers.utils import load_image
import torch

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
adapter_path = "path to adapter"

adapter = T2IAdapter.from_pretrained(adapter_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    base_model_path, adapter=adapter, torch_dtype=torch.float16
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = EulerAncestralDiscreteSchedulerTest.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

control_image = load_image("./conditioning\_image\_1.png")
prompt = "pale golden rod circle with old lace background"

# generate image
generator = torch.manual_seed(0)
image = pipe(
    prompt, num_inference_steps=20, generator=generator, image=control_image
).images[0]
image.save("./output.png")
```


## 笔记



### 


 指定更好的 VAE


众所周知，SDXL 的 VAE 存在数值不稳定问题。这就是为什么我们还公开一个 CLI 参数，即
 `--pretrained_vae_model_name_or_path`
 允许您指定更好的 VAE 的位置（例如
 [这个](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
 ）。