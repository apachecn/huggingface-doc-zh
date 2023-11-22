# 控制网

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/controlnet>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/controlnet>


[向文本到图像扩散模型添加条件控制](https://arxiv.org/abs/2302.05543)
 (ControlNet) 作者：Lvmin 张和 Maneesh Agrawala。


这个例子是基于
 [原始 ControlNet 存储库中的训练示例](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md)
 。它使用以下方法训练 ControlNet 来填充圆圈
 [小型合成数据集](https://huggingface.co/datasets/fusing/fill50k)
 。


## 安装依赖项



在运行脚本之前，请确保安装库的训练依赖项。


要成功运行示例脚本的最新版本，我们强烈建议
 **从源安装**
 并使安装保持最新。我们经常更新示例脚本并安装特定于示例的要求。


为此，请在新的虚拟环境中执行以下步骤：



```
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```


然后导航到
 [示例文件夹](https://github.com/huggingface/diffusers/tree/main/examples/controlnet)



```
cd examples/controlnet
```


现在运行：



```
pip install -r requirements.txt
```


并初始化一个
 [🤗加速](https://github.com/huggingface/accelerate/)
 环境：



```
accelerate config
```


或者对于默认的🤗加速配置而不回答有关您的环境的问题：



```
accelerate config default
```


或者，如果您的环境不支持笔记本等交互式 shell：



```
from accelerate.utils import write_basic_config

write_basic_config()
```


## 圆形填充数据集



原始数据集托管在 ControlNet 中
 [回购](https://huggingface.co/lllyasviel/ControlNet/blob/main/training/fill50k.zip)
 ，但我们重新上传了它
 [此处](https://huggingface.co/datasets/fusing/fill50k)
 与🤗数据集兼容，以便它可以处理训练脚本中的数据加载。


我们的训练示例使用
 [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 因为这就是原始 ControlNet 模型集的训练内容。然而，ControlNet 可以经过训练来增强任何兼容的稳定扩散模型（例如
 [`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4)
 ） 或者
 [`stabilityai/stable-diffusion-2-1`](https://huggingface.co/stabilityai/stable-diffusion-2-1)
 。


要使用您自己的数据集，请查看
 [创建训练数据集](create_dataset)
 指导。


## 训练



下载以下图像来调节我们的训练：



```
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png

wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```


指定
 `型号_名称`
 环境变量（集线器模型存储库 ID 或包含模型权重的目录的路径）并将其传递给
 [`pretrained_model_name_or_path`](https://huggingface.co/docs/diffusers/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained.pretrained_model_name_or_path)
 争论。


训练脚本创建并保存
 `diffusion_pytorch_model.bin`
 文件在您的存储库中。



```
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path to save model"

accelerate launch train_controlnet.py  --pretrained_model_name_or_path=$MODEL\_DIR  --output_dir=$OUTPUT\_DIR  --dataset_name=fusing/fill50k  --resolution=512  --learning_rate=1e-5  --validation_image "./conditioning\_image\_1.png" "./conditioning\_image\_2.png"  --validation_prompt "red circle with blue background" "cyan circle with brown floral background"  --train_batch_size=4  --push_to_hub
```


此默认配置需要 ~38GB VRAM。


默认情况下，训练脚本将输出记录到张量板。经过
 `--report_to wandb`
 使用权重 &
偏见。


具有较小批量大小的梯度累积可用于将训练要求降低至约 20 GB VRAM。



```
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path to save model"

accelerate launch train_controlnet.py  --pretrained_model_name_or_path=$MODEL\_DIR  --output_dir=$OUTPUT\_DIR  --dataset_name=fusing/fill50k  --resolution=512  --learning_rate=1e-5  --validation_image "./conditioning\_image\_1.png" "./conditioning\_image\_2.png"  --validation_prompt "red circle with blue background" "cyan circle with brown floral background"  --train_batch_size=1  --gradient_accumulation_steps=4   --push_to_hub
```


## 使用多个 GPU 进行训练



‘加速’
 允许无缝的多 GPU 训练。按照说明操作
 [此处](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
 用于运行分布式训练
 ‘加速’
 。这是一个示例命令：



```
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path to save model"

accelerate launch --mixed_precision="fp16" --multi_gpu train_controlnet.py  --pretrained_model_name_or_path=$MODEL\_DIR  --output_dir=$OUTPUT\_DIR  --dataset_name=fusing/fill50k  --resolution=512  --learning_rate=1e-5  --validation_image "./conditioning\_image\_1.png" "./conditioning\_image\_2.png"  --validation_prompt "red circle with blue background" "cyan circle with brown floral background"  --train_batch_size=4  --mixed_precision="fp16"  --tracker_project_name="controlnet-demo"  --report_to=wandb   --push_to_hub
```


## 结果示例



#### 


 批量大小为 8 的 300 个步骤后


|  |  |
| --- | --- |
|  | 	 red circle with blue background	  |
| conditioning image | red circle with blue background |
|  | 	 cyan circle with brown floral background	  |
| conditioning image | cyan circle with brown floral background |


#### 


 批量大小为 8 的 6000 步后：


|  |  |
| --- | --- |
|  | 	 red circle with blue background	  |
| conditioning image | red circle with blue background |
|  | 	 cyan circle with brown floral background	  |
| conditioning image | cyan circle with brown floral background |


## 在 16 GB GPU 上训练



启用以下优化以在 16GB GPU 上进行训练：


* 梯度检查点
* bitsandbyte 的 8 位优化器（看一下[安装]((
 [https://github.com/TimDettmers/bitsandbytes#requirements—安装](https://github.com/TimDettmers/bitsandbytes#requirements--installation)
 ）说明（如果您尚未安装）


现在您可以启动训练脚本：



```
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path to save model"

accelerate launch train_controlnet.py  --pretrained_model_name_or_path=$MODEL\_DIR  --output_dir=$OUTPUT\_DIR  --dataset_name=fusing/fill50k  --resolution=512  --learning_rate=1e-5  --validation_image "./conditioning\_image\_1.png" "./conditioning\_image\_2.png"  --validation_prompt "red circle with blue background" "cyan circle with brown floral background"  --train_batch_size=1  --gradient_accumulation_steps=4  --gradient_checkpointing  --use_8bit_adam   --push_to_hub
```


## 在 12 GB GPU 上训练



启用以下优化以在 12GB GPU 上进行训练：


* 梯度检查点
* bitsandbyte 的 8 位优化器（看一下[安装]((
 [https://github.com/TimDettmers/bitsandbytes#requirements—安装](https://github.com/TimDettmers/bitsandbytes#requirements--installation)
 ）说明（如果您尚未安装）
* xFormers（看看
 [安装](https://huggingface.co/docs/diffusers/training/optimization/xformers)
 如果您尚未安装，请参阅说明）
* 设置渐变为
 `无`



```
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path to save model"

accelerate launch train_controlnet.py  --pretrained_model_name_or_path=$MODEL\_DIR  --output_dir=$OUTPUT\_DIR  --dataset_name=fusing/fill50k  --resolution=512  --learning_rate=1e-5  --validation_image "./conditioning\_image\_1.png" "./conditioning\_image\_2.png"  --validation_prompt "red circle with blue background" "cyan circle with brown floral background"  --train_batch_size=1  --gradient_accumulation_steps=4  --gradient_checkpointing  --use_8bit_adam  --enable_xformers_memory_efficient_attention  --set_grads_to_none   --push_to_hub
```


使用时
 `enable_xformers_memory_efficient_attention`
 ，请确保安装
 `xformers`
 经过
 `pip 安装 xformers`
 。


## 在 8 GB GPU 上训练



我们尚未详尽测试 DeepSpeed 对 ControlNet 的支持。虽然配置确实
节省内存，我们还没有确认配置训练是否成功。你很可能会
必须更改配置才能成功运行训练。


启用以下优化以在 8GB GPU 上进行训练：


* 梯度检查点
* bitsandbyte 的 8 位优化器（看一下[安装]((
 [https://github.com/TimDettmers/bitsandbytes#requirements—安装](https://github.com/TimDettmers/bitsandbytes#requirements--installation)
 ）说明（如果您尚未安装）
* xFormers（看看
 [安装](https://huggingface.co/docs/diffusers/training/optimization/xformers)
 如果您尚未安装，请参阅说明）
* 设置渐变为
 `无`
* DeepSpeed 第 2 阶段，具有参数和优化器卸载功能
* fp16混合精度


[DeepSpeed](https://www.deepspeed.ai/)
 可以将张量从 VRAM 卸载到
CPU 或 NVME。这需要更多的 RAM（大约 25 GB）。


您必须配置您的环境
 `加速配置`
 启用 DeepSpeed 第 2 阶段。


配置文件应如下所示：



```
compute\_environment: LOCAL\_MACHINE
deepspeed\_config:
  gradient\_accumulation\_steps: 4
  offload\_optimizer\_device: cpu
  offload\_param\_device: cpu
  zero3\_init\_flag: false
  zero\_stage: 2
distributed\_type: DEEPSPEED
```


看
 [文档](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
 了解更多 DeepSpeed 配置选项。


将默认 Adam 优化器更改为 DeepSpeed 的 Adam
 `deepspeed.ops.adam.DeepSpeedCPUAdam`
 给出了显着的加速但是
它需要与 PyTorch 版本相同的 CUDA 工具链。 8位优化器
目前似乎与 DeepSpeed 不兼容。



```
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path to save model"

accelerate launch train_controlnet.py  --pretrained_model_name_or_path=$MODEL\_DIR  --output_dir=$OUTPUT\_DIR  --dataset_name=fusing/fill50k  --resolution=512  --validation_image "./conditioning\_image\_1.png" "./conditioning\_image\_2.png"  --validation_prompt "red circle with blue background" "cyan circle with brown floral background"  --train_batch_size=1  --gradient_accumulation_steps=4  --gradient_checkpointing  --enable_xformers_memory_efficient_attention  --set_grads_to_none  --mixed_precision fp16  --push_to_hub
```


## 推理



经过训练的模型可以使用以下命令运行
 [StableDiffusionControlNetPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetPipeline)
 。
放
 `基本模型路径`
 和
 `控制网络路径`
 到价值观
 `--pretrained_model_name_or_path`
 和
 `--output_dir`
 分别在训练脚本中设置为。



```
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

base_model_path = "path to model"
controlnet_path = "path to controlnet"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

control_image = load_image("./conditioning\_image\_1.png")
prompt = "pale golden rod circle with old lace background"

# generate image
generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=20, generator=generator, image=control_image).images[0]

image.save("./output.png")
```


## 稳定扩散XL



训练与
 [稳定扩散XL](https://huggingface.co/papers/2307.01952)
 还通过以下方式支持
 `train_controlnet_sdxl.py`
 脚本。请参考文档
 [此处](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/README_sdxl.md)
 。