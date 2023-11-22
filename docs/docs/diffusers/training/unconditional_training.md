# 无条件图像生成

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/unconditional_training>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/unconditional_training>


与文本或图像到图像模型不同，无条件图像生成不以任何文本或图像为条件。它仅生成与其训练数据分布相似的图像。


 本指南将向您展示如何在现有数据集以及您自己的自定义数据集上训练无条件图像生成模型。所有无条件图像生成的训练脚本都可以找到
 [此处](https://github.com/huggingface/diffusers/tree/main/examples/unconditional_image_ Generation)
 如果您有兴趣了解更多有关培训的详细信息。


在运行脚本之前，请确保安装库的训练依赖项：



```
pip install diffusers[training] accelerate datasets
```


接下来，初始化一个🤗
 [加速](https://github.com/huggingface/accelerate/)
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


## 将模型上传到 Hub



您可以通过将以下参数添加到训练脚本来将模型上传到 Hub：



```
--push_to_hub
```


## 保存和加载检查点



定期保存检查点是一个好主意，以防训练期间发生任何情况。要保存检查点，请将以下参数传递给训练脚本：



```
--checkpointing_steps=500
```


完整的训练状态保存在
 `输出目录`
 每 500 步，这允许您加载一个检查点并在通过后恢复训练
 `--resume_from_checkpoint`
 训练脚本的参数：



```
--resume_from_checkpoint="checkpoint-1500"
```


## 微调



您已准备好启动
 [训练脚本](https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_ Generation/train_unconditional.py)
 现在！指定要微调的数据集名称
 `--数据集名称`
 参数，然后将其保存到路径中
 `--output_dir`
 。要使用您自己的数据集，请查看
 [创建训练数据集](create_dataset)
 指导。


训练脚本创建并保存
 `diffusion_pytorch_model.bin`
 文件在您的存储库中。


💡 在 4xV100 GPU 上完整的训练运行需要 2 小时。


例如，要微调
 [牛津花](https://huggingface.co/datasets/huggan/flowers-102-categories)
 数据集：



```
accelerate launch train_unconditional.py   --dataset_name="huggan/flowers-102-categories"   --resolution=64   --output_dir="ddpm-ema-flowers-64"   --train_batch_size=16   --num_epochs=100   --gradient_accumulation_steps=1   --learning_rate=1e-4   --lr_warmup_steps=500   --mixed_precision=no   --push_to_hub
```


![](https://user-images.githubusercontent.com/26864830/180248660-a0b143d0-b89a-42c5-8656-2ebf6ece7e52.png)


 或者，如果您想在
 [口袋妖怪](https://huggingface.co/datasets/huggan/pokemon)
 数据集：



```
accelerate launch train_unconditional.py   --dataset_name="huggan/pokemon"   --resolution=64   --output_dir="ddpm-ema-pokemon-64"   --train_batch_size=16   --num_epochs=100   --gradient_accumulation_steps=1   --learning_rate=1e-4   --lr_warmup_steps=500   --mixed_precision=no   --push_to_hub
```


![](https://user-images.githubusercontent.com/26864830/180248200-928953b4-db38-48db-b0c6-8b740fe6786f.png)

###


 使用多个 GPU 进行训练


‘加速’
 允许无缝的多 GPU 训练。按照说明操作
 [此处](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
 用于运行分布式训练
 ‘加速’
 。这是一个示例命令：



```
accelerate launch --mixed_precision="fp16" --multi_gpu train_unconditional.py   --dataset_name="huggan/pokemon"   --resolution=64 --center_crop --random_flip   --output_dir="ddpm-ema-pokemon-64"   --train_batch_size=16   --num_epochs=100   --gradient_accumulation_steps=1   --use_ema   --learning_rate=1e-4   --lr_warmup_steps=500   --mixed_precision="fp16"   --logger="wandb"   --push_to_hub
```