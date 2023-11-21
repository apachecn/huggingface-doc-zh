# 自定义扩散训练示例

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/custom_diffusion>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/custom_diffusion>


[自定义扩散](https://arxiv.org/abs/2212.04488)
 是一种自定义文本到图像模型的方法，例如仅给定主题的几张（4~5）张图像的稳定扩散。
这
 `train_custom_diffusion.py`
 脚本展示了如何实施训练程序并对其进行调整以实现稳定扩散。


此训练示例由以下人员贡献
 [努普尔库玛丽](https://nupurkmr9.github.io/)
 （《自定义扩散》的作者之一）。


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
 [示例文件夹](https://github.com/huggingface/diffusers/tree/main/examples/custom_diffusion)



```
cd examples/custom_diffusion
```


现在运行



```
pip install -r requirements.txt
pip install clip-retrieval 
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


 猫的例子😺


现在让我们获取数据集。从下载数据集
 [此处](https://www.cs.cmu.edu/~custom-diffusion/assets/data.zip)
 并解压它。要使用您自己的数据集，请查看
 [创建训练数据集](create_dataset)
 指导。


我们还使用以下方法收集了 200 张真实图像
 `剪辑检索`
 与训练数据集中的目标图像组合作为正则化。这可以防止过度拟合给定的目标图像。以下标志启用正则化
 `with_prior_preservation`
 ,
 `真实先验`
 和
 `prior_loss_weight=1。`
 。
这
 `类提示`
 类别名称应与目标图像相同。收集到的真实图像的文字说明类似于
 `类提示`
 。检索到的图像保存在
 `类数据目录`
 。您可以禁用
 `真实先验`
 使用生成的图像作为正则化。要收集真实图像，请在训练之前先使用此命令。



```
pip install clip-retrieval
python retrieve.py --class_prompt cat --class_data_dir real_reg/samples_cat --num_class_images 200
```


*****注意：更改
 `决议`
 如果您使用的是 768
 [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2)
 768x768 模型。*****


该脚本创建并保存模型检查点和
 `pytorch_custom_diffusion_weights.bin`
 文件在您的存储库中。



```
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"
export INSTANCE_DIR="./data/cat"

accelerate launch train_custom_diffusion.py   --pretrained_model_name_or_path=$MODEL\_NAME    --instance_data_dir=$INSTANCE\_DIR   --output_dir=$OUTPUT\_DIR   --class_data_dir=./real_reg/samples_cat/   --with_prior_preservation --real_prior --prior_loss_weight=1.0   --class_prompt="cat" --num_class_images=200   --instance_prompt="photo of a <new1> cat"    --resolution=512    --train_batch_size=2    --learning_rate=1e-5    --lr_warmup_steps=0   --max_train_steps=250   --scale_lr --hflip    --modifier_token "<new1>"   --push_to_hub
```


**使用
 `--enable_xformers_memory_efficient_attention`
 以较低的 VRAM 要求（每个 GPU 16GB）实现更快的训练。跟随
 [本指南](https://github.com/facebookresearch/xformers)
 有关安装说明。**


使用权重和偏差跟踪您的实验（
 `万德布`
 ）并保存中间结果（我们强烈推荐），请按照以下步骤操作：


* 安装
 `万德布`
 :
 `pip 安装wandb`
 。
* 授权：
 `wandb登录`
 。
* 然后指定一个
 `验证提示`
 并设置
 `报告对象`
 到
 `万德布`
 在开展培训的同时。您还可以配置以下相关参数：
+ `num_validation_images`
+ `验证步骤`


这是一个示例命令：



```
accelerate launch train_custom_diffusion.py   --pretrained_model_name_or_path=$MODEL\_NAME    --instance_data_dir=$INSTANCE\_DIR   --output_dir=$OUTPUT\_DIR   --class_data_dir=./real_reg/samples_cat/   --with_prior_preservation --real_prior --prior_loss_weight=1.0   --class_prompt="cat" --num_class_images=200   --instance_prompt="photo of a <new1> cat"    --resolution=512    --train_batch_size=2    --learning_rate=1e-5    --lr_warmup_steps=0   --max_train_steps=250   --scale_lr --hflip    --modifier_token "<new1>"   --validation_prompt="<new1> cat sitting in a bucket"   --report_to="wandb"   --push_to_hub
```


这是一个例子
 [权重和偏差页面](https://wandb.ai/sayakpaul/custom-diffusion/runs/26ghrcau)
 您可以在其中查看中间结果以及其他培训详细信息。


如果您指定
 `--push_to_hub`
 ，学习到的参数将被推送到 Hugging Face Hub 上的存储库。这是一个
 [示例存储库](https://huggingface.co/sayakpaul/custom-diffusion-cat)
 。


### 


 多个概念的培训🐱🪵


提供一个
 [json](https://github.com/adobe-research/custom-diffusion/blob/main/assets/concept_list.json)
 包含每个概念信息的文件，类似于
 [这个](https://github.com/ShivamShrrirao/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)
 。


要收集真实图像，请为 json 文件中的每个概念运行此命令。



```
pip install clip-retrieval
python retrieve.py --class_prompt {} --class_data_dir {} --num_class_images 200
```


然后我们准备开始训练！



```
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_custom_diffusion.py   --pretrained_model_name_or_path=$MODEL\_NAME    --output_dir=$OUTPUT\_DIR   --concepts_list=./concept_list.json   --with_prior_preservation --real_prior --prior_loss_weight=1.0   --resolution=512    --train_batch_size=2    --learning_rate=1e-5    --lr_warmup_steps=0   --max_train_steps=500   --num_class_images=200   --scale_lr --hflip    --modifier_token "<new1>+<new2>"   --push_to_hub
```


这是一个例子
 [权重和偏差页面](https://wandb.ai/sayakpaul/custom-diffusion/runs/3990tzkg)
 您可以在其中查看中间结果以及其他培训详细信息。


### 


 人脸训练


为了对人脸进行微调，我们发现以下配置效果更好：
 `学习率=5e-6`
 ,
 `max_train_steps=1000 到 2000`
 ， 和
 `freeze_model=crossattn`
 至少有 15-20 张图片。


要收集真实图像，请在训练之前先使用此命令。



```
pip install clip-retrieval
python retrieve.py --class_prompt person --class_data_dir real_reg/samples_person --num_class_images 200
```


然后开始训练吧！



```
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"
export INSTANCE_DIR="path-to-images"

accelerate launch train_custom_diffusion.py   --pretrained_model_name_or_path=$MODEL\_NAME    --instance_data_dir=$INSTANCE\_DIR   --output_dir=$OUTPUT\_DIR   --class_data_dir=./real_reg/samples_person/   --with_prior_preservation --real_prior --prior_loss_weight=1.0   --class_prompt="person" --num_class_images=200   --instance_prompt="photo of a <new1> person"    --resolution=512    --train_batch_size=2    --learning_rate=5e-6    --lr_warmup_steps=0   --max_train_steps=1000   --scale_lr --hflip --noaug   --freeze_model crossattn   --modifier_token "<new1>"   --enable_xformers_memory_efficient_attention   --push_to_hub
```


## 推理



使用上述命令训练模型后，您可以使用以下命令运行推理。确保包括
 `修饰符标记`
 （例如上例中的 \<new1>）在提示符中。



```
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
pipe.unet.load_attn_procs("path-to-save-model", weight_name="pytorch\_custom\_diffusion\_weights.bin")
pipe.load_textual_inversion("path-to-save-model", weight_name="<new1>.bin")

image = pipe(
    "<new1> cat sitting in a bucket",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("cat.png")
```


可以直接从 Hub 存储库加载这些参数：



```
import torch
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline

model_id = "sayakpaul/custom-diffusion-cat"
card = RepoCard.load(model_id)
base_model_id = card.data.to_dict()["base\_model"]

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipe.unet.load_attn_procs(model_id, weight_name="pytorch\_custom\_diffusion\_weights.bin")
pipe.load_textual_inversion(model_id, weight_name="<new1>.bin")

image = pipe(
    "<new1> cat sitting in a bucket",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("cat.png")
```


以下是使用多个概念进行推理的示例：



```
import torch
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline

model_id = "sayakpaul/custom-diffusion-cat-wooden-pot"
card = RepoCard.load(model_id)
base_model_id = card.data.to_dict()["base\_model"]

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipe.unet.load_attn_procs(model_id, weight_name="pytorch\_custom\_diffusion\_weights.bin")
pipe.load_textual_inversion(model_id, weight_name="<new1>.bin")
pipe.load_textual_inversion(model_id, weight_name="<new2>.bin")

image = pipe(
    "the <new1> cat sculpture in the style of a <new2> wooden pot",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("multi-subject.png")
```


这里，
 `猫`
 和
 「木锅」
 参考多个概念。


### 


 从训练检查点推断


如果您使用了训练过程中保存的完整检查点之一，您还可以执行推理
 `--checkpointing_steps`
 争论。


全部。


## 将成绩设置为无



为了节省更多内存，请传递
 `--set_grads_to_none`
 脚本的参数。这会将 grads 设置为 None 而不是零。但是，请注意它会改变某些行为，因此如果您开始遇到任何问题，请删除此参数。


更多信息：
 <https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html>


## 实验结果



您可以参考
 [我们的网页](https://www.cs.cmu.edu/~custom-diffusion/)
 详细讨论了我们的实验。