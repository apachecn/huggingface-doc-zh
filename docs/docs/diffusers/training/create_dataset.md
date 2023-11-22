# 创建用于训练的数据集

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/create_dataset>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/create_dataset>


网上有很多数据集
 [中心](https://huggingface.co/datasets?task_categories=task_categories:text-to-image&sort=downloads)
 来训练模型，但如果您找不到您感兴趣的模型或想要使用自己的模型，您可以使用 🤗 创建一个数据集
 [数据集](hf.co/docs/datasets)
 图书馆。数据集结构取决于您要训练模型的任务。最基本的数据集结构是用于无条件图像生成等任务的图像目录。另一种数据集结构可能是图像目录和文本文件，其中包含用于文本到图像生成等任务的相应文本标题。


本指南将向您展示两种创建数据集进行微调的方法：


* 提供一个图像文件夹
 `--train_data_dir`
 争论
* 将数据集上传到Hub并将数据集存储库id传递给
 `--数据集名称`
 争论


💡 了解有关如何创建用于训练的图像数据集的更多信息
 [创建图像数据集](https://huggingface.co/docs/datasets/image_dataset)
 指导。


## 提供数据集作为文件夹



对于无条件生成，您可以提供自己的数据集作为图像文件夹。训练脚本使用
 [`ImageFolder`](https://huggingface.co/docs/datasets/en/image_dataset#imagefolder)
 🤗 数据集的构建器可自动从文件夹构建数据集。您的目录结构应如下所示：



```
data_dir/xxx.png
data_dir/xxy.png
data_dir/[...]/xxz.png
```


将数据集目录的路径传递给
 `--train_data_dir`
 论证，然后就可以开始训练了：



```
accelerate launch train_unconditional.py     --train_data_dir <path-to-train-directory>     <other-arguments>
```


## 将您的数据上传到 Hub



💡 有关创建数据集并将其上传到中心的更多详细信息和上下文，请查看
 [使用🤗数据集进行图像搜索](https://huggingface.co/blog/image-search-datasets)
 邮政。


首先创建一个数据集
 [`ImageFolder`](https://huggingface.co/docs/datasets/image_load#imagefolder)
 功能，这创建了一个
 `图像`
 包含 PIL 编码图像的列。


您可以使用
 `数据目录`
 或者
 `数据文件`
 参数来指定数据集的位置。这
 `数据文件`
 参数支持将特定文件映射到数据集分割，例如
 `火车`
 或者
 `测试`
 ：



```
from datasets import load_dataset

# example 1: local folder
dataset = load_dataset("imagefolder", data_dir="path\_to\_your\_folder")

# example 2: local files (supported formats are tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset("imagefolder", data_files="path\_to\_zip\_file")

# example 3: remote files (supported formats are tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset(
    "imagefolder",
    data_files="https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs\_3367a.zip",
)

# example 4: providing several splits
dataset = load_dataset(
    "imagefolder", data_files={"train": ["path/to/file1", "path/to/file2"], "test": ["path/to/file3", "path/to/file4"]}
)
```


然后使用
 `推送到集线器`
 将数据集上传到Hub的方法：



```
# assuming you have ran the huggingface-cli login command in a terminal
dataset.push_to_hub("name\_of\_your\_dataset")

# if you want to push to a private repo, simply pass private=True:
dataset.push_to_hub("name\_of\_your\_dataset", private=True)
```


现在，通过将数据集名称传递给
 `--数据集名称`
 争论：



```
accelerate launch --mixed_precision="fp16"  train_text_to_image.py   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"   --dataset_name="name\_of\_your\_dataset"   <other-arguments>
```


## 下一步



现在您已经创建了一个数据集，您可以将其插入到
 `train_data_dir`
 （如果您的数据集是本地的）或
 `数据集名称`
 （如果您的数据集位于 Hub 上）训练脚本的参数。


对于接下来的步骤，请随意尝试使用您的数据集来训练模型
 [无条件生成](unconditional_training)
 或者
 [文本转图像生成](text2image)
 ！