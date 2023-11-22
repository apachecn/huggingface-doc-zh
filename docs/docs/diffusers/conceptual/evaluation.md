# 评估扩散模型

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/conceptual/evaluation>
>
> 原始地址：<https://huggingface.co/docs/diffusers/conceptual/evaluation>


[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/主/扩散器/evaluation.ipynb)

 生成模型的评估，例如
 [稳定扩散](https://huggingface.co/docs/diffusers/stable_diffusion)
 本质上是主观的。但作为实践者和研究人员，我们常常必须在许多不同的可能性中做出谨慎的选择。那么，当使用不同的生成模型（如 GAN、Diffusion 等）时，我们如何选择一种模型？


对此类模型的定性评估可能容易出错，并且可能会错误地影响决策。
然而，定量指标不一定对应于图像质量。所以，通常情况下，组合
选择一种模型时，定性和定量评估提供了更强的信号
超过另一个。


在本文档中，我们提供了评估扩散模型的定性和定量方法的非详尽概述。对于定量方法，我们特别关注如何将它们与
 `扩散器`
 。


本文档中所示的方法也可用于评估不同的
 [噪声调度程序](https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview)
 保持底层生成模型固定。


## 应用场景



我们使用以下管道涵盖扩散模型：


* 文本引导图像生成（例如
 [`StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/text2img)
 ）。
* 文本引导图像生成，另外以输入图像为条件（例如
 [`StableDiffusionImg2ImgPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/img2img)
 和
 [`StableDiffusionInstructPix2PixPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pix2pix)
 ）。
* 类条件图像生成模型（例如
 [`DiTPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/dit)
 ）。


## 定性评价



定性评估通常涉及对生成图像的人工评估。质量是通过构图、图像文本对齐和空间关系等方面来衡量的。常见提示为主观指标提供了一定程度的一致性。
DrawBench 和 PartiPrompts 是用于定性基准测试的提示数据集。 DrawBench 和 PartiPrompts 是由
 [图像](https://imagen.research.google/)
 和
 [Parti](https://parti.research.google/)
 分别。


来自
 [Parti官方网站](https://parti.research.google/)
 :


>
>
> PartiPrompts (P2) 是一套丰富的英语提示，包含 1600 多个提示，我们作为这项工作的一部分发布。 P2 可用于衡量跨各种类别和挑战方面的模型能力。
>
>
>
>


![parti-prompts](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/parti-prompts.png)


PartiPrompts 具有以下列：


* 迅速的
* 提示的类别（如“摘要”、“世界知识”等）
* 反映难度的挑战（如“基本”、“复杂”、“书写和符号”等）


这些基准允许对不同图像生成模型进行并行评估。


为此，🧨 Diffusers 团队构建了
 **打开零件提示**
 ，这是一个基于 Parti Prompts 的社区驱动的定性基准，用于比较最先进的开源扩散模型：


* [打开 Parti 提示游戏](https://huggingface.co/spaces/OpenGenAI/open-parti-prompts)
 ：对于 10 部分提示，显示 4 个生成的图像，用户选择最适合提示的图像。
* [开放 Parti 提示排行榜](https://huggingface.co/spaces/OpenGenAI/parti-prompts-leaderboard)
 ：排行榜比较了当前最好的开源扩散模型。


要手动比较图像，让我们看看如何使用
 `扩散器`
 在几个 PartiPrompt 上。


下面我们展示了针对不同挑战的一些提示：基本、复杂、语言结构、想象力以及写作和符号。这里我们使用 PartiPrompts 作为
 [数据集](https://huggingface.co/datasets/nateraw/parti-prompts)
 。



```
from datasets import load_dataset

# prompts = load\_dataset("nateraw/parti-prompts", split="train")
# prompts = prompts.shuffle()
# sample\_prompts = [prompts[i]["Prompt"] for i in range(5)]

# Fixing these sample prompts in the interest of reproducibility.
sample_prompts = [
    "a corgi",
    "a hot air balloon with a yin-yang symbol, with the moon visible in the daytime sky",
    "a car with no windows",
    "a cube made of porcupine",
    'The saying "BE EXCELLENT TO EACH OTHER" written on a red brick wall with a graffiti image of a green alien wearing a tuxedo. A yellow fire hydrant is on a sidewalk in the foreground.',
]
```


现在我们可以使用这些提示来使用稳定扩散生成一些图像（
 [v1-4 检查点](https://huggingface.co/CompVis/stable-diffusion-v1-4)
 ）：



```
import torch

seed = 0
generator = torch.manual_seed(seed)

images = sd_pipeline(sample_prompts, num_images_per_prompt=1, generator=generator).images
```


![parti-prompts-14](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/parti-prompts-14.png)


我们还可以设置
 `num_images_per_prompt`
 相应地比较相同提示的不同图像。运行相同的管道但使用不同的检查点（
 [v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 ），产量：


![parti-prompts-15](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/parti-prompts-15.png)


一旦使用多个模型（正在评估）根据所有提示生成了多个图像，这些结果就会呈现给人类评估者进行评分。为了
有关 DrawBench 和 PartiPrompts 基准的更多详细信息，请参阅各自的论文。


在训练模型来测量模型时，查看一些推理样本很有用。
培训进度。在我们的
 [训练脚本](https://github.com/huggingface/diffusers/tree/main/examples/)
 ，我们支持此实用程序并提供额外支持
记录到 TensorBoard 以及权重和偏差。


## 定量评价



在本节中，我们将引导您了解如何使用以下方法评估三种不同的扩散管道：


* CLIP 分数
* CLIP方向相似度
* FID


### 


 文本引导图像生成


[CLIP评分](https://arxiv.org/abs/2104.08718)
 测量图像标题对的兼容性。较高的 CLIP 分数意味着较高的兼容性🔼。 CLIP 分数是定性概念“兼容性”的定量测量。图像-标题对兼容性也可以被认为是图像和标题之间的语义相似性。研究发现 CLIP 评分与人类判断具有高度相关性。


我们先加载一个
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 :



```
from diffusers import StableDiffusionPipeline
import torch

model_ckpt = "CompVis/stable-diffusion-v1-4"
sd_pipeline = StableDiffusionPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16).to("cuda")
```


生成一些带有多个提示的图像：



```
prompts = [
    "a photo of an astronaut riding a horse on mars",
    "A high tech solarpunk utopia in the Amazon rainforest",
    "A pikachu fine dining with a view to the Eiffel Tower",
    "A mecha robot in a favela in expressionist style",
    "an insect robot preparing a delicious meal",
    "A small cabin on top of a snowy mountain in the style of Disney, artstation",
]

images = sd_pipeline(prompts, num_images_per_prompt=1, output_type="np").images

print(images.shape)
# (6, 512, 512, 3)
```


然后，我们计算 CLIP 分数。



```
from torchmetrics.functional.multimodal import clip_score
from functools import partial

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate\_clip\_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

sd_clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {sd\_clip\_score}")
# CLIP score: 35.7038
```


在上面的示例中，我们为每个提示生成一张图像。如果我们每个提示生成多个图像，我们就必须从每个提示生成的图像中获取平均分数。


现在，如果我们想比较两个兼容的检查点
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)
 我们应该在调用管道时传递一个生成器。首先，我们生成图像
固定种子
 [v1-4 稳定扩散检查点](https://huggingface.co/CompVis/stable-diffusion-v1-4)
 :



```
seed = 0
generator = torch.manual_seed(seed)

images = sd_pipeline(prompts, num_images_per_prompt=1, generator=generator, output_type="np").images
```


然后我们加载
 [v1-5 检查点](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 生成图像：



```
model_ckpt_1_5 = "runwayml/stable-diffusion-v1-5"
sd_pipeline_1_5 = StableDiffusionPipeline.from_pretrained(model_ckpt_1_5, torch_dtype=weight_dtype).to(device)

images_1_5 = sd_pipeline_1_5(prompts, num_images_per_prompt=1, generator=generator, output_type="np").images
```


最后，我们比较他们的 CLIP 分数：



```
sd_clip_score_1_4 = calculate_clip_score(images, prompts)
print(f"CLIP Score with v-1-4: {sd\_clip\_score\_1\_4}")
# CLIP Score with v-1-4: 34.9102

sd_clip_score_1_5 = calculate_clip_score(images_1_5, prompts)
print(f"CLIP Score with v-1-5: {sd\_clip\_score\_1\_5}")
# CLIP Score with v-1-5: 36.2137
```


似乎是
 [v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 检查点的性能比其前身更好。但请注意，我们用于计算 CLIP 分数的提示数量非常少。对于更实际的评估，这个数字应该更高，并且提示应该多样化。


从结构上看，这个分数存在一些局限性。训练数据集中的标题
从网络上抓取并提取自
 `替代`
 和类似的标签关联了互联网上的图像。
它们不一定代表人类用来描述图像的内容。因此我们
必须在这里“设计”一些提示。



### 


 图像条件文本到图像的生成


在这种情况下，我们使用输入图像和文本提示来调节生成管道。让我们以
 [StableDiffusionInstructPix2PixPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/pix2pix#diffusers.StableDiffusionInstructPix2PixPipeline)
 ， 举个例子。它以编辑指令作为输入提示，并输入待编辑的图像。


这是一个例子：


![编辑指令](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/edit-instruction.png)


评估此类模型的一种策略是测量两个图像之间变化的一致性（在
 [剪辑](https://huggingface.co/docs/transformers/model_doc/clip)
 space）随着两个图像标题之间的变化（如图所示
 [CLIP引导的图像生成器域适应](https://arxiv.org/abs/2108.00946)
 ）。这被称为“
 **CLIP方向相似性**
 ”。


* 标题 1 对应于要编辑的输入图像（图像 1）。
* 标题 2 对应于编辑后的图像（图像 2）。它应该反映编辑指令。


以下是图片概述：


![编辑一致性](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/edit-consistency.png)


我们准备了一个迷你数据集来实现这个指标。我们首先加载数据集。



```
from datasets import load_dataset

dataset = load_dataset("sayakpaul/instructpix2pix-demo", split="train")
dataset.features
```



```
{'input': Value(dtype='string', id=None),
 'edit': Value(dtype='string', id=None),
 'output': Value(dtype='string', id=None),
 'image': Image(decode=True, id=None)}
```


这里我们有：


* `输入`
 是对应于的标题
 `图像`
 。
* `编辑`
 表示编辑指令。
* `输出`
 表示修改后的标题，反映
 `编辑`
 操作说明。


让我们看一个示例。



```
idx = 0
print(f"Original caption: {dataset[idx]['input']}")
print(f"Edit instruction: {dataset[idx]['edit']}")
print(f"Modified caption: {dataset[idx]['output']}")
```



```
Original caption: 2. FAROE ISLANDS: An archipelago of 18 mountainous isles in the North Atlantic Ocean between Norway and Iceland, the Faroe Islands has 'everything you could hope for', according to Big 7 Travel. It boasts 'crystal clear waterfalls, rocky cliffs that seem to jut out of nowhere and velvety green hills'
Edit instruction: make the isles all white marble
Modified caption: 2. WHITE MARBLE ISLANDS: An archipelago of 18 mountainous white marble isles in the North Atlantic Ocean between Norway and Iceland, the White Marble Islands has 'everything you could hope for', according to Big 7 Travel. It boasts 'crystal clear waterfalls, rocky cliffs that seem to jut out of nowhere and velvety green hills'
```


这是图像：



```
dataset[idx]["image"]
```


![编辑数据集](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/edit-dataset.png)


我们将首先使用编辑指令编辑数据集的图像并计算方向相似度。


我们首先加载
 [StableDiffusionInstructPix2PixPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/pix2pix#diffusers.StableDiffusionInstructPix2PixPipeline)
 ：



```
from diffusers import StableDiffusionInstructPix2PixPipeline

instruct_pix2pix_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
).to(device)
```


现在，我们执行编辑：



```
import numpy as np


def edit\_image(input\_image, instruction):
    image = instruct_pix2pix_pipeline(
        instruction,
        image=input_image,
        output_type="np",
        generator=generator,
    ).images[0]
    return image

input_images = []
original_captions = []
modified_captions = []
edited_images = []

for idx in range(len(dataset)):
    input_image = dataset[idx]["image"]
    edit_instruction = dataset[idx]["edit"]
    edited_image = edit_image(input_image, edit_instruction)

    input_images.append(np.array(input_image))
    original_captions.append(dataset[idx]["input"])
    modified_captions.append(dataset[idx]["output"])
    edited_images.append(edited_image)
```


为了测量方向相似性，我们首先加载 CLIP 的图像和文本编码器：



```
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

clip_id = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_id)
text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
image_processor = CLIPImageProcessor.from_pretrained(clip_id)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)
```


请注意，我们正在使用特定的 CLIP 检查点，即
 `openai/clip-vit-large-patch14`
 。这是因为稳定扩散预训练是使用此 CLIP 变体进行的。有关更多详细信息，请参阅
 [文档](https://huggingface.co/docs/transformers/model_doc/clip)
 。


接下来，我们准备一个PyTorch
 `nn.模块`
 计算方向相似度：



```
import torch.nn as nn
import torch.nn.functional as F


class DirectionalSimilarity(nn.Module):
    def \_\_init\_\_(self, tokenizer, text\_encoder, image\_processor, image\_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def preprocess\_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel\_values"]
        return {"pixel\_values": image.to(device)}

    def tokenize\_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max\_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input\_ids": inputs.input_ids.to(device)}

    def encode\_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode\_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute\_directional\_similarity(self, img\_feat\_one, img\_feat\_two, text\_feat\_one, text\_feat\_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
        return sim_direction

    def forward(self, image\_one, image\_two, caption\_one, caption\_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )
        return directional_similarity
```


让我们把
 `方向相似性`
 现在使用。



```
dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)
scores = []

for i in range(len(input_images)):
    original_image = input_images[i]
    original_caption = original_captions[i]
    edited_image = edited_images[i]
    modified_caption = modified_captions[i]

    similarity_score = dir_similarity(original_image, edited_image, original_caption, modified_caption)
    scores.append(float(similarity_score.detach().cpu()))

print(f"CLIP directional similarity: {np.mean(scores)}")
# CLIP directional similarity: 0.0797976553440094
```


与 CLIP Score 一样，CLIP 方向相似度越高越好。


应该指出的是，
 `StableDiffusionInstructPix2PixPipeline`
 暴露了两个论点，即
 `image_guidance_scale`
 和
 `指导规模`
 让您可以控制最终编辑图像的质量。我们鼓励您尝试这两个参数，看看它们对方向相似性的影响。


我们可以扩展这个指标的想法来衡量原始图像和编辑后的版本的相似程度。为此，我们可以这样做
 `F.cosine_similarity(img_feat_two, img_feat_one)`
 。对于此类编辑，我们仍然希望尽可能保留图像的主要语义，即高相似度得分。


我们可以将这些指标用于类似的管道，例如
 [`StableDiffusionPix2PixZeroPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pix2pix_zero#diffusers.StableDiffusionPix2PixZeroPipeline)
 。


CLIP得分和CLIP方向相似度都依赖于CLIP模型，这可能会使评估产生偏差。


***扩展 IS、FID（稍后讨论）或 KID 等指标可能很困难***
 当评估的模型在大型图像字幕数据集（例如
 [LAION-5B数据集](https://laion.ai/blog/laion-5b/)
 ）。这是因为这些指标的底层是用于提取中间图像特征的 InceptionNet（在 ImageNet-1k 数据集上预先训练）。 Stable Diffusion 的预训练数据集可能与 InceptionNet 的预训练数据集有有限的重叠，因此它不是特征提取的良好候选者。


***使用上述指标有助于评估类条件模型。例如，
 [DiT](https://huggingface.co/docs/diffusers/main/en/api/pipelines/dit)
 。它是在 ImageNet-1k 类上进行预训练的。***



### 


 类条件图像生成


类条件生成模型通常在类标记数据集上进行预训练，例如
 [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k)
 。评估这些模型的流行指标包括 Fréchet 起始距离 (FID)、内核起始距离 (KID) 和起始分数 (IS)。在本文档中，我们重点关注 FID（
 [Heusel 等人](https://arxiv.org/abs/1706.08500)
 ）。我们展示了如何计算它
 [`DiTPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/dit)
 ，它使用
 [DiT模型](https://arxiv.org/abs/2212.09748)
 在引擎盖下。


FID 旨在衡量两个图像数据集的相似程度。按照
 [此资源](https://mm Generation.readthedocs.io/en/latest/quick_run.html#fid)
 ：


>
>
> Fréchet 起始距离是两个图像数据集之间相似性的度量。它被证明与人类对视觉质量的判断有很好的相关性，并且最常用于评估生成对抗网络样本的质量。 FID 是通过计算适合 Inception 网络特征表示的两个高斯之间的 Fréchet 距离来计算的。
>
>
>
>


这两个数据集本质上是真实图像的数据集和假图像的数据集（在我们的例子中是生成的图像）​​。 FID 通常使用两个大型数据集来计算。但是，对于本文档，我们将使用两个小型数据集。


我们首先从 ImageNet-1k 训练集中下载一些图像：



```
from zipfile import ZipFile
import requests


def download(url, local\_filepath):
    r = requests.get(url)
    with open(local_filepath, "wb") as f:
        f.write(r.content)
    return local_filepath

dummy_dataset_url = "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/sample-imagenet-images.zip"
local_filepath = download(dummy_dataset_url, dummy_dataset_url.split("/")[-1])

with ZipFile(local_filepath, "r") as zipper:
    zipper.extractall(".")
```



```
from PIL import Image
import os

dataset_path = "sample-imagenet-images"
image_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path)])

real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
```


这些是来自以下 ImageNet-1k 类别的 10 张图像：“cassette\_player”、“chain\_saw”(x2)、“church”、“gas\_pump”(x3)、“parachute”(x2) 和“tench” ”。


![真实图像](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/real-images.png)
  

*真实图像。*


现在图像已加载，让我们对它们应用一些轻量级预处理，以将它们用于 FID 计算。



```
from torchvision.transforms import functional as F


def preprocess\_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (256, 256))

real_images = torch.cat([preprocess_image(image) for image in real_images])
print(real_images.shape)
# torch.Size([10, 3, 256, 256])
```


我们现在加载
 [`DiTPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/dit)
 生成以上述类别为条件的图像。



```
from diffusers import DiTPipeline, DPMSolverMultistepScheduler

dit_pipeline = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
dit_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(dit_pipeline.scheduler.config)
dit_pipeline = dit_pipeline.to("cuda")

words = [
    "cassette player",
    "chainsaw",
    "chainsaw",
    "church",
    "gas pump",
    "gas pump",
    "gas pump",
    "parachute",
    "parachute",
    "tench",
]

class_ids = dit_pipeline.get_label_ids(words)
output = dit_pipeline(class_labels=class_ids, generator=generator, output_type="np")

fake_images = output.images
fake_images = torch.tensor(fake_images)
fake_images = fake_images.permute(0, 3, 1, 2)
print(fake_images.shape)
# torch.Size([10, 3, 256, 256])
```


现在，我们可以使用以下方法计算 FID
 [`torchmetrics`](https://torchmetrics.readthedocs.io/)
 。



```
from torchmetrics.image.fid import FrechetInceptionDistance

fid = FrechetInceptionDistance(normalize=True)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)

print(f"FID: {float(fid.compute())}")
# FID: 177.7147216796875
```


FID 越低越好。有几个因素会影响 FID：


* 图片数量（真实和虚假）
* 扩散过程中引入的随机性
* 扩散过程中的推理步骤数
* 扩散过程中使用的调度器


因此，对于最后两点，最好在不同的种子和推理步骤上运行评估，然后报告平均结果。


FID 结果往往很脆弱，因为它们取决于很多因素：


* 计算时使用的具体Inception模型。
* 计算的实现精度。
* 图像格式（如果我们从 PNG 开始，与从 JPG 开始就不一样）。


请记住，FID 在比较相似的运行时通常最有用，但它
除非作者仔细披露 FID，否则很难重现论文结果
测量代码。


这些要点也适用于其他相关指标，例如 KID 和 IS。


作为最后一步，让我们目视检查
 `假图像`
 。


![假图像](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/fake-images.png)
  

*假图像。*