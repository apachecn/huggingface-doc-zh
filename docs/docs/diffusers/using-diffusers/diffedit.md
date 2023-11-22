# 差异编辑

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/diffedit>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/diffedit>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


图像编辑通常需要提供要编辑区域的掩模。 DiffEdit 根据文本查询自动为您生成蒙版，从而无需图像编辑软件即可更轻松地创建蒙版。 DiffEdit 算法分三个步骤工作：


1. 扩散模型以某些查询文本和参考文本为条件对图像进行去噪，这对图像的不同区域产生不同的噪声估计；差异用于推断掩码，以识别需要更改图像的哪个区域以匹配查询文本
2.使用DDIM将输入图像编码到潜在空间中
3. 使用以文本查询为条件的扩散模型对潜伏进行解码，使用掩模作为指导，使得掩模外的像素保持与输入图像中的相同


本指南将向您展示如何使用 DiffEdit 编辑图像，而无需手动创建蒙版。


在开始之前，请确保已安装以下库：



```
# uncomment to install the necessary libraries in Colab
#!pip install diffusers transformers accelerate safetensors
```


这
 [StableDiffusionDiffEditPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/diffedit#diffusers.StableDiffusionDiffEditPipeline)
 需要一个图像掩模和一组部分反转的潜在变量。图像掩模是从生成的
 [generate\_mask()](/docs/diffusers/v0.23.0/en/api/pipelines/diffedit#diffusers.StableDiffusionDiffEditPipeline.generate_mask)
 函数，并包含两个参数，
 `源提示`
 和
 `目标提示`
 。这些参数决定了要在图像中编辑的内容。例如，如果你想换一碗
 *水果*
 到一碗
 *梨*
 ， 然后：



```
source_prompt = "a bowl of fruits"
target_prompt = "a bowl of pears"
```


部分反转的潜伏是从
 [invert()](/docs/diffusers/v0.23.0/en/api/pipelines/diffedit#diffusers.StableDiffusionDiffEditPipeline.invert)
 函数，通常最好包含一个
 `提示`
 或者
 *标题*
 描述图像以帮助指导逆潜在采样过程。标题通常可以是你的
 `源提示`
 ，但请随意尝试其他文字描述！


让我们加载管道、调度程序、逆调度程序，并启用一些优化以减少内存使用：



```
import torch
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionDiffEditPipeline

pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    safety_checker=None,
    use_safetensors=True,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()
```


加载图像进行编辑：



```
from diffusers.utils import load_image

img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"
raw_image = load_image(img_url).convert("RGB").resize((768, 768))
```


使用
 [generate\_mask()](/docs/diffusers/v0.23.0/en/api/pipelines/diffedit#diffusers.StableDiffusionDiffEditPipeline.generate_mask)
 函数来生成图像掩模。您需要将其传递给
 `源提示`
 和
 `目标提示`
 指定要在图像中编辑的内容：



```
source_prompt = "a bowl of fruits"
target_prompt = "a basket of pears"
mask_image = pipeline.generate_mask(
    image=raw_image,
    source_prompt=source_prompt,
    target_prompt=target_prompt,
)
```


接下来，创建反转潜在变量并向其传递描述图像的标题：



```
inv_latents = pipeline.invert(prompt=source_prompt, image=raw_image).latents
```


最后，将图像掩模和反转潜伏传递到管道。这
 `目标提示`
 成为
 `提示`
 现在，以及
 `源提示`
 被用作
 `否定提示`
 :



```
image = pipeline(
    prompt=target_prompt,
    mask_image=mask_image,
    image_latents=inv_latents,
    negative_prompt=source_prompt,
).images[0]
image.save("edited\_image.png")
```


![](https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png)

 原始图像


![](https://github.com/Xiang-cd/DiffEdit-stable-diffusion/blob/main/assets/target.png?raw=true)

 编辑后的图像


## 生成源嵌入和目标嵌入



源嵌入和目标嵌入可以使用以下命令自动生成
 [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)
 模型而不是手动创建它们。


从 🤗 Transformers 库加载 Flan-T5 模型和分词器：



```
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto", torch_dtype=torch.float16)
```


提供一些初始文本以提示模型生成源提示和目标提示。



```
source_concept = "bowl"
target_concept = "basket"

source_text = f"Provide a caption for images containing a {source\_concept}. "
"The captions should be in English and should be no longer than 150 characters."

target_text = f"Provide a caption for images containing a {target\_concept}. "
"The captions should be in English and should be no longer than 150 characters."
```


接下来，创建一个实用函数来生成提示：



```
@torch.no\_grad
def generate\_prompts(input\_prompt):
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(
        input_ids, temperature=0.8, num_return_sequences=16, do_sample=True, max_new_tokens=128, top_k=10
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

source_prompts = generate_prompts(source_text)
target_prompts = generate_prompts(target_text)
print(source_prompts)
print(target_prompts)
```


查看
 [生成策略](https://huggingface.co/docs/transformers/main/en/Generation_strategies)
 如果您有兴趣了解有关生成不同质量文本的策略的更多信息，请参阅指南。


加载使用的文本编码器模型
 [StableDiffusionDiffEditPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/diffedit#diffusers.StableDiffusionDiffEditPipeline)
 对文本进行编码。您将使用文本编码器来计算文本嵌入：



```
import torch 
from diffusers import StableDiffusionDiffEditPipeline 

pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()

@torch.no\_grad()
def embed\_prompts(sentences, tokenizer, text\_encoder, device="cuda"):
    embeddings = []
    for sent in sentences:
        text_inputs = tokenizer(
            sent,
            padding="max\_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=None)[0]
        embeddings.append(prompt_embeds)
    return torch.concatenate(embeddings, dim=0).mean(dim=0).unsqueeze(0)

source_embeds = embed_prompts(source_prompts, pipeline.tokenizer, pipeline.text_encoder)
target_embeds = embed_prompts(target_prompts, pipeline.tokenizer, pipeline.text_encoder)
```


最后，将嵌入传递给
 [generate\_mask()](/docs/diffusers/v0.23.0/en/api/pipelines/diffedit#diffusers.StableDiffusionDiffEditPipeline.generate_mask)
 和
 [invert()](/docs/diffusers/v0.23.0/en/api/pipelines/diffedit#diffusers.StableDiffusionDiffEditPipeline.invert)
 函数和生成图像的管道：



```
  from diffusers import DDIMInverseScheduler, DDIMScheduler
  from diffusers.utils import load_image

  pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
  pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)

  img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"
  raw_image = load_image(img_url).convert("RGB").resize((768, 768))


  mask_image = pipeline.generate_mask(
      image=raw_image,
+ source\_prompt\_embeds=source\_embeds,
+ target\_prompt\_embeds=target\_embeds,
  )

  inv_latents = pipeline.invert(
+ prompt\_embeds=source\_embeds,
      image=raw_image,
  ).latents

  images = pipeline(
      mask_image=mask_image,
      image_latents=inv_latents,
+ prompt\_embeds=target\_embeds,
+ negative\_prompt\_embeds=source\_embeds,
  ).images
  images[0].save("edited_image.png")
```


## 生成反转标题



虽然您可以使用
 `源提示`
 作为帮助生成部分反转潜在变量的标题，您还可以使用
 [BLIP](https://huggingface.co/docs/transformers/model_doc/blip)
 模型自动生成标题。


从 Transformers 库加载 BLIP 模型和处理器：



```
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16, low_cpu_mem_usage=True)
```


创建一个实用函数以从输入图像生成标题：



```
@torch.no\_grad()
def generate\_caption(images, caption\_generator, caption\_processor):
    text = "a photograph of"

    inputs = caption_processor(images, text, return_tensors="pt").to(device="cuda", dtype=caption_generator.dtype)
    caption_generator.to("cuda")
    outputs = caption_generator.generate(**inputs, max_new_tokens=128)

    # offload caption generator
    caption_generator.to("cpu")

    caption = caption_processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return caption
```


加载输入图像并使用以下命令为其生成标题
 `生成标题`
 功能：



```
from diffusers.utils import load_image

img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"
raw_image = load_image(img_url).convert("RGB").resize((768, 768))
caption = generate_caption(raw_image, model, processor)
```


![](https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png)

 生成的标题：“桌子上一碗水果的照片”


现在您可以将标题放入
 [invert()](/docs/diffusers/v0.23.0/en/api/pipelines/diffedit#diffusers.StableDiffusionDiffEditPipeline.invert)
 函数来生成部分反转的潜在变量！