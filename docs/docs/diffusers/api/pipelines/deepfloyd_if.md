# DeepFloyd IF

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/deepfloyd_if>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if>


## Overview




 DeepFloyd IF is a novel state-of-the-art open-source text-to-image model with a high degree of photorealism and language understanding.
The model is a modular composed of a frozen text encoder and three cascaded pixel diffusion modules:
 


* Stage 1: a base model that generates 64x64 px image based on text prompt,
* Stage 2: a 64x64 px => 256x256 px super-resolution model, and a
* Stage 3: a 256x256 px => 1024x1024 px super-resolution model
Stage 1 and Stage 2 utilize a frozen text encoder based on the T5 transformer to extract text embeddings,
which are then fed into a UNet architecture enhanced with cross-attention and attention pooling.
Stage 3 is
 [Stability’s x4 Upscaling model](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) 
.
The result is a highly efficient model that outperforms current state-of-the-art models, achieving a zero-shot FID score of 6.66 on the COCO dataset.
Our work underscores the potential of larger UNet architectures in the first stage of cascaded diffusion models and depicts a promising future for text-to-image synthesis.


## Usage




 Before you can use IF, you need to accept its usage conditions. To do so:
 


1. Make sure to have a
 [Hugging Face account](https://huggingface.co/join) 
 and be logged in
2. Accept the license on the model card of
 [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0) 
. Accepting the license on the stage I model card will auto accept for the other IF models.
3. Make sure to login locally. Install
 `huggingface_hub`



```
pip install huggingface_hub --upgrade
```



 run the login function in a Python shell
 



```
from huggingface_hub import login

login()
```



 and enter your
 [Hugging Face Hub access token](https://huggingface.co/docs/hub/security-tokens#what-are-user-access-tokens) 
.
 



 Next we install
 `diffusers` 
 and dependencies:
 



```
pip install diffusers accelerate transformers safetensors
```



 The following sections give more in-detail examples of how to use IF. Specifically:
 


* [Text-to-Image Generation](#text-to-image-generation)
* [Image-to-Image Generation](#text-guided-image-to-image-generation)
* [Inpainting](#text-guided-inpainting-generation)
* [Reusing model weights](#converting-between-different-pipelines)
* [Speed optimization](#optimizing-for-speed)
* [Memory optimization](#optimizing-for-memory)



**Available checkpoints** 



* *Stage-1* 



	+ [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)
	+ [DeepFloyd/IF-I-L-v1.0](https://huggingface.co/DeepFloyd/IF-I-L-v1.0)
	+ [DeepFloyd/IF-I-M-v1.0](https://huggingface.co/DeepFloyd/IF-I-M-v1.0)
* *Stage-2* 



	+ [DeepFloyd/IF-II-L-v1.0](https://huggingface.co/DeepFloyd/IF-II-L-v1.0)
	+ [DeepFloyd/IF-II-M-v1.0](https://huggingface.co/DeepFloyd/IF-II-M-v1.0)
* *Stage-3* 



	+ [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)



**Demo** 
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/DeepFloyd/IF)




**Google Colab** 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/deepfloyd_if_free_tier_google_colab.ipynb)



### 


 Text-to-Image Generation



 By default diffusers makes use of
 [model cpu offloading](https://huggingface.co/docs/diffusers/optimization/fp16#model-offloading-for-fast-inference-and-memory-savings) 
 to run the whole IF pipeline with as little as 14 GB of VRAM.
 



```
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

# stage 1
stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_model_cpu_offload()

# stage 2
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
)
stage_2.enable_model_cpu_offload()

# stage 3
safety_modules = {
    "feature\_extractor": stage_1.feature_extractor,
    "safety\_checker": stage_1.safety_checker,
    "watermarker": stage_1.watermarker,
}
stage_3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
)
stage_3.enable_model_cpu_offload()

prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
generator = torch.manual_seed(1)

# text embeds
prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

# stage 1
image = stage_1(
    prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
).images
pt_to_pil(image)[0].save("./if\_stage\_I.png")

# stage 2
image = stage_2(
    image=image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="pt",
).images
pt_to_pil(image)[0].save("./if\_stage\_II.png")

# stage 3
image = stage_3(prompt=prompt, image=image, noise_level=100, generator=generator).images
image[0].save("./if\_stage\_III.png")
```


### 


 Text Guided Image-to-Image Generation



 The same IF model weights can be used for text-guided image-to-image translation or image variation.
In this case just make sure to load the weights using the
 [IFInpaintingPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/deepfloyd_if#diffusers.IFInpaintingPipeline) 
 and
 [IFInpaintingSuperResolutionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/deepfloyd_if#diffusers.IFInpaintingSuperResolutionPipeline) 
 pipelines.
 



**Note** 
 : You can also directly move the weights of the text-to-image pipelines to the image-to-image pipelines
without loading them twice by making use of the
 `~DiffusionPipeline.components()` 
 function as explained
 [here](#converting-between-different-pipelines) 
.
 



```
from diffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline, DiffusionPipeline
from diffusers.utils import pt_to_pil

import torch

from PIL import Image
import requests
from io import BytesIO

# download image
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
response = requests.get(url)
original_image = Image.open(BytesIO(response.content)).convert("RGB")
original_image = original_image.resize((768, 512))

# stage 1
stage_1 = IFImg2ImgPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_model_cpu_offload()

# stage 2
stage_2 = IFImg2ImgSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
)
stage_2.enable_model_cpu_offload()

# stage 3
safety_modules = {
    "feature\_extractor": stage_1.feature_extractor,
    "safety\_checker": stage_1.safety_checker,
    "watermarker": stage_1.watermarker,
}
stage_3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
)
stage_3.enable_model_cpu_offload()

prompt = "A fantasy landscape in style minecraft"
generator = torch.manual_seed(1)

# text embeds
prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

# stage 1
image = stage_1(
    image=original_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="pt",
).images
pt_to_pil(image)[0].save("./if\_stage\_I.png")

# stage 2
image = stage_2(
    image=image,
    original_image=original_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="pt",
).images
pt_to_pil(image)[0].save("./if\_stage\_II.png")

# stage 3
image = stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
image[0].save("./if\_stage\_III.png")
```


### 


 Text Guided Inpainting Generation



 The same IF model weights can be used for text-guided image-to-image translation or image variation.
In this case just make sure to load the weights using the
 [IFInpaintingPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/deepfloyd_if#diffusers.IFInpaintingPipeline) 
 and
 [IFInpaintingSuperResolutionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/deepfloyd_if#diffusers.IFInpaintingSuperResolutionPipeline) 
 pipelines.
 



**Note** 
 : You can also directly move the weights of the text-to-image pipelines to the image-to-image pipelines
without loading them twice by making use of the
 `~DiffusionPipeline.components()` 
 function as explained
 [here](#converting-between-different-pipelines) 
.
 



```
from diffusers import IFInpaintingPipeline, IFInpaintingSuperResolutionPipeline, DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

from PIL import Image
import requests
from io import BytesIO

# download image
url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/person.png"
response = requests.get(url)
original_image = Image.open(BytesIO(response.content)).convert("RGB")
original_image = original_image

# download mask
url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/glasses\_mask.png"
response = requests.get(url)
mask_image = Image.open(BytesIO(response.content))
mask_image = mask_image

# stage 1
stage_1 = IFInpaintingPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_model_cpu_offload()

# stage 2
stage_2 = IFInpaintingSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
)
stage_2.enable_model_cpu_offload()

# stage 3
safety_modules = {
    "feature\_extractor": stage_1.feature_extractor,
    "safety\_checker": stage_1.safety_checker,
    "watermarker": stage_1.watermarker,
}
stage_3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
)
stage_3.enable_model_cpu_offload()

prompt = "blue sunglasses"
generator = torch.manual_seed(1)

# text embeds
prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

# stage 1
image = stage_1(
    image=original_image,
    mask_image=mask_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="pt",
).images
pt_to_pil(image)[0].save("./if\_stage\_I.png")

# stage 2
image = stage_2(
    image=image,
    original_image=original_image,
    mask_image=mask_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="pt",
).images
pt_to_pil(image)[0].save("./if\_stage\_II.png")

# stage 3
image = stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
image[0].save("./if\_stage\_III.png")
```


### 


 Converting between different pipelines



 In addition to being loaded with
 `from_pretrained` 
 , Pipelines can also be loaded directly from each other.
 



```
from diffusers import IFPipeline, IFSuperResolutionPipeline

pipe_1 = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0")
pipe_2 = IFSuperResolutionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0")


from diffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline

pipe_1 = IFImg2ImgPipeline(**pipe_1.components)
pipe_2 = IFImg2ImgSuperResolutionPipeline(**pipe_2.components)


from diffusers import IFInpaintingPipeline, IFInpaintingSuperResolutionPipeline

pipe_1 = IFInpaintingPipeline(**pipe_1.components)
pipe_2 = IFInpaintingSuperResolutionPipeline(**pipe_2.components)
```


### 


 Optimizing for speed



 The simplest optimization to run IF faster is to move all model components to the GPU.
 



```
pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
pipe.to("cuda")
```



 You can also run the diffusion process for a shorter number of timesteps.
 



 This can either be done with the
 `num_inference_steps` 
 argument
 



```
pipe("<prompt>", num_inference_steps=30)
```



 Or with the
 `timesteps` 
 argument
 



```
from diffusers.pipelines.deepfloyd_if import fast27_timesteps

pipe("<prompt>", timesteps=fast27_timesteps)
```



 When doing image variation or inpainting, you can also decrease the number of timesteps
with the strength argument. The strength argument is the amount of noise to add to
the input image which also determines how many steps to run in the denoising process.
A smaller number will vary the image less but run faster.
 



```
pipe = IFImg2ImgPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(image=image, prompt="<prompt>", strength=0.3).images
```



 You can also use
 [`torch.compile`](../../optimization/torch2.0)
. Note that we have not exhaustively tested
 `torch.compile` 
 with IF and it might not give expected results.
 



```
import torch

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

pipe.text_encoder = torch.compile(pipe.text_encoder)
pipe.unet = torch.compile(pipe.unet)
```


### 


 Optimizing for memory



 When optimizing for GPU memory, we can use the standard diffusers cpu offloading APIs.
 



 Either the model based CPU offloading,
 



```
pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
```



 or the more aggressive layer based CPU offloading.
 



```
pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
pipe.enable_sequential_cpu_offload()
```



 Additionally, T5 can be loaded in 8bit precision
 



```
from transformers import T5EncoderModel

text_encoder = T5EncoderModel.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", subfolder="text\_encoder", device_map="auto", load_in_8bit=True, variant="8bit"
)

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    text_encoder=text_encoder,  # pass the previously instantiated 8bit text encoder
    unet=None,
    device_map="auto",
)

prompt_embeds, negative_embeds = pipe.encode_prompt("<prompt>")
```



 For CPU RAM constrained machines like google colab free tier where we can’t load all
model components to the CPU at once, we can manually only load the pipeline with
the text encoder or unet when the respective model components are needed.
 



```
from diffusers import IFPipeline, IFSuperResolutionPipeline
import torch
import gc
from transformers import T5EncoderModel
from diffusers.utils import pt_to_pil

text_encoder = T5EncoderModel.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", subfolder="text\_encoder", device_map="auto", load_in_8bit=True, variant="8bit"
)

# text to image

pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    text_encoder=text_encoder,  # pass the previously instantiated 8bit text encoder
    unet=None,
    device_map="auto",
)

prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

# Remove the pipeline so we can re-load the pipeline with the unet
del text_encoder
del pipe
gc.collect()
torch.cuda.empty_cache()

pipe = IFPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16, device_map="auto"
)

generator = torch.Generator().manual_seed(0)
image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    output_type="pt",
    generator=generator,
).images

pt_to_pil(image)[0].save("./if\_stage\_I.png")

# Remove the pipeline so we can load the super-resolution pipeline
del pipe
gc.collect()
torch.cuda.empty_cache()

# First super resolution

pipe = IFSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16, device_map="auto"
)

generator = torch.Generator().manual_seed(0)
image = pipe(
    image=image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    output_type="pt",
    generator=generator,
).images

pt_to_pil(image)[0].save("./if\_stage\_II.png")
```


## Available Pipelines:



| 	 Pipeline	  | 	 Tasks	  | 	 Colab	  |
| --- | --- | --- |
| [pipeline\_if.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py)  | *Text-to-Image Generation*  | 	 -	  |
| [pipeline\_if\_superresolution.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py)  | *Text-to-Image Generation*  | 	 -	  |
| [pipeline\_if\_img2img.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img.py)  | *Image-to-Image Generation*  | 	 -	  |
| [pipeline\_if\_img2img\_superresolution.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img_superresolution.py)  | *Image-to-Image Generation*  | 	 -	  |
| [pipeline\_if\_inpainting.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting.py)  | *Image-to-Image Generation*  | 	 -	  |
| [pipeline\_if\_inpainting\_superresolution.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting_superresolution.py)  | *Image-to-Image Generation*  | 	 -	  |


## IFPipeline




### 




 class
 

 diffusers.
 

 IFPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py#L88)



 (
 


 tokenizer
 
 : T5Tokenizer
 




 text\_encoder
 
 : T5EncoderModel
 




 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : DDPMScheduler
 




 safety\_checker
 
 : typing.Optional[diffusers.pipelines.deepfloyd\_if.safety\_checker.IFSafetyChecker]
 




 feature\_extractor
 
 : typing.Optional[transformers.models.clip.image\_processing\_clip.CLIPImageProcessor]
 




 watermarker
 
 : typing.Optional[diffusers.pipelines.deepfloyd\_if.watermark.IFWatermarker]
 




 requires\_safety\_checker
 
 : bool = True
 



 )
 




#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py#L543)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 num\_inference\_steps
 
 : int = 100
 




 timesteps
 
 : typing.List[int] = None
 




 guidance\_scale
 
 : float = 7.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_images\_per\_prompt
 
 : typing.Optional[int] = 1
 




 height
 
 : typing.Optional[int] = None
 




 width
 
 : typing.Optional[int] = None
 




 eta
 
 : float = 0.0
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : int = 1
 




 clean\_caption
 
 : bool = True
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.IFPipelineOutput` 
 or
 `tuple` 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to guide the image generation. If not defined, one has to pass
 `prompt_embeds` 
.
instead.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **timesteps** 
 (
 `List[int]` 
 ,
 *optional* 
 ) —
Custom timesteps to use for the denoising process. If not defined, equal spaced
 `num_inference_steps` 
 timesteps are used. Must be in descending order.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 7.0) —
Guidance scale as defined in
 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) 
.
 `guidance_scale` 
 is defined as
 `w` 
 of equation 2. of
 [Imagen
Paper](https://arxiv.org/pdf/2205.11487.pdf) 
. Guidance scale is enabled by setting
 `guidance_scale > 1` 
. Higher guidance scale encourages to generate images that are closely linked to the text
 `prompt` 
 ,
usually at the expense of lower image quality.
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
 instead. Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is
less than
 `1` 
 ).
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **height** 
 (
 `int` 
 ,
 *optional* 
 , defaults to self.unet.config.sample\_size) —
The height in pixels of the generated image.
* **width** 
 (
 `int` 
 ,
 *optional* 
 , defaults to self.unet.config.sample\_size) —
The width in pixels of the generated image.
* **eta** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
Corresponds to parameter eta (η) in the DDIM paper:
 <https://arxiv.org/abs/2010.02502>
. Only applies to
 [schedulers.DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 , will be ignored for others.
* **generator** 
 (
 `torch.Generator` 
 or
 `List[torch.Generator]` 
 ,
 *optional* 
 ) —
One or a list of
 [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) 
 to make generation deterministic.
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pil"` 
 ) —
The output format of the generate image. Choose between
 [PIL](https://pillow.readthedocs.io/en/stable/) 
 :
 `PIL.Image.Image` 
 or
 `np.array` 
.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 `~pipelines.stable_diffusion.IFPipelineOutput` 
 instead of a plain tuple.
* **callback** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
A function that will be called every
 `callback_steps` 
 steps during inference. The function will be
called with the following arguments:
 `callback(step: int, timestep: int, latents: torch.FloatTensor)` 
.
* **callback\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The frequency at which the
 `callback` 
 function will be called. If not specified, the callback will be
called at every step.
* **clean\_caption** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to clean the caption before creating embeddings. Requires
 `beautifulsoup4` 
 and
 `ftfy` 
 to
be installed. If the dependencies are not installed, the embeddings will be created from the raw
prompt.
* **cross\_attention\_kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
A kwargs dictionary that if specified is passed along to the
 `AttentionProcessor` 
 as defined under
 `self.processor` 
 in
 [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py) 
.




 Returns
 




 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.IFPipelineOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




`~pipelines.stable_diffusion.IFPipelineOutput` 
 if
 `return_dict` 
 is True, otherwise a
 `tuple. When returning a tuple, the first element is a list with the generated images, and the second element is a list of` 
 bool
 `s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw) or watermarked content, according to the` 
 safety\_checker`.
 



 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline
>>> from diffusers.utils import pt_to_pil
>>> import torch

>>> pipe = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
>>> pipe.enable_model_cpu_offload()

>>> prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
>>> prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

>>> image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt").images

>>> # save intermediate image
>>> pil_image = pt_to_pil(image)
>>> pil_image[0].save("./if\_stage\_I.png")

>>> super_res_1_pipe = IFSuperResolutionPipeline.from_pretrained(
...     "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
... )
>>> super_res_1_pipe.enable_model_cpu_offload()

>>> image = super_res_1_pipe(
...     image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt"
... ).images

>>> # save intermediate image
>>> pil_image = pt_to_pil(image)
>>> pil_image[0].save("./if\_stage\_I.png")

>>> safety_modules = {
...     "feature\_extractor": pipe.feature_extractor,
...     "safety\_checker": pipe.safety_checker,
...     "watermarker": pipe.watermarker,
... }
>>> super_res_2_pipe = DiffusionPipeline.from_pretrained(
...     "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
... )
>>> super_res_2_pipe.enable_model_cpu_offload()

>>> image = super_res_2_pipe(
...     prompt=prompt,
...     image=image,
... ).images
>>> image[0].save("./if\_stage\_II.png")
```


#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py#L161)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]]
 




 do\_classifier\_free\_guidance
 
 : bool = True
 




 num\_images\_per\_prompt
 
 : int = 1
 




 device
 
 : typing.Optional[torch.device] = None
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 clean\_caption
 
 : bool = False
 



 )
 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
prompt to be encoded
* **do\_classifier\_free\_guidance** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
whether to use classifier free guidance or not
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
number of images that should be generated per prompt
device — (
 `torch.device` 
 ,
 *optional* 
 ):
torch device to place the resulting embeddings on
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
. instead. If not defined, one has to pass
 `negative_prompt_embeds` 
. instead.
Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is less than
 `1` 
 ).
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **clean\_caption** 
 (bool, defaults to
 `False` 
 ) —
If
 `True` 
 , the function will preprocess and clean the provided caption before encoding.


 Encodes the prompt into text encoder hidden states.
 


## IFSuperResolutionPipeline




### 




 class
 

 diffusers.
 

 IFSuperResolutionPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if_superresolution.py#L73)



 (
 


 tokenizer
 
 : T5Tokenizer
 




 text\_encoder
 
 : T5EncoderModel
 




 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : DDPMScheduler
 




 image\_noising\_scheduler
 
 : DDPMScheduler
 




 safety\_checker
 
 : typing.Optional[diffusers.pipelines.deepfloyd\_if.safety\_checker.IFSafetyChecker]
 




 feature\_extractor
 
 : typing.Optional[transformers.models.clip.image\_processing\_clip.CLIPImageProcessor]
 




 watermarker
 
 : typing.Optional[diffusers.pipelines.deepfloyd\_if.watermark.IFWatermarker]
 




 requires\_safety\_checker
 
 : bool = True
 



 )
 




#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if_superresolution.py#L610)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 height
 
 : int = None
 




 width
 
 : int = None
 




 image
 
 : typing.Union[PIL.Image.Image, numpy.ndarray, torch.FloatTensor] = None
 




 num\_inference\_steps
 
 : int = 50
 




 timesteps
 
 : typing.List[int] = None
 




 guidance\_scale
 
 : float = 4.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_images\_per\_prompt
 
 : typing.Optional[int] = 1
 




 eta
 
 : float = 0.0
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : int = 1
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 noise\_level
 
 : int = 250
 




 clean\_caption
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.IFPipelineOutput` 
 or
 `tuple` 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to guide the image generation. If not defined, one has to pass
 `prompt_embeds` 
.
instead.
* **height** 
 (
 `int` 
 ,
 *optional* 
 , defaults to None) —
The height in pixels of the generated image.
* **width** 
 (
 `int` 
 ,
 *optional* 
 , defaults to None) —
The width in pixels of the generated image.
* **image** 
 (
 `PIL.Image.Image` 
 ,
 `np.ndarray` 
 ,
 `torch.FloatTensor` 
 ) —
The image to be upscaled.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 50) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **timesteps** 
 (
 `List[int]` 
 ,
 *optional* 
 , defaults to None) —
Custom timesteps to use for the denoising process. If not defined, equal spaced
 `num_inference_steps` 
 timesteps are used. Must be in descending order.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
Guidance scale as defined in
 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) 
.
 `guidance_scale` 
 is defined as
 `w` 
 of equation 2. of
 [Imagen
Paper](https://arxiv.org/pdf/2205.11487.pdf) 
. Guidance scale is enabled by setting
 `guidance_scale > 1` 
. Higher guidance scale encourages to generate images that are closely linked to the text
 `prompt` 
 ,
usually at the expense of lower image quality.
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
 instead. Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is
less than
 `1` 
 ).
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **eta** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
Corresponds to parameter eta (η) in the DDIM paper:
 <https://arxiv.org/abs/2010.02502>
. Only applies to
 [schedulers.DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 , will be ignored for others.
* **generator** 
 (
 `torch.Generator` 
 or
 `List[torch.Generator]` 
 ,
 *optional* 
 ) —
One or a list of
 [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) 
 to make generation deterministic.
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pil"` 
 ) —
The output format of the generate image. Choose between
 [PIL](https://pillow.readthedocs.io/en/stable/) 
 :
 `PIL.Image.Image` 
 or
 `np.array` 
.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 `~pipelines.stable_diffusion.IFPipelineOutput` 
 instead of a plain tuple.
* **callback** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
A function that will be called every
 `callback_steps` 
 steps during inference. The function will be
called with the following arguments:
 `callback(step: int, timestep: int, latents: torch.FloatTensor)` 
.
* **callback\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The frequency at which the
 `callback` 
 function will be called. If not specified, the callback will be
called at every step.
* **cross\_attention\_kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
A kwargs dictionary that if specified is passed along to the
 `AttentionProcessor` 
 as defined under
 `self.processor` 
 in
 [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py) 
.
* **noise\_level** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 250) —
The amount of noise to add to the upscaled image. Must be in the range
 `[0, 1000)`
* **clean\_caption** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to clean the caption before creating embeddings. Requires
 `beautifulsoup4` 
 and
 `ftfy` 
 to
be installed. If the dependencies are not installed, the embeddings will be created from the raw
prompt.




 Returns
 




 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.IFPipelineOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




`~pipelines.stable_diffusion.IFPipelineOutput` 
 if
 `return_dict` 
 is True, otherwise a
 `tuple. When returning a tuple, the first element is a list with the generated images, and the second element is a list of` 
 bool
 `s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw) or watermarked content, according to the` 
 safety\_checker`.
 



 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline
>>> from diffusers.utils import pt_to_pil
>>> import torch

>>> pipe = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
>>> pipe.enable_model_cpu_offload()

>>> prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
>>> prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

>>> image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt").images

>>> # save intermediate image
>>> pil_image = pt_to_pil(image)
>>> pil_image[0].save("./if\_stage\_I.png")

>>> super_res_1_pipe = IFSuperResolutionPipeline.from_pretrained(
...     "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
... )
>>> super_res_1_pipe.enable_model_cpu_offload()

>>> image = super_res_1_pipe(
...     image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds
... ).images
>>> image[0].save("./if\_stage\_II.png")
```


#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if_superresolution.py#L295)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]]
 




 do\_classifier\_free\_guidance
 
 : bool = True
 




 num\_images\_per\_prompt
 
 : int = 1
 




 device
 
 : typing.Optional[torch.device] = None
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 clean\_caption
 
 : bool = False
 



 )
 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
prompt to be encoded
* **do\_classifier\_free\_guidance** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
whether to use classifier free guidance or not
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
number of images that should be generated per prompt
device — (
 `torch.device` 
 ,
 *optional* 
 ):
torch device to place the resulting embeddings on
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
. instead. If not defined, one has to pass
 `negative_prompt_embeds` 
. instead.
Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is less than
 `1` 
 ).
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **clean\_caption** 
 (bool, defaults to
 `False` 
 ) —
If
 `True` 
 , the function will preprocess and clean the provided caption before encoding.


 Encodes the prompt into text encoder hidden states.
 


## IFImg2ImgPipeline




### 




 class
 

 diffusers.
 

 IFImg2ImgPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img.py#L112)



 (
 


 tokenizer
 
 : T5Tokenizer
 




 text\_encoder
 
 : T5EncoderModel
 




 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : DDPMScheduler
 




 safety\_checker
 
 : typing.Optional[diffusers.pipelines.deepfloyd\_if.safety\_checker.IFSafetyChecker]
 




 feature\_extractor
 
 : typing.Optional[transformers.models.clip.image\_processing\_clip.CLIPImageProcessor]
 




 watermarker
 
 : typing.Optional[diffusers.pipelines.deepfloyd\_if.watermark.IFWatermarker]
 




 requires\_safety\_checker
 
 : bool = True
 



 )
 




#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img.py#L655)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 image
 
 : typing.Union[PIL.Image.Image, torch.Tensor, numpy.ndarray, typing.List[PIL.Image.Image], typing.List[torch.Tensor], typing.List[numpy.ndarray]] = None
 




 strength
 
 : float = 0.7
 




 num\_inference\_steps
 
 : int = 80
 




 timesteps
 
 : typing.List[int] = None
 




 guidance\_scale
 
 : float = 10.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_images\_per\_prompt
 
 : typing.Optional[int] = 1
 




 eta
 
 : float = 0.0
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : int = 1
 




 clean\_caption
 
 : bool = True
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.IFPipelineOutput` 
 or
 `tuple` 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to guide the image generation. If not defined, one has to pass
 `prompt_embeds` 
.
instead.
* **image** 
 (
 `torch.FloatTensor` 
 or
 `PIL.Image.Image` 
 ) —
 `Image` 
 , or tensor representing an image batch, that will be used as the starting point for the
process.
* **strength** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.7) —
Conceptually, indicates how much to transform the reference
 `image` 
. Must be between 0 and 1.
 `image` 
 will be used as a starting point, adding more noise to it the larger the
 `strength` 
. The number of
denoising steps depends on the amount of noise initially added. When
 `strength` 
 is 1, added noise will
be maximum and the denoising process will run for the full number of iterations specified in
 `num_inference_steps` 
. A value of 1, therefore, essentially ignores
 `image` 
.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 80) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **timesteps** 
 (
 `List[int]` 
 ,
 *optional* 
 ) —
Custom timesteps to use for the denoising process. If not defined, equal spaced
 `num_inference_steps` 
 timesteps are used. Must be in descending order.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 10.0) —
Guidance scale as defined in
 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) 
.
 `guidance_scale` 
 is defined as
 `w` 
 of equation 2. of
 [Imagen
Paper](https://arxiv.org/pdf/2205.11487.pdf) 
. Guidance scale is enabled by setting
 `guidance_scale > 1` 
. Higher guidance scale encourages to generate images that are closely linked to the text
 `prompt` 
 ,
usually at the expense of lower image quality.
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
 instead. Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is
less than
 `1` 
 ).
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **eta** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
Corresponds to parameter eta (η) in the DDIM paper:
 <https://arxiv.org/abs/2010.02502>
. Only applies to
 [schedulers.DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 , will be ignored for others.
* **generator** 
 (
 `torch.Generator` 
 or
 `List[torch.Generator]` 
 ,
 *optional* 
 ) —
One or a list of
 [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) 
 to make generation deterministic.
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pil"` 
 ) —
The output format of the generate image. Choose between
 [PIL](https://pillow.readthedocs.io/en/stable/) 
 :
 `PIL.Image.Image` 
 or
 `np.array` 
.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 `~pipelines.stable_diffusion.IFPipelineOutput` 
 instead of a plain tuple.
* **callback** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
A function that will be called every
 `callback_steps` 
 steps during inference. The function will be
called with the following arguments:
 `callback(step: int, timestep: int, latents: torch.FloatTensor)` 
.
* **callback\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The frequency at which the
 `callback` 
 function will be called. If not specified, the callback will be
called at every step.
* **clean\_caption** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to clean the caption before creating embeddings. Requires
 `beautifulsoup4` 
 and
 `ftfy` 
 to
be installed. If the dependencies are not installed, the embeddings will be created from the raw
prompt.
* **cross\_attention\_kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
A kwargs dictionary that if specified is passed along to the
 `AttentionProcessor` 
 as defined under
 `self.processor` 
 in
 [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py) 
.




 Returns
 




 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.IFPipelineOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




`~pipelines.stable_diffusion.IFPipelineOutput` 
 if
 `return_dict` 
 is True, otherwise a
 `tuple. When returning a tuple, the first element is a list with the generated images, and the second element is a list of` 
 bool
 `s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw) or watermarked content, according to the` 
 safety\_checker`.
 



 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> from diffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline, DiffusionPipeline
>>> from diffusers.utils import pt_to_pil
>>> import torch
>>> from PIL import Image
>>> import requests
>>> from io import BytesIO

>>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
>>> response = requests.get(url)
>>> original_image = Image.open(BytesIO(response.content)).convert("RGB")
>>> original_image = original_image.resize((768, 512))

>>> pipe = IFImg2ImgPipeline.from_pretrained(
...     "DeepFloyd/IF-I-XL-v1.0",
...     variant="fp16",
...     torch_dtype=torch.float16,
... )
>>> pipe.enable_model_cpu_offload()

>>> prompt = "A fantasy landscape in style minecraft"
>>> prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

>>> image = pipe(
...     image=original_image,
...     prompt_embeds=prompt_embeds,
...     negative_prompt_embeds=negative_embeds,
...     output_type="pt",
... ).images

>>> # save intermediate image
>>> pil_image = pt_to_pil(image)
>>> pil_image[0].save("./if\_stage\_I.png")

>>> super_res_1_pipe = IFImg2ImgSuperResolutionPipeline.from_pretrained(
...     "DeepFloyd/IF-II-L-v1.0",
...     text_encoder=None,
...     variant="fp16",
...     torch_dtype=torch.float16,
... )
>>> super_res_1_pipe.enable_model_cpu_offload()

>>> image = super_res_1_pipe(
...     image=image,
...     original_image=original_image,
...     prompt_embeds=prompt_embeds,
...     negative_prompt_embeds=negative_embeds,
... ).images
>>> image[0].save("./if\_stage\_II.png")
```


#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img.py#L186)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]]
 




 do\_classifier\_free\_guidance
 
 : bool = True
 




 num\_images\_per\_prompt
 
 : int = 1
 




 device
 
 : typing.Optional[torch.device] = None
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 clean\_caption
 
 : bool = False
 



 )
 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
prompt to be encoded
* **do\_classifier\_free\_guidance** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
whether to use classifier free guidance or not
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
number of images that should be generated per prompt
device — (
 `torch.device` 
 ,
 *optional* 
 ):
torch device to place the resulting embeddings on
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
. instead. If not defined, one has to pass
 `negative_prompt_embeds` 
. instead.
Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is less than
 `1` 
 ).
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **clean\_caption** 
 (bool, defaults to
 `False` 
 ) —
If
 `True` 
 , the function will preprocess and clean the provided caption before encoding.


 Encodes the prompt into text encoder hidden states.
 


## IFImg2ImgSuperResolutionPipeline




### 




 class
 

 diffusers.
 

 IFImg2ImgSuperResolutionPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img_superresolution.py#L115)



 (
 


 tokenizer
 
 : T5Tokenizer
 




 text\_encoder
 
 : T5EncoderModel
 




 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : DDPMScheduler
 




 image\_noising\_scheduler
 
 : DDPMScheduler
 




 safety\_checker
 
 : typing.Optional[diffusers.pipelines.deepfloyd\_if.safety\_checker.IFSafetyChecker]
 




 feature\_extractor
 
 : typing.Optional[transformers.models.clip.image\_processing\_clip.CLIPImageProcessor]
 




 watermarker
 
 : typing.Optional[diffusers.pipelines.deepfloyd\_if.watermark.IFWatermarker]
 




 requires\_safety\_checker
 
 : bool = True
 



 )
 




#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img_superresolution.py#L738)



 (
 


 image
 
 : typing.Union[PIL.Image.Image, numpy.ndarray, torch.FloatTensor]
 




 original\_image
 
 : typing.Union[PIL.Image.Image, torch.Tensor, numpy.ndarray, typing.List[PIL.Image.Image], typing.List[torch.Tensor], typing.List[numpy.ndarray]] = None
 




 strength
 
 : float = 0.8
 




 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 num\_inference\_steps
 
 : int = 50
 




 timesteps
 
 : typing.List[int] = None
 




 guidance\_scale
 
 : float = 4.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_images\_per\_prompt
 
 : typing.Optional[int] = 1
 




 eta
 
 : float = 0.0
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : int = 1
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 noise\_level
 
 : int = 250
 




 clean\_caption
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.IFPipelineOutput` 
 or
 `tuple` 


 Parameters
 




* **image** 
 (
 `torch.FloatTensor` 
 or
 `PIL.Image.Image` 
 ) —
 `Image` 
 , or tensor representing an image batch, that will be used as the starting point for the
process.
* **original\_image** 
 (
 `torch.FloatTensor` 
 or
 `PIL.Image.Image` 
 ) —
The original image that
 `image` 
 was varied from.
* **strength** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.8) —
Conceptually, indicates how much to transform the reference
 `image` 
. Must be between 0 and 1.
 `image` 
 will be used as a starting point, adding more noise to it the larger the
 `strength` 
. The number of
denoising steps depends on the amount of noise initially added. When
 `strength` 
 is 1, added noise will
be maximum and the denoising process will run for the full number of iterations specified in
 `num_inference_steps` 
. A value of 1, therefore, essentially ignores
 `image` 
.
* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to guide the image generation. If not defined, one has to pass
 `prompt_embeds` 
.
instead.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 50) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **timesteps** 
 (
 `List[int]` 
 ,
 *optional* 
 ) —
Custom timesteps to use for the denoising process. If not defined, equal spaced
 `num_inference_steps` 
 timesteps are used. Must be in descending order.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
Guidance scale as defined in
 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) 
.
 `guidance_scale` 
 is defined as
 `w` 
 of equation 2. of
 [Imagen
Paper](https://arxiv.org/pdf/2205.11487.pdf) 
. Guidance scale is enabled by setting
 `guidance_scale > 1` 
. Higher guidance scale encourages to generate images that are closely linked to the text
 `prompt` 
 ,
usually at the expense of lower image quality.
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
 instead. Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is
less than
 `1` 
 ).
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **eta** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
Corresponds to parameter eta (η) in the DDIM paper:
 <https://arxiv.org/abs/2010.02502>
. Only applies to
 [schedulers.DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 , will be ignored for others.
* **generator** 
 (
 `torch.Generator` 
 or
 `List[torch.Generator]` 
 ,
 *optional* 
 ) —
One or a list of
 [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) 
 to make generation deterministic.
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pil"` 
 ) —
The output format of the generate image. Choose between
 [PIL](https://pillow.readthedocs.io/en/stable/) 
 :
 `PIL.Image.Image` 
 or
 `np.array` 
.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 `~pipelines.stable_diffusion.IFPipelineOutput` 
 instead of a plain tuple.
* **callback** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
A function that will be called every
 `callback_steps` 
 steps during inference. The function will be
called with the following arguments:
 `callback(step: int, timestep: int, latents: torch.FloatTensor)` 
.
* **callback\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The frequency at which the
 `callback` 
 function will be called. If not specified, the callback will be
called at every step.
* **cross\_attention\_kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
A kwargs dictionary that if specified is passed along to the
 `AttentionProcessor` 
 as defined under
 `self.processor` 
 in
 [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py) 
.
* **noise\_level** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 250) —
The amount of noise to add to the upscaled image. Must be in the range
 `[0, 1000)`
* **clean\_caption** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to clean the caption before creating embeddings. Requires
 `beautifulsoup4` 
 and
 `ftfy` 
 to
be installed. If the dependencies are not installed, the embeddings will be created from the raw
prompt.




 Returns
 




 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.IFPipelineOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




`~pipelines.stable_diffusion.IFPipelineOutput` 
 if
 `return_dict` 
 is True, otherwise a
 `tuple. When returning a tuple, the first element is a list with the generated images, and the second element is a list of` 
 bool
 `s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw) or watermarked content, according to the` 
 safety\_checker`.
 



 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> from diffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline, DiffusionPipeline
>>> from diffusers.utils import pt_to_pil
>>> import torch
>>> from PIL import Image
>>> import requests
>>> from io import BytesIO

>>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
>>> response = requests.get(url)
>>> original_image = Image.open(BytesIO(response.content)).convert("RGB")
>>> original_image = original_image.resize((768, 512))

>>> pipe = IFImg2ImgPipeline.from_pretrained(
...     "DeepFloyd/IF-I-XL-v1.0",
...     variant="fp16",
...     torch_dtype=torch.float16,
... )
>>> pipe.enable_model_cpu_offload()

>>> prompt = "A fantasy landscape in style minecraft"
>>> prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

>>> image = pipe(
...     image=original_image,
...     prompt_embeds=prompt_embeds,
...     negative_prompt_embeds=negative_embeds,
...     output_type="pt",
... ).images

>>> # save intermediate image
>>> pil_image = pt_to_pil(image)
>>> pil_image[0].save("./if\_stage\_I.png")

>>> super_res_1_pipe = IFImg2ImgSuperResolutionPipeline.from_pretrained(
...     "DeepFloyd/IF-II-L-v1.0",
...     text_encoder=None,
...     variant="fp16",
...     torch_dtype=torch.float16,
... )
>>> super_res_1_pipe.enable_model_cpu_offload()

>>> image = super_res_1_pipe(
...     image=image,
...     original_image=original_image,
...     prompt_embeds=prompt_embeds,
...     negative_prompt_embeds=negative_embeds,
... ).images
>>> image[0].save("./if\_stage\_II.png")
```


#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img_superresolution.py#L337)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]]
 




 do\_classifier\_free\_guidance
 
 : bool = True
 




 num\_images\_per\_prompt
 
 : int = 1
 




 device
 
 : typing.Optional[torch.device] = None
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 clean\_caption
 
 : bool = False
 



 )
 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
prompt to be encoded
* **do\_classifier\_free\_guidance** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
whether to use classifier free guidance or not
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
number of images that should be generated per prompt
device — (
 `torch.device` 
 ,
 *optional* 
 ):
torch device to place the resulting embeddings on
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
. instead. If not defined, one has to pass
 `negative_prompt_embeds` 
. instead.
Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is less than
 `1` 
 ).
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **clean\_caption** 
 (bool, defaults to
 `False` 
 ) —
If
 `True` 
 , the function will preprocess and clean the provided caption before encoding.


 Encodes the prompt into text encoder hidden states.
 


## IFInpaintingPipeline




### 




 class
 

 diffusers.
 

 IFInpaintingPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting.py#L115)



 (
 


 tokenizer
 
 : T5Tokenizer
 




 text\_encoder
 
 : T5EncoderModel
 




 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : DDPMScheduler
 




 safety\_checker
 
 : typing.Optional[diffusers.pipelines.deepfloyd\_if.safety\_checker.IFSafetyChecker]
 




 feature\_extractor
 
 : typing.Optional[transformers.models.clip.image\_processing\_clip.CLIPImageProcessor]
 




 watermarker
 
 : typing.Optional[diffusers.pipelines.deepfloyd\_if.watermark.IFWatermarker]
 




 requires\_safety\_checker
 
 : bool = True
 



 )
 




#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting.py#L748)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 image
 
 : typing.Union[PIL.Image.Image, torch.Tensor, numpy.ndarray, typing.List[PIL.Image.Image], typing.List[torch.Tensor], typing.List[numpy.ndarray]] = None
 




 mask\_image
 
 : typing.Union[PIL.Image.Image, torch.Tensor, numpy.ndarray, typing.List[PIL.Image.Image], typing.List[torch.Tensor], typing.List[numpy.ndarray]] = None
 




 strength
 
 : float = 1.0
 




 num\_inference\_steps
 
 : int = 50
 




 timesteps
 
 : typing.List[int] = None
 




 guidance\_scale
 
 : float = 7.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_images\_per\_prompt
 
 : typing.Optional[int] = 1
 




 eta
 
 : float = 0.0
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : int = 1
 




 clean\_caption
 
 : bool = True
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.IFPipelineOutput` 
 or
 `tuple` 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to guide the image generation. If not defined, one has to pass
 `prompt_embeds` 
.
instead.
* **image** 
 (
 `torch.FloatTensor` 
 or
 `PIL.Image.Image` 
 ) —
 `Image` 
 , or tensor representing an image batch, that will be used as the starting point for the
process.
* **mask\_image** 
 (
 `PIL.Image.Image` 
 ) —
 `Image` 
 , or tensor representing an image batch, to mask
 `image` 
. White pixels in the mask will be
repainted, while black pixels will be preserved. If
 `mask_image` 
 is a PIL image, it will be converted
to a single channel (luminance) before use. If it’s a tensor, it should contain one color channel (L)
instead of 3, so the expected shape would be
 `(B, H, W, 1)` 
.
* **strength** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1.0) —
Conceptually, indicates how much to transform the reference
 `image` 
. Must be between 0 and 1.
 `image` 
 will be used as a starting point, adding more noise to it the larger the
 `strength` 
. The number of
denoising steps depends on the amount of noise initially added. When
 `strength` 
 is 1, added noise will
be maximum and the denoising process will run for the full number of iterations specified in
 `num_inference_steps` 
. A value of 1, therefore, essentially ignores
 `image` 
.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 50) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **timesteps** 
 (
 `List[int]` 
 ,
 *optional* 
 ) —
Custom timesteps to use for the denoising process. If not defined, equal spaced
 `num_inference_steps` 
 timesteps are used. Must be in descending order.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 7.0) —
Guidance scale as defined in
 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) 
.
 `guidance_scale` 
 is defined as
 `w` 
 of equation 2. of
 [Imagen
Paper](https://arxiv.org/pdf/2205.11487.pdf) 
. Guidance scale is enabled by setting
 `guidance_scale > 1` 
. Higher guidance scale encourages to generate images that are closely linked to the text
 `prompt` 
 ,
usually at the expense of lower image quality.
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
 instead. Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is
less than
 `1` 
 ).
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **eta** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
Corresponds to parameter eta (η) in the DDIM paper:
 <https://arxiv.org/abs/2010.02502>
. Only applies to
 [schedulers.DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 , will be ignored for others.
* **generator** 
 (
 `torch.Generator` 
 or
 `List[torch.Generator]` 
 ,
 *optional* 
 ) —
One or a list of
 [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) 
 to make generation deterministic.
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pil"` 
 ) —
The output format of the generate image. Choose between
 [PIL](https://pillow.readthedocs.io/en/stable/) 
 :
 `PIL.Image.Image` 
 or
 `np.array` 
.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 `~pipelines.stable_diffusion.IFPipelineOutput` 
 instead of a plain tuple.
* **callback** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
A function that will be called every
 `callback_steps` 
 steps during inference. The function will be
called with the following arguments:
 `callback(step: int, timestep: int, latents: torch.FloatTensor)` 
.
* **callback\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The frequency at which the
 `callback` 
 function will be called. If not specified, the callback will be
called at every step.
* **clean\_caption** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to clean the caption before creating embeddings. Requires
 `beautifulsoup4` 
 and
 `ftfy` 
 to
be installed. If the dependencies are not installed, the embeddings will be created from the raw
prompt.
* **cross\_attention\_kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
A kwargs dictionary that if specified is passed along to the
 `AttentionProcessor` 
 as defined under
 `self.processor` 
 in
 [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py) 
.




 Returns
 




 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.IFPipelineOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




`~pipelines.stable_diffusion.IFPipelineOutput` 
 if
 `return_dict` 
 is True, otherwise a
 `tuple. When returning a tuple, the first element is a list with the generated images, and the second element is a list of` 
 bool
 `s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw) or watermarked content, according to the` 
 safety\_checker`.
 



 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> from diffusers import IFInpaintingPipeline, IFInpaintingSuperResolutionPipeline, DiffusionPipeline
>>> from diffusers.utils import pt_to_pil
>>> import torch
>>> from PIL import Image
>>> import requests
>>> from io import BytesIO

>>> url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/person.png"
>>> response = requests.get(url)
>>> original_image = Image.open(BytesIO(response.content)).convert("RGB")
>>> original_image = original_image

>>> url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/glasses\_mask.png"
>>> response = requests.get(url)
>>> mask_image = Image.open(BytesIO(response.content))
>>> mask_image = mask_image

>>> pipe = IFInpaintingPipeline.from_pretrained(
...     "DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16
... )
>>> pipe.enable_model_cpu_offload()

>>> prompt = "blue sunglasses"
>>> prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

>>> image = pipe(
...     image=original_image,
...     mask_image=mask_image,
...     prompt_embeds=prompt_embeds,
...     negative_prompt_embeds=negative_embeds,
...     output_type="pt",
... ).images

>>> # save intermediate image
>>> pil_image = pt_to_pil(image)
>>> pil_image[0].save("./if\_stage\_I.png")

>>> super_res_1_pipe = IFInpaintingSuperResolutionPipeline.from_pretrained(
...     "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
... )
>>> super_res_1_pipe.enable_model_cpu_offload()

>>> image = super_res_1_pipe(
...     image=image,
...     mask_image=mask_image,
...     original_image=original_image,
...     prompt_embeds=prompt_embeds,
...     negative_prompt_embeds=negative_embeds,
... ).images
>>> image[0].save("./if\_stage\_II.png")
```


#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting.py#L189)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]]
 




 do\_classifier\_free\_guidance
 
 : bool = True
 




 num\_images\_per\_prompt
 
 : int = 1
 




 device
 
 : typing.Optional[torch.device] = None
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 clean\_caption
 
 : bool = False
 



 )
 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
prompt to be encoded
* **do\_classifier\_free\_guidance** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
whether to use classifier free guidance or not
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
number of images that should be generated per prompt
device — (
 `torch.device` 
 ,
 *optional* 
 ):
torch device to place the resulting embeddings on
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
. instead. If not defined, one has to pass
 `negative_prompt_embeds` 
. instead.
Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is less than
 `1` 
 ).
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **clean\_caption** 
 (bool, defaults to
 `False` 
 ) —
If
 `True` 
 , the function will preprocess and clean the provided caption before encoding.


 Encodes the prompt into text encoder hidden states.
 


## IFInpaintingSuperResolutionPipeline




### 




 class
 

 diffusers.
 

 IFInpaintingSuperResolutionPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting_superresolution.py#L117)



 (
 


 tokenizer
 
 : T5Tokenizer
 




 text\_encoder
 
 : T5EncoderModel
 




 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : DDPMScheduler
 




 image\_noising\_scheduler
 
 : DDPMScheduler
 




 safety\_checker
 
 : typing.Optional[diffusers.pipelines.deepfloyd\_if.safety\_checker.IFSafetyChecker]
 




 feature\_extractor
 
 : typing.Optional[transformers.models.clip.image\_processing\_clip.CLIPImageProcessor]
 




 watermarker
 
 : typing.Optional[diffusers.pipelines.deepfloyd\_if.watermark.IFWatermarker]
 




 requires\_safety\_checker
 
 : bool = True
 



 )
 




#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting_superresolution.py#L826)



 (
 


 image
 
 : typing.Union[PIL.Image.Image, numpy.ndarray, torch.FloatTensor]
 




 original\_image
 
 : typing.Union[PIL.Image.Image, torch.Tensor, numpy.ndarray, typing.List[PIL.Image.Image], typing.List[torch.Tensor], typing.List[numpy.ndarray]] = None
 




 mask\_image
 
 : typing.Union[PIL.Image.Image, torch.Tensor, numpy.ndarray, typing.List[PIL.Image.Image], typing.List[torch.Tensor], typing.List[numpy.ndarray]] = None
 




 strength
 
 : float = 0.8
 




 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 num\_inference\_steps
 
 : int = 100
 




 timesteps
 
 : typing.List[int] = None
 




 guidance\_scale
 
 : float = 4.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_images\_per\_prompt
 
 : typing.Optional[int] = 1
 




 eta
 
 : float = 0.0
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : int = 1
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 noise\_level
 
 : int = 0
 




 clean\_caption
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.IFPipelineOutput` 
 or
 `tuple` 


 Parameters
 




* **image** 
 (
 `torch.FloatTensor` 
 or
 `PIL.Image.Image` 
 ) —
 `Image` 
 , or tensor representing an image batch, that will be used as the starting point for the
process.
* **original\_image** 
 (
 `torch.FloatTensor` 
 or
 `PIL.Image.Image` 
 ) —
The original image that
 `image` 
 was varied from.
* **mask\_image** 
 (
 `PIL.Image.Image` 
 ) —
 `Image` 
 , or tensor representing an image batch, to mask
 `image` 
. White pixels in the mask will be
repainted, while black pixels will be preserved. If
 `mask_image` 
 is a PIL image, it will be converted
to a single channel (luminance) before use. If it’s a tensor, it should contain one color channel (L)
instead of 3, so the expected shape would be
 `(B, H, W, 1)` 
.
* **strength** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.8) —
Conceptually, indicates how much to transform the reference
 `image` 
. Must be between 0 and 1.
 `image` 
 will be used as a starting point, adding more noise to it the larger the
 `strength` 
. The number of
denoising steps depends on the amount of noise initially added. When
 `strength` 
 is 1, added noise will
be maximum and the denoising process will run for the full number of iterations specified in
 `num_inference_steps` 
. A value of 1, therefore, essentially ignores
 `image` 
.
* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to guide the image generation. If not defined, one has to pass
 `prompt_embeds` 
.
instead.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **timesteps** 
 (
 `List[int]` 
 ,
 *optional* 
 ) —
Custom timesteps to use for the denoising process. If not defined, equal spaced
 `num_inference_steps` 
 timesteps are used. Must be in descending order.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
Guidance scale as defined in
 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) 
.
 `guidance_scale` 
 is defined as
 `w` 
 of equation 2. of
 [Imagen
Paper](https://arxiv.org/pdf/2205.11487.pdf) 
. Guidance scale is enabled by setting
 `guidance_scale > 1` 
. Higher guidance scale encourages to generate images that are closely linked to the text
 `prompt` 
 ,
usually at the expense of lower image quality.
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
 instead. Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is
less than
 `1` 
 ).
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **eta** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
Corresponds to parameter eta (η) in the DDIM paper:
 <https://arxiv.org/abs/2010.02502>
. Only applies to
 [schedulers.DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 , will be ignored for others.
* **generator** 
 (
 `torch.Generator` 
 or
 `List[torch.Generator]` 
 ,
 *optional* 
 ) —
One or a list of
 [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) 
 to make generation deterministic.
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pil"` 
 ) —
The output format of the generate image. Choose between
 [PIL](https://pillow.readthedocs.io/en/stable/) 
 :
 `PIL.Image.Image` 
 or
 `np.array` 
.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 `~pipelines.stable_diffusion.IFPipelineOutput` 
 instead of a plain tuple.
* **callback** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
A function that will be called every
 `callback_steps` 
 steps during inference. The function will be
called with the following arguments:
 `callback(step: int, timestep: int, latents: torch.FloatTensor)` 
.
* **callback\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The frequency at which the
 `callback` 
 function will be called. If not specified, the callback will be
called at every step.
* **cross\_attention\_kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
A kwargs dictionary that if specified is passed along to the
 `AttentionProcessor` 
 as defined under
 `self.processor` 
 in
 [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py) 
.
* **noise\_level** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 0) —
The amount of noise to add to the upscaled image. Must be in the range
 `[0, 1000)`
* **clean\_caption** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to clean the caption before creating embeddings. Requires
 `beautifulsoup4` 
 and
 `ftfy` 
 to
be installed. If the dependencies are not installed, the embeddings will be created from the raw
prompt.




 Returns
 




 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.IFPipelineOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




`~pipelines.stable_diffusion.IFPipelineOutput` 
 if
 `return_dict` 
 is True, otherwise a
 `tuple. When returning a tuple, the first element is a list with the generated images, and the second element is a list of` 
 bool
 `s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw) or watermarked content, according to the` 
 safety\_checker`.
 



 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> from diffusers import IFInpaintingPipeline, IFInpaintingSuperResolutionPipeline, DiffusionPipeline
>>> from diffusers.utils import pt_to_pil
>>> import torch
>>> from PIL import Image
>>> import requests
>>> from io import BytesIO

>>> url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/person.png"
>>> response = requests.get(url)
>>> original_image = Image.open(BytesIO(response.content)).convert("RGB")
>>> original_image = original_image

>>> url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/glasses\_mask.png"
>>> response = requests.get(url)
>>> mask_image = Image.open(BytesIO(response.content))
>>> mask_image = mask_image

>>> pipe = IFInpaintingPipeline.from_pretrained(
...     "DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16
... )
>>> pipe.enable_model_cpu_offload()

>>> prompt = "blue sunglasses"

>>> prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)
>>> image = pipe(
...     image=original_image,
...     mask_image=mask_image,
...     prompt_embeds=prompt_embeds,
...     negative_prompt_embeds=negative_embeds,
...     output_type="pt",
... ).images

>>> # save intermediate image
>>> pil_image = pt_to_pil(image)
>>> pil_image[0].save("./if\_stage\_I.png")

>>> super_res_1_pipe = IFInpaintingSuperResolutionPipeline.from_pretrained(
...     "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
... )
>>> super_res_1_pipe.enable_model_cpu_offload()

>>> image = super_res_1_pipe(
...     image=image,
...     mask_image=mask_image,
...     original_image=original_image,
...     prompt_embeds=prompt_embeds,
...     negative_prompt_embeds=negative_embeds,
... ).images
>>> image[0].save("./if\_stage\_II.png")
```


#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting_superresolution.py#L339)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]]
 




 do\_classifier\_free\_guidance
 
 : bool = True
 




 num\_images\_per\_prompt
 
 : int = 1
 




 device
 
 : typing.Optional[torch.device] = None
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 clean\_caption
 
 : bool = False
 



 )
 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
prompt to be encoded
* **do\_classifier\_free\_guidance** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
whether to use classifier free guidance or not
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
number of images that should be generated per prompt
device — (
 `torch.device` 
 ,
 *optional* 
 ):
torch device to place the resulting embeddings on
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
. instead. If not defined, one has to pass
 `negative_prompt_embeds` 
. instead.
Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is less than
 `1` 
 ).
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **clean\_caption** 
 (bool, defaults to
 `False` 
 ) —
If
 `True` 
 , the function will preprocess and clean the provided caption before encoding.


 Encodes the prompt into text encoder hidden states.