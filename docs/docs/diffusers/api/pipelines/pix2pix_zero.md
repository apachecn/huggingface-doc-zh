# Pix2Pix Zero

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/pix2pix_zero>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/pix2pix_zero>



[Zero-shot Image-to-Image Translation](https://huggingface.co/papers/2302.03027) 
 is by Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu.
 



 The abstract from the paper is:
 



*Large-scale text-to-image generative models have shown their remarkable ability to synthesize diverse and high-quality images. However, it is still challenging to directly apply these models for editing real images for two reasons. First, it is hard for users to come up with a perfect text prompt that accurately describes every visual detail in the input image. Second, while existing models can introduce desirable changes in certain regions, they often dramatically alter the input content and introduce unexpected changes in unwanted regions. In this work, we propose pix2pix-zero, an image-to-image translation method that can preserve the content of the original image without manual prompting. We first automatically discover editing directions that reflect desired edits in the text embedding space. To preserve the general content structure after editing, we further propose cross-attention guidance, which aims to retain the cross-attention maps of the input image throughout the diffusion process. In addition, our method does not need additional training for these edits and can directly use the existing pre-trained text-to-image diffusion model. We conduct extensive experiments and show that our method outperforms existing and concurrent works for both real and synthetic image editing.* 




 You can find additional information about Pix2Pix Zero on the
 [project page](https://pix2pixzero.github.io/) 
 ,
 [original codebase](https://github.com/pix2pixzero/pix2pix-zero) 
 , and try it out in a
 [demo](https://huggingface.co/spaces/pix2pix-zero-library/pix2pix-zero-demo) 
.
 


## Tips



* The pipeline can be conditioned on real input images. Check out the code examples below to know more.
* The pipeline exposes two arguments namely
 `source_embeds` 
 and
 `target_embeds` 
 that let you control the direction of the semantic edits in the final image to be generated. Let’s say,
you wanted to translate from “cat” to “dog”. In this case, the edit direction will be “cat -> dog”. To reflect
this in the pipeline, you simply have to set the embeddings related to the phrases including “cat” to
 `source_embeds` 
 and “dog” to
 `target_embeds` 
. Refer to the code example below for more details.
* When you’re using this pipeline from a prompt, specify the
 *source* 
 concept in the prompt. Taking
the above example, a valid input prompt would be: “a high resolution painting of a
 **cat** 
 in the style of van gough”.
* If you wanted to reverse the direction in the example above, i.e., “dog -> cat”, then it’s recommended to:
	+ Swap the
	 `source_embeds` 
	 and
	 `target_embeds` 
	.
	+ Change the input prompt to include “dog”.
* To learn more about how the source and target embeddings are generated, refer to the
 [original
paper](https://arxiv.org/abs/2302.03027) 
. Below, we also provide some directions on how to generate the embeddings.
* Note that the quality of the outputs generated with this pipeline is dependent on how good the
 `source_embeds` 
 and
 `target_embeds` 
 are. Please, refer to
 [this discussion](#generating-source-and-target-embeddings) 
 for some suggestions on the topic.


## Available Pipelines:



| 	 Pipeline	  | 	 Tasks	  | 	 Demo	  |
| --- | --- | --- |
| [StableDiffusionPix2PixZeroPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_pix2pix_zero.py)  | *Text-Based Image Editing*  | [🤗 Space](https://huggingface.co/spaces/pix2pix-zero-library/pix2pix-zero-demo)  |


## Usage example



### 


 Based on an image generated with the input prompt



```
import requests
import torch

from diffusers import DDIMScheduler, StableDiffusionPix2PixZeroPipeline


def download(embedding\_url, local\_filepath):
    r = requests.get(embedding_url)
    with open(local_filepath, "wb") as f:
        f.write(r.content)


model_ckpt = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
    model_ckpt, conditions_input_image=False, torch_dtype=torch.float16
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.to("cuda")

prompt = "a high resolution painting of a cat in the style of van gogh"
src_embs_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/embeddings\_sd\_1.4/cat.pt"
target_embs_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/embeddings\_sd\_1.4/dog.pt"

for url in [src_embs_url, target_embs_url]:
    download(url, url.split("/")[-1])

src_embeds = torch.load(src_embs_url.split("/")[-1])
target_embeds = torch.load(target_embs_url.split("/")[-1])

images = pipeline(
    prompt,
    source_embeds=src_embeds,
    target_embeds=target_embeds,
    num_inference_steps=50,
    cross_attention_guidance_amount=0.15,
).images
images[0].save("edited\_image\_dog.png")
```


### 


 Based on an input image



 When the pipeline is conditioned on an input image, we first obtain an inverted
noise from it using a
 `DDIMInverseScheduler` 
 with the help of a generated caption. Then
the inverted noise is used to start the generation process.
 



 First, let’s load our pipeline:
 



```
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline

captioner_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(captioner_id)
model = BlipForConditionalGeneration.from_pretrained(captioner_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)

sd_model_ckpt = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
    sd_model_ckpt,
    caption_generator=model,
    caption_processor=processor,
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()
```



 Then, we load an input image for conditioning and obtain a suitable caption for it:
 



```
import requests
from PIL import Image

img_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/test\_images/cats/cat\_6.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB").resize((512, 512))
caption = pipeline.generate_caption(raw_image)
```



 Then we employ the generated caption and the input image to get the inverted noise:
 



```
generator = torch.manual_seed(0)
inv_latents = pipeline.invert(caption, image=raw_image, generator=generator).latents
```



 Now, generate the image with edit directions:
 



```
# See the "Generating source and target embeddings" section below to
# automate the generation of these captions with a pre-trained model like Flan-T5 as explained below.
source_prompts = ["a cat sitting on the street", "a cat playing in the field", "a face of a cat"]
target_prompts = ["a dog sitting on the street", "a dog playing in the field", "a face of a dog"]

source_embeds = pipeline.get_embeds(source_prompts, batch_size=2)
target_embeds = pipeline.get_embeds(target_prompts, batch_size=2)


image = pipeline(
    caption,
    source_embeds=source_embeds,
    target_embeds=target_embeds,
    num_inference_steps=50,
    cross_attention_guidance_amount=0.15,
    generator=generator,
    latents=inv_latents,
    negative_prompt=caption,
).images[0]
image.save("edited\_image.png")
```


## Generating source and target embeddings




 The authors originally used the
 [GPT-3 API](https://openai.com/api/) 
 to generate the source and target captions for discovering
edit directions. However, we can also leverage open source and public models for the same purpose.
Below, we provide an end-to-end example with the
 [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) 
 model
for generating captions and
 [CLIP](https://huggingface.co/docs/transformers/model_doc/clip) 
 for
computing embeddings on the generated captions.
 



**1. Load the generation model** 
 :
 



```
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto", torch_dtype=torch.float16)
```



**2. Construct a starting prompt** 
 :
 



```
source_concept = "cat"
target_concept = "dog"

source_text = f"Provide a caption for images containing a {source\_concept}. "
"The captions should be in English and should be no longer than 150 characters."

target_text = f"Provide a caption for images containing a {target\_concept}. "
"The captions should be in English and should be no longer than 150 characters."
```



 Here, we’re interested in the “cat -> dog” direction.
 



**3. Generate captions** 
 :
 



 We can use a utility like so for this purpose.
 



```
def generate\_captions(input\_prompt):
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(
        input_ids, temperature=0.8, num_return_sequences=16, do_sample=True, max_new_tokens=128, top_k=10
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
```



 And then we just call it to generate our captions:
 



```
source_captions = generate_captions(source_text)
target_captions = generate_captions(target_concept)
```



 We encourage you to play around with the different parameters supported by the
 `generate()` 
 method (
 [documentation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.generation_tf_utils.TFGenerationMixin.generate) 
 ) for the generation quality you are looking for.
 



**4. Load the embedding model** 
 :
 



 Here, we need to use the same text encoder model used by the subsequent Stable Diffusion model.
 



```
from diffusers import StableDiffusionPix2PixZeroPipeline 

pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")
tokenizer = pipeline.tokenizer
text_encoder = pipeline.text_encoder
```



**5. Compute embeddings** 
 :
 



```
import torch 

def embed\_captions(sentences, tokenizer, text\_encoder, device="cuda"):
    with torch.no_grad():
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

source_embeddings = embed_captions(source_captions, tokenizer, text_encoder)
target_embeddings = embed_captions(target_captions, tokenizer, text_encoder)
```



 And you’re done!
 [Here](https://colab.research.google.com/drive/1tz2C1EdfZYAPlzXXbTnf-5PRBiR8_R1F?usp=sharing) 
 is a Colab Notebook that you can use to interact with the entire process.
 



 Now, you can use these embeddings directly while calling the pipeline:
 



```
from diffusers import DDIMScheduler

pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

images = pipeline(
    prompt,
    source_embeds=source_embeddings,
    target_embeds=target_embeddings,
    num_inference_steps=50,
    cross_attention_guidance_amount=0.15,
).images
images[0].save("edited\_image\_dog.png")
```


## StableDiffusionPix2PixZeroPipeline




### 




 class
 

 diffusers.
 

 StableDiffusionPix2PixZeroPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_pix2pix_zero.py#L283)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : CLIPTextModel
 




 tokenizer
 
 : CLIPTokenizer
 




 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : typing.Union[diffusers.schedulers.scheduling\_ddpm.DDPMScheduler, diffusers.schedulers.scheduling\_ddim.DDIMScheduler, diffusers.schedulers.scheduling\_euler\_ancestral\_discrete.EulerAncestralDiscreteScheduler, diffusers.schedulers.scheduling\_lms\_discrete.LMSDiscreteScheduler]
 




 feature\_extractor
 
 : CLIPImageProcessor
 




 safety\_checker
 
 : StableDiffusionSafetyChecker
 




 inverse\_scheduler
 
 : DDIMInverseScheduler
 




 caption\_generator
 
 : BlipForConditionalGeneration
 




 caption\_processor
 
 : BlipProcessor
 




 requires\_safety\_checker
 
 : bool = True
 



 )
 


 Parameters
 




* **vae** 
 (
 [AutoencoderKL](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL) 
 ) —
Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
* **text\_encoder** 
 (
 `CLIPTextModel` 
 ) —
Frozen text-encoder. Stable Diffusion uses the text portion of
 [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel) 
 , specifically
the
 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 
 variant.
* **tokenizer** 
 (
 `CLIPTokenizer` 
 ) —
Tokenizer of class
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 
.
* **unet** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) — Conditional U-Net architecture to denoise the encoded image latents.
* **scheduler** 
 (
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 ) —
A scheduler to be used in combination with
 `unet` 
 to denoise the encoded image latents. Can be one of
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 ,
 [LMSDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler) 
 ,
 [EulerAncestralDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/euler_ancestral#diffusers.EulerAncestralDiscreteScheduler) 
 , or
 [DDPMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.DDPMScheduler) 
.
* **safety\_checker** 
 (
 `StableDiffusionSafetyChecker` 
 ) —
Classification module that estimates whether generated images could be considered offensive or harmful.
Please, refer to the
 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) 
 for details.
* **feature\_extractor** 
 (
 `CLIPImageProcessor` 
 ) —
Model that extracts features from generated images to be used as inputs for the
 `safety_checker` 
.
* **requires\_safety\_checker** 
 (bool) —
Whether the pipeline requires a safety checker. We recommend setting it to True if you’re using the
pipeline publicly.


 Pipeline for pixel-levl image editing using Pix2Pix Zero. Based on Stable Diffusion.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_pix2pix_zero.py#L809)



 (
 


 prompt
 
 : typing.Union[typing.List[str], str, NoneType] = None
 




 source\_embeds
 
 : Tensor = None
 




 target\_embeds
 
 : Tensor = None
 




 height
 
 : typing.Optional[int] = None
 




 width
 
 : typing.Optional[int] = None
 




 num\_inference\_steps
 
 : int = 50
 




 guidance\_scale
 
 : float = 7.5
 




 negative\_prompt
 
 : typing.Union[typing.List[str], str, NoneType] = None
 




 num\_images\_per\_prompt
 
 : typing.Optional[int] = 1
 




 eta
 
 : float = 0.0
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 cross\_attention\_guidance\_amount
 
 : float = 0.1
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : typing.Optional[int] = 1
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 clip\_skip
 
 : typing.Optional[int] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[StableDiffusionPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) 
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
* **source\_embeds** 
 (
 `torch.Tensor` 
 ) —
Source concept embeddings. Generation of the embeddings as per the
 [original
paper](https://arxiv.org/abs/2302.03027) 
. Used in discovering the edit direction.
* **target\_embeds** 
 (
 `torch.Tensor` 
 ) —
Target concept embeddings. Generation of the embeddings as per the
 [original
paper](https://arxiv.org/abs/2302.03027) 
. Used in discovering the edit direction.
* **height** 
 (
 `int` 
 ,
 *optional* 
 , defaults to self.unet.config.sample\_size \* self.vae\_scale\_factor) —
The height in pixels of the generated image.
* **width** 
 (
 `int` 
 ,
 *optional* 
 , defaults to self.unet.config.sample\_size \* self.vae\_scale\_factor) —
The width in pixels of the generated image.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 50) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 7.5) —
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
* **latents** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
tensor will ge generated by sampling using the supplied random
 `generator` 
.
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
* **cross\_attention\_guidance\_amount** 
 (
 `float` 
 , defaults to 0.1) —
Amount of guidance needed from the reference cross-attention maps.
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
 [StableDiffusionPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) 
 instead of a
plain tuple.
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
* **clip\_skip** 
 (
 `int` 
 ,
 *optional* 
 ) —
Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
the output of the pre-final layer will be used for computing the prompt embeddings.




 Returns
 




 export const metadata = 'undefined';
 

[StableDiffusionPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




[StableDiffusionPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) 
 if
 `return_dict` 
 is True, otherwise a
 `tuple. When returning a tuple, the first element is a list with the generated images, and the second element is a list of` 
 bool
 `s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw) content, according to the` 
 safety\_checker`.
 



 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> import requests
>>> import torch

>>> from diffusers import DDIMScheduler, StableDiffusionPix2PixZeroPipeline


>>> def download(embedding\_url, local\_filepath):
...     r = requests.get(embedding_url)
...     with open(local_filepath, "wb") as f:
...         f.write(r.content)


>>> model_ckpt = "CompVis/stable-diffusion-v1-4"
>>> pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16)
>>> pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
>>> pipeline.to("cuda")

>>> prompt = "a high resolution painting of a cat in the style of van gough"
>>> source_emb_url = "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/cat.pt"
>>> target_emb_url = "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/dog.pt"

>>> for url in [source_emb_url, target_emb_url]:
...     download(url, url.split("/")[-1])

>>> src_embeds = torch.load(source_emb_url.split("/")[-1])
>>> target_embeds = torch.load(target_emb_url.split("/")[-1])
>>> images = pipeline(
...     prompt,
...     source_embeds=src_embeds,
...     target_embeds=target_embeds,
...     num_inference_steps=50,
...     cross_attention_guidance_amount=0.15,
... ).images

>>> images[0].save("edited\_image\_dog.png")
```


#### 




 construct\_direction




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_pix2pix_zero.py#L699)



 (
 


 embs\_source
 
 : Tensor
 




 embs\_target
 
 : Tensor
 



 )
 




 Constructs the edit direction to steer the image generation process semantically.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_pix2pix_zero.py#L405)



 (
 


 prompt
 


 device
 


 num\_images\_per\_prompt
 


 do\_classifier\_free\_guidance
 


 negative\_prompt
 
 = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 lora\_scale
 
 : typing.Optional[float] = None
 




 clip\_skip
 
 : typing.Optional[int] = None
 



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
device — (
 `torch.device` 
 ):
torch device
* **num\_images\_per\_prompt** 
 (
 `int` 
 ) —
number of images that should be generated per prompt
* **do\_classifier\_free\_guidance** 
 (
 `bool` 
 ) —
whether to use classifier free guidance or not
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
* **lora\_scale** 
 (
 `float` 
 ,
 *optional* 
 ) —
A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
* **clip\_skip** 
 (
 `int` 
 ,
 *optional* 
 ) —
Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
the output of the pre-final layer will be used for computing the prompt embeddings.


 Encodes the prompt into text encoder hidden states.
 




#### 




 generate\_caption




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_pix2pix_zero.py#L679)



 (
 


 images
 




 )
 




 Generates caption for a given image.
 




#### 




 invert




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_pix2pix_zero.py#L1100)



 (
 


 prompt
 
 : typing.Optional[str] = None
 




 image
 
 : typing.Union[PIL.Image.Image, numpy.ndarray, torch.FloatTensor, typing.List[PIL.Image.Image], typing.List[numpy.ndarray], typing.List[torch.FloatTensor]] = None
 




 num\_inference\_steps
 
 : int = 50
 




 guidance\_scale
 
 : float = 1
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 cross\_attention\_guidance\_amount
 
 : float = 0.1
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : typing.Optional[int] = 1
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 lambda\_auto\_corr
 
 : float = 20.0
 




 lambda\_kl
 
 : float = 20.0
 




 num\_reg\_steps
 
 : int = 5
 




 num\_auto\_corr\_rolls
 
 : int = 5
 



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
The prompt or prompts to guide the image generation. If not defined, one has to pass
 `prompt_embeds` 
.
instead.
* **image** 
 (
 `torch.FloatTensor` 
`np.ndarray` 
 ,
 `PIL.Image.Image` 
 ,
 `List[torch.FloatTensor]` 
 ,
 `List[PIL.Image.Image]` 
 , or
 `List[np.ndarray]` 
 ) —
 `Image` 
 , or tensor representing an image batch which will be used for conditioning. Can also accept
image latents as
 `image` 
 , if passing latents directly, it will not be encoded again.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 50) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1) —
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
* **latents** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
tensor will ge generated by sampling using the supplied random
 `generator` 
.
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
* **cross\_attention\_guidance\_amount** 
 (
 `float` 
 , defaults to 0.1) —
Amount of guidance needed from the reference cross-attention maps.
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
 [StableDiffusionPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) 
 instead of a
plain tuple.
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
* **lambda\_auto\_corr** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 20.0) —
Lambda parameter to control auto correction
* **lambda\_kl** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 20.0) —
Lambda parameter to control Kullback–Leibler divergence output
* **num\_reg\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 5) —
Number of regularization loss steps
* **num\_auto\_corr\_rolls** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 5) —
Number of auto correction roll steps


 Function used to generate inverted latents given a prompt and image.
 


 Examples:
 



```
>>> import torch
>>> from transformers import BlipForConditionalGeneration, BlipProcessor
>>> from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline

>>> import requests
>>> from PIL import Image

>>> captioner_id = "Salesforce/blip-image-captioning-base"
>>> processor = BlipProcessor.from_pretrained(captioner_id)
>>> model = BlipForConditionalGeneration.from_pretrained(
...     captioner_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
... )

>>> sd_model_ckpt = "CompVis/stable-diffusion-v1-4"
>>> pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
...     sd_model_ckpt,
...     caption_generator=model,
...     caption_processor=processor,
...     torch_dtype=torch.float16,
...     safety_checker=None,
... )

>>> pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
>>> pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
>>> pipeline.enable_model_cpu_offload()

>>> img_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/test\_images/cats/cat\_6.png"

>>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB").resize((512, 512))
>>> # generate caption
>>> caption = pipeline.generate_caption(raw_image)

>>> # "a photography of a cat with flowers and dai dai daie - daie - daie kasaii"
>>> inv_latents = pipeline.invert(caption, image=raw_image).latents
>>> # we need to generate source and target embeds

>>> source_prompts = ["a cat sitting on the street", "a cat playing in the field", "a face of a cat"]

>>> target_prompts = ["a dog sitting on the street", "a dog playing in the field", "a face of a dog"]

>>> source_embeds = pipeline.get_embeds(source_prompts)
>>> target_embeds = pipeline.get_embeds(target_prompts)
>>> # the latents can then be used to edit a real image
>>> # when using Stable Diffusion 2 or other models that use v-prediction
>>> # set `cross\_attention\_guidance\_amount` to 0.01 or less to avoid input latent gradient explosion

>>> image = pipeline(
...     caption,
...     source_embeds=source_embeds,
...     target_embeds=target_embeds,
...     num_inference_steps=50,
...     cross_attention_guidance_amount=0.15,
...     generator=generator,
...     latents=inv_latents,
...     negative_prompt=caption,
... ).images[0]
>>> image.save("edited\_image.png")
```