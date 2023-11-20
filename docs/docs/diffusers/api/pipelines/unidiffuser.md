# UniDiffuser

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/unidiffuser>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/unidiffuser>



 The UniDiffuser model was proposed in
 [One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale](https://huggingface.co/papers/2303.06555) 
 by Fan Bao, Shen Nie, Kaiwen Xue, Chongxuan Li, Shi Pu, Yaole Wang, Gang Yue, Yue Cao, Hang Su, Jun Zhu.
 



 The abstract from the
 [paper](https://arxiv.org/abs/2303.06555) 
 is:
 



*This paper proposes a unified diffusion framework (dubbed UniDiffuser) to fit all distributions relevant to a set of multi-modal data in one model. Our key insight is — learning diffusion models for marginal, conditional, and joint distributions can be unified as predicting the noise in the perturbed data, where the perturbation levels (i.e. timesteps) can be different for different modalities. Inspired by the unified view, UniDiffuser learns all distributions simultaneously with a minimal modification to the original diffusion model — perturbs data in all modalities instead of a single modality, inputs individual timesteps in different modalities, and predicts the noise of all modalities instead of a single modality. UniDiffuser is parameterized by a transformer for diffusion models to handle input types of different modalities. Implemented on large-scale paired image-text data, UniDiffuser is able to perform image, text, text-to-image, image-to-text, and image-text pair generation by setting proper timesteps without additional overhead. In particular, UniDiffuser is able to produce perceptually realistic samples in all tasks and its quantitative results (e.g., the FID and CLIP score) are not only superior to existing general-purpose models but also comparable to the bespoken models (e.g., Stable Diffusion and DALL-E 2) in representative tasks (e.g., text-to-image generation).* 




 You can find the original codebase at
 [thu-ml/unidiffuser](https://github.com/thu-ml/unidiffuser) 
 and additional checkpoints at
 [thu-ml](https://huggingface.co/thu-ml) 
.
 




 There is currently an issue on PyTorch 1.X where the output images are all black or the pixel values become
 `NaNs` 
. This issue can be mitigated by switching to PyTorch 2.X.
 




 This pipeline was contributed by
 [dg845](https://github.com/dg845) 
. ❤️
 


## Usage Examples




 Because the UniDiffuser model is trained to model the joint distribution of (image, text) pairs, it is capable of performing a diverse range of generation tasks:
 


### 


 Unconditional Image and Text Generation



 Unconditional generation (where we start from only latents sampled from a standard Gaussian prior) from a
 [UniDiffuserPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/unidiffuser#diffusers.UniDiffuserPipeline) 
 will produce a (image, text) pair:
 



```
import torch

from diffusers import UniDiffuserPipeline

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# Unconditional image and text generation. The generation task is automatically inferred.
sample = pipe(num_inference_steps=20, guidance_scale=8.0)
image = sample.images[0]
text = sample.text[0]
image.save("unidiffuser\_joint\_sample\_image.png")
print(text)
```



 This is also called “joint” generation in the UniDiffusers paper, since we are sampling from the joint image-text distribution.
 



 Note that the generation task is inferred from the inputs used when calling the pipeline.
It is also possible to manually specify the unconditional generation task (“mode”) manually with
 [UniDiffuserPipeline.set\_joint\_mode()](/docs/diffusers/v0.23.0/en/api/pipelines/unidiffuser#diffusers.UniDiffuserPipeline.set_joint_mode) 
 :
 



```
# Equivalent to the above.
pipe.set_joint_mode()
sample = pipe(num_inference_steps=20, guidance_scale=8.0)
```



 When the mode is set manually, subsequent calls to the pipeline will use the set mode without attempting the infer the mode.
You can reset the mode with
 [UniDiffuserPipeline.reset\_mode()](/docs/diffusers/v0.23.0/en/api/pipelines/unidiffuser#diffusers.UniDiffuserPipeline.reset_mode) 
 , after which the pipeline will once again infer the mode.
 



 You can also generate only an image or only text (which the UniDiffuser paper calls “marginal” generation since we sample from the marginal distribution of images and text, respectively):
 



```
# Unlike other generation tasks, image-only and text-only generation don't use classifier-free guidance
# Image-only generation
pipe.set_image_mode()
sample_image = pipe(num_inference_steps=20).images[0]
# Text-only generation
pipe.set_text_mode()
sample_text = pipe(num_inference_steps=20).text[0]
```


### 


 Text-to-Image Generation



 UniDiffuser is also capable of sampling from conditional distributions; that is, the distribution of images conditioned on a text prompt or the distribution of texts conditioned on an image.
Here is an example of sampling from the conditional image distribution (text-to-image generation or text-conditioned image generation):
 



```
import torch

from diffusers import UniDiffuserPipeline

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# Text-to-image generation
prompt = "an elephant under the sea"

sample = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.0)
t2i_image = sample.images[0]
t2i_image.save("unidiffuser\_text2img\_sample\_image.png")
```



 The
 `text2img` 
 mode requires that either an input
 `prompt` 
 or
 `prompt_embeds` 
 be supplied. You can set the
 `text2img` 
 mode manually with
 [UniDiffuserPipeline.set\_text\_to\_image\_mode()](/docs/diffusers/v0.23.0/en/api/pipelines/unidiffuser#diffusers.UniDiffuserPipeline.set_text_to_image_mode) 
.
 


### 


 Image-to-Text Generation



 Similarly, UniDiffuser can also produce text samples given an image (image-to-text or image-conditioned text generation):
 



```
import torch

from diffusers import UniDiffuserPipeline
from diffusers.utils import load_image

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# Image-to-text generation
image_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unidiffuser/unidiffuser\_example\_image.jpg"
init_image = load_image(image_url).resize((512, 512))

sample = pipe(image=init_image, num_inference_steps=20, guidance_scale=8.0)
i2t_text = sample.text[0]
print(i2t_text)
```



 The
 `img2text` 
 mode requires that an input
 `image` 
 be supplied. You can set the
 `img2text` 
 mode manually with
 [UniDiffuserPipeline.set\_image\_to\_text\_mode()](/docs/diffusers/v0.23.0/en/api/pipelines/unidiffuser#diffusers.UniDiffuserPipeline.set_image_to_text_mode) 
.
 


### 


 Image Variation



 The UniDiffuser authors suggest performing image variation through a “round-trip” generation method, where given an input image, we first perform an image-to-text generation, and the perform a text-to-image generation on the outputs of the first generation.
This produces a new image which is semantically similar to the input image:
 



```
import torch

from diffusers import UniDiffuserPipeline
from diffusers.utils import load_image

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# Image variation can be performed with a image-to-text generation followed by a text-to-image generation:
# 1. Image-to-text generation
image_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unidiffuser/unidiffuser\_example\_image.jpg"
init_image = load_image(image_url).resize((512, 512))

sample = pipe(image=init_image, num_inference_steps=20, guidance_scale=8.0)
i2t_text = sample.text[0]
print(i2t_text)

# 2. Text-to-image generation
sample = pipe(prompt=i2t_text, num_inference_steps=20, guidance_scale=8.0)
final_image = sample.images[0]
final_image.save("unidiffuser\_image\_variation\_sample.png")
```


### 


 Text Variation



 Similarly, text variation can be performed on an input prompt with a text-to-image generation followed by a image-to-text generation:
 



```
import torch

from diffusers import UniDiffuserPipeline

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# Text variation can be performed with a text-to-image generation followed by a image-to-text generation:
# 1. Text-to-image generation
prompt = "an elephant under the sea"

sample = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.0)
t2i_image = sample.images[0]
t2i_image.save("unidiffuser\_text2img\_sample\_image.png")

# 2. Image-to-text generation
sample = pipe(image=t2i_image, num_inference_steps=20, guidance_scale=8.0)
final_prompt = sample.text[0]
print(final_prompt)
```


## UniDiffuserPipeline




### 




 class
 

 diffusers.
 

 UniDiffuserPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py#L51)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : CLIPTextModel
 




 image\_encoder
 
 : CLIPVisionModelWithProjection
 




 clip\_image\_processor
 
 : CLIPImageProcessor
 




 clip\_tokenizer
 
 : CLIPTokenizer
 




 text\_decoder
 
 : UniDiffuserTextDecoder
 




 text\_tokenizer
 
 : GPT2Tokenizer
 




 unet
 
 : UniDiffuserModel
 




 scheduler
 
 : KarrasDiffusionSchedulers
 



 )
 


 Parameters
 




* **vae** 
 (
 [AutoencoderKL](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL) 
 ) —
Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations. This
is part of the UniDiffuser image representation along with the CLIP vision encoding.
* **text\_encoder** 
 (
 `CLIPTextModel` 
 ) —
Frozen text-encoder (
 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 
 ).
* **image\_encoder** 
 (
 `CLIPVisionModel` 
 ) —
A
 [CLIPVisionModel](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPVisionModel) 
 to encode images as part of its image representation along with the VAE
latent representation.
* **image\_processor** 
 (
 `CLIPImageProcessor` 
 ) —
 [CLIPImageProcessor](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPImageProcessor) 
 to preprocess an image before CLIP encoding it with
 `image_encoder` 
.
* **clip\_tokenizer** 
 (
 `CLIPTokenizer` 
 ) —
A
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPTokenizer) 
 to tokenize the prompt before encoding it with
 `text_encoder` 
.
* **text\_decoder** 
 (
 `UniDiffuserTextDecoder` 
 ) —
Frozen text decoder. This is a GPT-style model which is used to generate text from the UniDiffuser
embedding.
* **text\_tokenizer** 
 (
 `GPT2Tokenizer` 
 ) —
A
 [GPT2Tokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/gpt2#transformers.GPT2Tokenizer) 
 to decode text for text generation; used along with the
 `text_decoder` 
.
* **unet** 
 (
 `UniDiffuserModel` 
 ) —
A
 [U-ViT](https://github.com/baofff/U-ViT) 
 model with UNNet-style skip connections between transformer
layers to denoise the encoded image latents.
* **scheduler** 
 (
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 ) —
A scheduler to be used in combination with
 `unet` 
 to denoise the encoded image and/or text latents. The
original UniDiffuser paper uses the
 [DPMSolverMultistepScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler) 
 scheduler.


 Pipeline for a bimodal image-text model which supports unconditional text and image generation, text-conditioned
image generation, image-conditioned text generation, and joint image-text generation.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py#L1079)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 image
 
 : typing.Union[torch.FloatTensor, PIL.Image.Image, NoneType] = None
 




 height
 
 : typing.Optional[int] = None
 




 width
 
 : typing.Optional[int] = None
 




 data\_type
 
 : typing.Optional[int] = 1
 




 num\_inference\_steps
 
 : int = 50
 




 guidance\_scale
 
 : float = 8.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_images\_per\_prompt
 
 : typing.Optional[int] = 1
 




 num\_prompts\_per\_image
 
 : typing.Optional[int] = 1
 




 eta
 
 : float = 0.0
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 prompt\_latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 vae\_latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 clip\_latents
 
 : typing.Optional[torch.FloatTensor] = None
 




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
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[ImageTextPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/unidiffuser#diffusers.ImageTextPipelineOutput) 
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
The prompt or prompts to guide image generation. If not defined, you need to pass
 `prompt_embeds` 
.
Required for text-conditioned image generation (
 `text2img` 
 ) mode.
* **image** 
 (
 `torch.FloatTensor` 
 or
 `PIL.Image.Image` 
 ,
 *optional* 
 ) —
 `Image` 
 or tensor representing an image batch. Required for image-conditioned text generation
(
 `img2text` 
 ) mode.
* **height** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `self.unet.config.sample_size * self.vae_scale_factor` 
 ) —
The height in pixels of the generated image.
* **width** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `self.unet.config.sample_size * self.vae_scale_factor` 
 ) —
The width in pixels of the generated image.
* **data\_type** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The data type (either 0 or 1). Only used if you are loading a checkpoint which supports a data type
embedding; this is added for compatibility with the
 [UniDiffuser-v1](https://huggingface.co/thu-ml/unidiffuser-v1) 
 checkpoint.
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
 , defaults to 8.0) —
A higher guidance scale value encourages the model to generate images closely linked to the text
 `prompt` 
 at the expense of lower image quality. Guidance scale is enabled when
 `guidance_scale > 1` 
.
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to guide what to not include in image generation. If not defined, you need to
pass
 `negative_prompt_embeds` 
 instead. Ignored when not using guidance (
 `guidance_scale < 1` 
 ). Used in
text-conditioned image generation (
 `text2img` 
 ) mode.
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt. Used in
 `text2img` 
 (text-conditioned image generation) and
 `img` 
 mode. If the mode is joint and both
 `num_images_per_prompt` 
 and
 `num_prompts_per_image` 
 are
supplied,
 `min(num_images_per_prompt, num_prompts_per_image)` 
 samples are generated.
* **num\_prompts\_per\_image** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of prompts to generate per image. Used in
 `img2text` 
 (image-conditioned text generation) and
 `text` 
 mode. If the mode is joint and both
 `num_images_per_prompt` 
 and
 `num_prompts_per_image` 
 are
supplied,
 `min(num_images_per_prompt, num_prompts_per_image)` 
 samples are generated.
* **eta** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
Corresponds to parameter eta (η) from the
 [DDIM](https://arxiv.org/abs/2010.02502) 
 paper. Only applies
to the
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 , and is ignored in other schedulers.
* **generator** 
 (
 `torch.Generator` 
 or
 `List[torch.Generator]` 
 ,
 *optional* 
 ) —
A
 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)
 to make
generation deterministic.
* **latents** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for joint
image-text generation. Can be used to tweak the same generation with different prompts. If not
provided, a latents tensor is generated by sampling using the supplied random
 `generator` 
. This assumes
a full set of VAE, CLIP, and text latents, if supplied, overrides the value of
 `prompt_latents` 
 ,
 `vae_latents` 
 , and
 `clip_latents` 
.
* **prompt\_latents** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for text
generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
tensor is generated by sampling using the supplied random
 `generator` 
.
* **vae\_latents** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
tensor is generated by sampling using the supplied random
 `generator` 
.
* **clip\_latents** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
tensor is generated by sampling using the supplied random
 `generator` 
.
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
provided, text embeddings are generated from the
 `prompt` 
 input argument. Used in text-conditioned
image generation (
 `text2img` 
 ) mode.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
not provided,
 `negative_prompt_embeds` 
 are be generated from the
 `negative_prompt` 
 input argument. Used
in text-conditioned image generation (
 `text2img` 
 ) mode.
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pil"` 
 ) —
The output format of the generated image. Choose between
 `PIL.Image` 
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
 [ImageTextPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/unidiffuser#diffusers.ImageTextPipelineOutput) 
 instead of a plain tuple.
* **callback** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
A function that calls every
 `callback_steps` 
 steps during inference. The function is called with the
following arguments:
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
 function is called. If not specified, the callback is called at
every step.




 Returns
 




 export const metadata = 'undefined';
 

[ImageTextPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/unidiffuser#diffusers.ImageTextPipelineOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is
 `True` 
 ,
 [ImageTextPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/unidiffuser#diffusers.ImageTextPipelineOutput) 
 is returned, otherwise a
 `tuple` 
 is returned where the first element is a list with the generated images and the second element
is a list of generated texts.
 



 The call function to the pipeline for generation.
 




#### 




 disable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py#L147)



 (
 

 )
 




 Disable sliced VAE decoding. If
 `enable_vae_slicing` 
 was previously enabled, this method will go back to
computing decoding in one step.
 




#### 




 disable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py#L164)



 (
 

 )
 




 Disable tiled VAE decoding. If
 `enable_vae_tiling` 
 was previously enabled, this method will go back to
computing decoding in one step.
 




#### 




 enable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py#L139)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 enable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py#L155)



 (
 

 )
 




 Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py#L382)



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




 reset\_mode




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py#L268)



 (
 

 )
 




 Removes a manually set mode; after calling this, the pipeline will infer the mode from inputs.
 




#### 




 set\_image\_mode




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py#L252)



 (
 

 )
 




 Manually set the generation mode to unconditional (“marginal”) image generation.
 




#### 




 set\_image\_to\_text\_mode




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py#L260)



 (
 

 )
 




 Manually set the generation mode to image-conditioned text generation.
 




#### 




 set\_joint\_mode




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py#L264)



 (
 

 )
 




 Manually set the generation mode to unconditional joint image-text generation.
 




#### 




 set\_text\_mode




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py#L248)



 (
 

 )
 




 Manually set the generation mode to unconditional (“marginal”) text generation.
 




#### 




 set\_text\_to\_image\_mode




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py#L256)



 (
 

 )
 




 Manually set the generation mode to text-conditioned image generation.
 


## ImageTextPipelineOutput




### 




 class
 

 diffusers.
 

 ImageTextPipelineOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py#L34)



 (
 


 images
 
 : typing.Union[typing.List[PIL.Image.Image], numpy.ndarray, NoneType]
 




 text
 
 : typing.Union[typing.List[str], typing.List[typing.List[str]], NoneType]
 



 )
 


 Parameters
 




* **images** 
 (
 `List[PIL.Image.Image]` 
 or
 `np.ndarray` 
 ) —
List of denoised PIL images of length
 `batch_size` 
 or NumPy array of shape
 `(batch_size, height, width, num_channels)` 
.
* **text** 
 (
 `List[str]` 
 or
 `List[List[str]]` 
 ) —
List of generated text strings of length
 `batch_size` 
 or a list of list of strings whose outer list has
length
 `batch_size` 
.


 Output class for joint image-text pipelines.