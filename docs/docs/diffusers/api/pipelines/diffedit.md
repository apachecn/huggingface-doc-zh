# DiffEdit

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/diffedit>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/diffedit>



[DiffEdit: Diffusion-based semantic image editing with mask guidance](https://huggingface.co/papers/2210.11427) 
 is by Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord.
 



 The abstract from the paper is:
 



*Image generation has recently seen tremendous advances, with diffusion models allowing to synthesize convincing images for a large variety of text prompts. In this article, we propose DiffEdit, a method to take advantage of text-conditioned diffusion models for the task of semantic image editing, where the goal is to edit an image based on a text query. Semantic image editing is an extension of image generation, with the additional constraint that the generated image should be as similar as possible to a given input image. Current editing methods based on diffusion models usually require to provide a mask, making the task much easier by treating it as a conditional inpainting task. In contrast, our main contribution is able to automatically generate a mask highlighting regions of the input image that need to be edited, by contrasting predictions of a diffusion model conditioned on different text prompts. Moreover, we rely on latent inference to preserve content in those regions of interest and show excellent synergies with mask-based diffusion. DiffEdit achieves state-of-the-art editing performance on ImageNet. In addition, we evaluate semantic image editing in more challenging settings, using images from the COCO dataset as well as text-based generated images.* 




 The original codebase can be found at
 [Xiang-cd/DiffEdit-stable-diffusion](https://github.com/Xiang-cd/DiffEdit-stable-diffusion) 
 , and you can try it out in this
 [demo](https://blog.problemsolversguild.com/technical/research/2022/11/02/DiffEdit-Implementation.html) 
.
 



 This pipeline was contributed by
 [clarencechen](https://github.com/clarencechen) 
. ❤️
 


## Tips



* The pipeline can generate masks that can be fed into other inpainting pipelines.
* In order to generate an image using this pipeline, both an image mask (source and target prompts can be manually specified or generated, and passed to
 [generate\_mask()](/docs/diffusers/v0.23.0/en/api/pipelines/diffedit#diffusers.StableDiffusionDiffEditPipeline.generate_mask) 
 )
and a set of partially inverted latents (generated using
 [invert()](/docs/diffusers/v0.23.0/en/api/pipelines/diffedit#diffusers.StableDiffusionDiffEditPipeline.invert) 
 )
 *must* 
 be provided as arguments when calling the pipeline to generate the final edited image.
* The function
 [generate\_mask()](/docs/diffusers/v0.23.0/en/api/pipelines/diffedit#diffusers.StableDiffusionDiffEditPipeline.generate_mask) 
 exposes two prompt arguments,
 `source_prompt` 
 and
 `target_prompt` 
 that let you control the locations of the semantic edits in the final image to be generated. Let’s say,
you wanted to translate from “cat” to “dog”. In this case, the edit direction will be “cat -> dog”. To reflect
this in the generated mask, you simply have to set the embeddings related to the phrases including “cat” to
 `source_prompt` 
 and “dog” to
 `target_prompt` 
.
* When generating partially inverted latents using
 `invert` 
 , assign a caption or text embedding describing the
overall image to the
 `prompt` 
 argument to help guide the inverse latent sampling process. In most cases, the
source concept is sufficiently descriptive to yield good results, but feel free to explore alternatives.
* When calling the pipeline to generate the final edited image, assign the source concept to
 `negative_prompt` 
 and the target concept to
 `prompt` 
. Taking the above example, you simply have to set the embeddings related to
the phrases including “cat” to
 `negative_prompt` 
 and “dog” to
 `prompt` 
.
* If you wanted to reverse the direction in the example above, i.e., “dog -> cat”, then it’s recommended to:
	+ Swap the
	 `source_prompt` 
	 and
	 `target_prompt` 
	 in the arguments to
	 `generate_mask` 
	.
	+ Change the input prompt in
	 [invert()](/docs/diffusers/v0.23.0/en/api/pipelines/diffedit#diffusers.StableDiffusionDiffEditPipeline.invert) 
	 to include “dog”.
	+ Swap the
	 `prompt` 
	 and
	 `negative_prompt` 
	 in the arguments to call the pipeline to generate the final edited image.
* The source and target prompts, or their corresponding embeddings, can also be automatically generated. Please refer to the
 [DiffEdit](/using-diffusers/diffedit) 
 guide for more details.


## StableDiffusionDiffEditPipeline




### 




 class
 

 diffusers.
 

 StableDiffusionDiffEditPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_diffedit.py#L238)



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
 
 : KarrasDiffusionSchedulers
 




 safety\_checker
 
 : StableDiffusionSafetyChecker
 




 feature\_extractor
 
 : CLIPImageProcessor
 




 inverse\_scheduler
 
 : DDIMInverseScheduler
 




 requires\_safety\_checker
 
 : bool = True
 



 )
 


 Parameters
 




* **vae** 
 (
 [AutoencoderKL](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL) 
 ) —
Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
* **text\_encoder** 
 (
 [CLIPTextModel](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPTextModel) 
 ) —
Frozen text-encoder (
 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 
 ).
* **tokenizer** 
 (
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPTokenizer) 
 ) —
A
 `CLIPTokenizer` 
 to tokenize text.
* **unet** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) —
A
 `UNet2DConditionModel` 
 to denoise the encoded image latents.
* **scheduler** 
 (
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 ) —
A scheduler to be used in combination with
 `unet` 
 to denoise the encoded image latents.
* **inverse\_scheduler** 
 (
 [DDIMInverseScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim_inverse#diffusers.DDIMInverseScheduler) 
 ) —
A scheduler to be used in combination with
 `unet` 
 to fill in the unmasked part of the input latents.
* **safety\_checker** 
 (
 `StableDiffusionSafetyChecker` 
 ) —
Classification module that estimates whether generated images could be considered offensive or harmful.
Please refer to the
 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) 
 for more details
about a model’s potential harms.
* **feature\_extractor** 
 (
 [CLIPImageProcessor](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPImageProcessor) 
 ) —
A
 `CLIPImageProcessor` 
 to extract features from generated images; used as inputs to the
 `safety_checker` 
.



 This is an experimental feature!
 




 Pipeline for text-guided image inpainting using Stable Diffusion and DiffEdit.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



 The pipeline also inherits the following loading and saving methods:
 


* [load\_textual\_inversion()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.load_textual_inversion) 
 for loading textual inversion embeddings
* [load\_lora\_weights()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.load_lora_weights) 
 for loading LoRA weights
* [save\_lora\_weights()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.save_lora_weights) 
 for saving LoRA weights



#### 




 generate\_mask




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_diffedit.py#L857)



 (
 


 image
 
 : typing.Union[torch.FloatTensor, PIL.Image.Image] = None
 




 target\_prompt
 
 : typing.Union[typing.List[str], str, NoneType] = None
 




 target\_negative\_prompt
 
 : typing.Union[typing.List[str], str, NoneType] = None
 




 target\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 target\_negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 source\_prompt
 
 : typing.Union[typing.List[str], str, NoneType] = None
 




 source\_negative\_prompt
 
 : typing.Union[typing.List[str], str, NoneType] = None
 




 source\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 source\_negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 num\_maps\_per\_mask
 
 : typing.Optional[int] = 10
 




 mask\_encode\_strength
 
 : typing.Optional[float] = 0.5
 




 mask\_thresholding\_ratio
 
 : typing.Optional[float] = 3.0
 




 num\_inference\_steps
 
 : int = 50
 




 guidance\_scale
 
 : float = 7.5
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 output\_type
 
 : typing.Optional[str] = 'np'
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`List[PIL.Image.Image]` 
 or
 `np.array` 


 Parameters
 




* **image** 
 (
 `PIL.Image.Image` 
 ) —
 `Image` 
 or tensor representing an image batch to be used for computing the mask.
* **target\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to guide semantic mask generation. If not defined, you need to pass
 `prompt_embeds` 
.
* **target\_negative\_prompt** 
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
 ).
* **target\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
provided, text embeddings are generated from the
 `prompt` 
 input argument.
* **target\_negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
not provided,
 `negative_prompt_embeds` 
 are generated from the
 `negative_prompt` 
 input argument.
* **source\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to guide semantic mask generation using DiffEdit. If not defined, you need to
pass
 `source_prompt_embeds` 
 or
 `source_image` 
 instead.
* **source\_negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to guide semantic mask generation away from using DiffEdit. If not defined, you
need to pass
 `source_negative_prompt_embeds` 
 or
 `source_image` 
 instead.
* **source\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings to guide the semantic mask generation. Can be used to easily tweak text
inputs (prompt weighting). If not provided, text embeddings are generated from
 `source_prompt` 
 input
argument.
* **source\_negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings to negatively guide the semantic mask generation. Can be used to easily
tweak text inputs (prompt weighting). If not provided, text embeddings are generated from
 `source_negative_prompt` 
 input argument.
* **num\_maps\_per\_mask** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 10) —
The number of noise maps sampled to generate the semantic mask using DiffEdit.
* **mask\_encode\_strength** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.5) —
The strength of the noise maps sampled to generate the semantic mask using DiffEdit. Must be between 0
and 1.
* **mask\_thresholding\_ratio** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 3.0) —
The maximum multiple of the mean absolute difference used to clamp the semantic guidance map before
mask binarization.
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
A higher guidance scale value encourages the model to generate images closely linked to the text
 `prompt` 
 at the expense of lower image quality. Guidance scale is enabled when
 `guidance_scale > 1` 
.
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
* **cross\_attention\_kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
A kwargs dictionary that if specified is passed along to the
 [AttnProcessor](/docs/diffusers/v0.23.0/en/api/attnprocessor#diffusers.models.attention_processor.AttnProcessor) 
 as defined in
 [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
.




 Returns
 




 export const metadata = 'undefined';
 

`List[PIL.Image.Image]` 
 or
 `np.array` 




 export const metadata = 'undefined';
 




 When returning a
 `List[PIL.Image.Image]` 
 , the list consists of a batch of single-channel binary images
with dimensions
 `(height //self.vae_scale_factor, width //self.vae_scale_factor)` 
. If it’s
 `np.array` 
 , the shape is
 `(batch_size, height //self.vae_scale_factor, width //self.vae_scale_factor)` 
.
 



 Generate a latent mask given a mask prompt, a target prompt, and an image.
 



```
>>> import PIL
>>> import requests
>>> import torch
>>> from io import BytesIO

>>> from diffusers import StableDiffusionDiffEditPipeline


>>> def download\_image(url):
...     response = requests.get(url)
...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


>>> img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"

>>> init_image = download_image(img_url).resize((768, 768))

>>> pipe = StableDiffusionDiffEditPipeline.from_pretrained(
...     "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
... )
>>> pipe = pipe.to("cuda")

>>> pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
>>> pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
>>> pipeline.enable_model_cpu_offload()

>>> mask_prompt = "A bowl of fruits"
>>> prompt = "A bowl of pears"

>>> mask_image = pipe.generate_mask(image=init_image, source_prompt=prompt, target_prompt=mask_prompt)
>>> image_latents = pipe.invert(image=init_image, prompt=mask_prompt).latents
>>> image = pipe(prompt=prompt, mask_image=mask_image, image_latents=image_latents).images[0]
```


#### 




 invert




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_diffedit.py#L1076)



 (
 


 prompt
 
 : typing.Union[typing.List[str], str, NoneType] = None
 




 image
 
 : typing.Union[torch.FloatTensor, PIL.Image.Image] = None
 




 num\_inference\_steps
 
 : int = 50
 




 inpaint\_strength
 
 : float = 0.8
 




 guidance\_scale
 
 : float = 7.5
 




 negative\_prompt
 
 : typing.Union[typing.List[str], str, NoneType] = None
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 decode\_latents
 
 : bool = False
 




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
 
 : int = 0
 




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
The prompt or prompts to guide image generation. If not defined, you need to pass
 `prompt_embeds` 
.
* **image** 
 (
 `PIL.Image.Image` 
 ) —
 `Image` 
 or tensor representing an image batch to produce the inverted latents guided by
 `prompt` 
.
* **inpaint\_strength** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.8) —
Indicates extent of the noising process to run latent inversion. Must be between 0 and 1. When
 `inpaint_strength` 
 is 1, the inversion process is run for the full number of iterations specified in
 `num_inference_steps` 
.
 `image` 
 is used as a reference for the inversion process, and adding more noise
increases
 `inpaint_strength` 
. If
 `inpaint_strength` 
 is 0, no inpainting occurs.
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
 ).
* **generator** 
 (
 `torch.Generator` 
 ,
 *optional* 
 ) —
A
 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)
 to make
generation deterministic.
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
provided, text embeddings are generated from the
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
not provided,
 `negative_prompt_embeds` 
 are generated from the
 `negative_prompt` 
 input argument.
* **decode\_latents** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether or not to decode the inverted latents into a generated image. Setting this argument to
 `True` 
 decodes all inverted latents for each timestep into a list of generated images.
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
 `~pipelines.stable_diffusion.DiffEditInversionPipelineOutput` 
 instead of a
plain tuple.
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
* **cross\_attention\_kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
A kwargs dictionary that if specified is passed along to the
 [AttnProcessor](/docs/diffusers/v0.23.0/en/api/attnprocessor#diffusers.models.attention_processor.AttnProcessor) 
 as defined in
 [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
.
* **lambda\_auto\_corr** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 20.0) —
Lambda parameter to control auto correction.
* **lambda\_kl** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 20.0) —
Lambda parameter to control Kullback-Leibler divergence output.
* **num\_reg\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 0) —
Number of regularization loss steps.
* **num\_auto\_corr\_rolls** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 5) —
Number of auto correction roll steps.


 Generate inverted latents given a prompt and image.
 



```
>>> import PIL
>>> import requests
>>> import torch
>>> from io import BytesIO

>>> from diffusers import StableDiffusionDiffEditPipeline


>>> def download\_image(url):
...     response = requests.get(url)
...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


>>> img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"

>>> init_image = download_image(img_url).resize((768, 768))

>>> pipe = StableDiffusionDiffEditPipeline.from_pretrained(
...     "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
... )
>>> pipe = pipe.to("cuda")

>>> pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
>>> pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
>>> pipeline.enable_model_cpu_offload()

>>> prompt = "A bowl of fruits"

>>> inverted_latents = pipe.invert(image=init_image, prompt=prompt).latents
```


#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_diffedit.py#L1314)



 (
 


 prompt
 
 : typing.Union[typing.List[str], str, NoneType] = None
 




 mask\_image
 
 : typing.Union[torch.FloatTensor, PIL.Image.Image] = None
 




 image\_latents
 
 : typing.Union[torch.FloatTensor, PIL.Image.Image] = None
 




 inpaint\_strength
 
 : typing.Optional[float] = 0.8
 




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
 




 clip\_ckip
 
 : int = None
 



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
The prompt or prompts to guide image generation. If not defined, you need to pass
 `prompt_embeds` 
.
* **mask\_image** 
 (
 `PIL.Image.Image` 
 ) —
 `Image` 
 or tensor representing an image batch to mask the generated image. White pixels in the mask are
repainted, while black pixels are preserved. If
 `mask_image` 
 is a PIL image, it is converted to a
single channel (luminance) before use. If it’s a tensor, it should contain one color channel (L)
instead of 3, so the expected shape would be
 `(B, 1, H, W)` 
.
* **image\_latents** 
 (
 `PIL.Image.Image` 
 or
 `torch.FloatTensor` 
 ) —
Partially noised image latents from the inversion process to be used as inputs for image generation.
* **inpaint\_strength** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.8) —
Indicates extent to inpaint the masked area. Must be between 0 and 1. When
 `inpaint_strength` 
 is 1, the
denoising process is run on the masked area for the full number of iterations specified in
 `num_inference_steps` 
.
 `image_latents` 
 is used as a reference for the masked area, and adding more
noise to a region increases
 `inpaint_strength` 
. If
 `inpaint_strength` 
 is 0, no inpainting occurs.
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
Corresponds to parameter eta (η) from the
 [DDIM](https://arxiv.org/abs/2010.02502) 
 paper. Only applies
to the
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 , and is ignored in other schedulers.
* **generator** 
 (
 `torch.Generator` 
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
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
not provided,
 `negative_prompt_embeds` 
 are generated from the
 `negative_prompt` 
 input argument.
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
 [StableDiffusionPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) 
 instead of a
plain tuple.
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
* **cross\_attention\_kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
A kwargs dictionary that if specified is passed along to the
 `AttentionProcessor` 
 as defined in
 [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
.
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
 




 If
 `return_dict` 
 is
 `True` 
 ,
 [StableDiffusionPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) 
 is returned,
otherwise a
 `tuple` 
 is returned where the first element is a list with the generated images and the
second element is a list of
 `bool` 
 s indicating whether the corresponding generated image contains
“not-safe-for-work” (nsfw) content.
 



 The call function to the pipeline for generation.
 



```
>>> import PIL
>>> import requests
>>> import torch
>>> from io import BytesIO

>>> from diffusers import StableDiffusionDiffEditPipeline


>>> def download\_image(url):
...     response = requests.get(url)
...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


>>> img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"

>>> init_image = download_image(img_url).resize((768, 768))

>>> pipe = StableDiffusionDiffEditPipeline.from_pretrained(
...     "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
... )
>>> pipe = pipe.to("cuda")

>>> pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
>>> pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
>>> pipeline.enable_model_cpu_offload()

>>> mask_prompt = "A bowl of fruits"
>>> prompt = "A bowl of pears"

>>> mask_image = pipe.generate_mask(image=init_image, source_prompt=prompt, target_prompt=mask_prompt)
>>> image_latents = pipe.invert(image=init_image, prompt=mask_prompt).latents
>>> image = pipe(prompt=prompt, mask_image=mask_image, image_latents=image_latents).images[0]
```


#### 




 disable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_diffedit.py#L382)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_diffedit.py#L399)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_diffedit.py#L374)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 enable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_diffedit.py#L390)



 (
 

 )
 




 Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_diffedit.py#L440)



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
 


## StableDiffusionPipelineOutput




### 




 class
 

 diffusers.pipelines.stable\_diffusion.
 

 StableDiffusionPipelineOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_output.py#L11)



 (
 


 images
 
 : typing.Union[typing.List[PIL.Image.Image], numpy.ndarray]
 




 nsfw\_content\_detected
 
 : typing.Optional[typing.List[bool]]
 



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
* **nsfw\_content\_detected** 
 (
 `List[bool]` 
 ) —
List indicating whether the corresponding generated image contains “not-safe-for-work” (nsfw) content or
 `None` 
 if safety checking could not be performed.


 Output class for Stable Diffusion pipelines.