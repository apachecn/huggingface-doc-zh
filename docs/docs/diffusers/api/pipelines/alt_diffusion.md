# AltDiffusion

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/alt_diffusion>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/alt_diffusion>



 AltDiffusion was proposed in
 [AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities](https://huggingface.co/papers/2211.06679) 
 by Zhongzhi Chen, Guang Liu, Bo-Wen Zhang, Fulong Ye, Qinghong Yang, Ledell Wu.
 



 The abstract from the paper is:
 



*In this work, we present a conceptually simple and effective method to train a strong bilingual multimodal representation model. Starting from the pretrained multimodal representation model CLIP released by OpenAI, we switched its text encoder with a pretrained multilingual text encoder XLM-R, and aligned both languages and image representations by a two-stage training schema consisting of teacher learning and contrastive learning. We validate our method through evaluations of a wide range of tasks. We set new state-of-the-art performances on a bunch of tasks including ImageNet-CN, Flicker30k- CN, and COCO-CN. Further, we obtain very close performances with CLIP on almost all tasks, suggesting that one can simply alter the text encoder in CLIP for extended capabilities such as multilingual understanding.* 


## Tips




`AltDiffusion` 
 is conceptually the same as
 [Stable Diffusion](./stable_diffusion/overview) 
.
 




 Make sure to check out the Schedulers
 [guide](../../using-diffusers/schedulers) 
 to learn how to explore the tradeoff between scheduler speed and quality, and see the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## AltDiffusionPipeline




### 




 class
 

 diffusers.
 

 AltDiffusionPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion.py#L77)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : RobertaSeriesModelWithTransformation
 




 tokenizer
 
 : XLMRobertaTokenizer
 




 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : KarrasDiffusionSchedulers
 




 safety\_checker
 
 : StableDiffusionSafetyChecker
 




 feature\_extractor
 
 : CLIPImageProcessor
 




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
 `RobertaSeriesModelWithTransformation` 
 ) —
Frozen text-encoder (
 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 
 ).
* **tokenizer** 
 (
 [XLMRobertaTokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer) 
 ) —
A
 `XLMRobertaTokenizer` 
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
 to denoise the encoded image latents. Can be one of
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 ,
 [LMSDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler) 
 , or
 [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler) 
.
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


 Pipeline for text-to-image generation using Alt Diffusion.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



 The pipeline also inherits the following loading methods:
 


* [load\_textual\_inversion()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.load_textual_inversion) 
 for loading textual inversion embeddings
* [load\_lora\_weights()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.load_lora_weights) 
 for loading LoRA weights
* [save\_lora\_weights()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.save_lora_weights) 
 for saving LoRA weights
* [from\_single\_file()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.from_single_file) 
 for loading
 `.ckpt` 
 files



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion.py#L646)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 height
 
 : typing.Optional[int] = None
 




 width
 
 : typing.Optional[int] = None
 




 num\_inference\_steps
 
 : int = 50
 




 guidance\_scale
 
 : float = 7.5
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




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
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 guidance\_rescale
 
 : float = 0.0
 




 clip\_skip
 
 : typing.Optional[int] = None
 




 callback\_on\_step\_end
 
 : typing.Union[typing.Callable[[int, int, typing.Dict], NoneType], NoneType] = None
 




 callback\_on\_step\_end\_tensor\_inputs
 
 : typing.List[str] = ['latents']
 




 \*\*kwargs
 




 )
 

 →
 



 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.AltDiffusionPipelineOutput` 
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
 `~pipelines.stable_diffusion.AltDiffusionPipelineOutput` 
 instead of a
plain tuple.
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
* **guidance\_rescale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
Guidance rescale factor from
 [Common Diffusion Noise Schedules and Sample Steps are
Flawed](https://arxiv.org/pdf/2305.08891.pdf) 
. Guidance rescale factor should fix overexposure when
using zero terminal SNR.
* **clip\_skip** 
 (
 `int` 
 ,
 *optional* 
 ) —
Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
the output of the pre-final layer will be used for computing the prompt embeddings.
* **callback\_on\_step\_end** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
A function that calls at the end of each denoising steps during the inference. The function is called
with the following arguments:
 `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)` 
.
 `callback_kwargs` 
 will include a list of all tensors as specified by
 `callback_on_step_end_tensor_inputs` 
.
* **callback\_on\_step\_end\_tensor\_inputs** 
 (
 `List` 
 ,
 *optional* 
 ) —
The list of tensor inputs for the
 `callback_on_step_end` 
 function. The tensors specified in the list
will be passed as
 `callback_kwargs` 
 argument. You will only be able to include variables listed in the
 `._callback_tensor_inputs` 
 attribute of your pipeine class.




 Returns
 




 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.AltDiffusionPipelineOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is
 `True` 
 ,
 `~pipelines.stable_diffusion.AltDiffusionPipelineOutput` 
 is returned,
otherwise a
 `tuple` 
 is returned where the first element is a list with the generated images and the
second element is a list of
 `bool` 
 s indicating whether the corresponding generated image contains
“not-safe-for-work” (nsfw) content.
 



 The call function to the pipeline for generation.
 


 Examples:
 



```
>>> import torch
>>> from diffusers import AltDiffusionPipeline

>>> pipe = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion-m9", torch_dtype=torch.float16)
>>> pipe = pipe.to("cuda")

>>> # "dark elf princess, highly detailed, d & d, fantasy, highly detailed, digital painting, trending on artstation, concept art, sharp focus, illustration, art by artgerm and greg rutkowski and fuji choko and viktoria gavrilenko and hoang lap"
>>> prompt = "黑暗精灵公主，非常详细，幻想，非常详细，数字绘画，概念艺术，敏锐的焦点，插图"
>>> image = pipe(prompt).images[0]
```


#### 




 disable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion.py#L587)



 (
 

 )
 




 Disables the FreeU mechanism if enabled.
 




#### 




 disable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion.py#L212)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion.py#L227)



 (
 

 )
 




 Disable tiled VAE decoding. If
 `enable_vae_tiling` 
 was previously enabled, this method will go back to
computing decoding in one step.
 




#### 




 enable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion.py#L565)



 (
 


 s1
 
 : float
 




 s2
 
 : float
 




 b1
 
 : float
 




 b2
 
 : float
 



 )
 


 Parameters
 




* **s1** 
 (
 `float` 
 ) —
Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
mitigate “oversmoothing effect” in the enhanced denoising process.
* **s2** 
 (
 `float` 
 ) —
Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
mitigate “oversmoothing effect” in the enhanced denoising process.
* **b1** 
 (
 `float` 
 ) — Scaling factor for stage 1 to amplify the contributions of backbone features.
* **b2** 
 (
 `float` 
 ) — Scaling factor for stage 2 to amplify the contributions of backbone features.


 Enables the FreeU mechanism as in
 <https://arxiv.org/abs/2309.11497>
.
 



 The suffixes after the scaling factors represent the stages where they are being applied.
 



 Please refer to the
 [official repository](https://github.com/ChenyangSi/FreeU) 
 for combinations of the values
that are known to work well for different pipelines such as Alt Diffusion v1, v2, and Alt Diffusion XL.
 




#### 




 enable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion.py#L205)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 enable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion.py#L219)



 (
 

 )
 




 Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion.py#L269)



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




 get\_guidance\_scale\_embedding




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion.py#L591)



 (
 


 w
 


 embedding\_dim
 
 = 512
 




 dtype
 
 = torch.float32
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`torch.FloatTensor` 


 Parameters
 




* **timesteps** 
 (
 `torch.Tensor` 
 ) —
generate embedding vectors at these timesteps
* **embedding\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 512) —
dimension of the embeddings to generate
dtype —
data type of the generated embeddings




 Returns
 




 export const metadata = 'undefined';
 

`torch.FloatTensor` 




 export const metadata = 'undefined';
 




 Embedding vectors with shape
 `(len(timesteps), embedding_dim)` 




 See
 <https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298>


## AltDiffusionImg2ImgPipeline




### 




 class
 

 diffusers.
 

 AltDiffusionImg2ImgPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion_img2img.py#L113)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : RobertaSeriesModelWithTransformation
 




 tokenizer
 
 : XLMRobertaTokenizer
 




 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : KarrasDiffusionSchedulers
 




 safety\_checker
 
 : StableDiffusionSafetyChecker
 




 feature\_extractor
 
 : CLIPImageProcessor
 




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
 `RobertaSeriesModelWithTransformation` 
 ) —
Frozen text-encoder (
 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 
 ).
* **tokenizer** 
 (
 [XLMRobertaTokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer) 
 ) —
A
 `XLMRobertaTokenizer` 
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
 to denoise the encoded image latents. Can be one of
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 ,
 [LMSDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler) 
 , or
 [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler) 
.
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


 Pipeline for text-guided image-to-image generation using Alt Diffusion.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



 The pipeline also inherits the following loading methods:
 


* [load\_textual\_inversion()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.load_textual_inversion) 
 for loading textual inversion embeddings
* [load\_lora\_weights()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.load_lora_weights) 
 for loading LoRA weights
* [save\_lora\_weights()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.save_lora_weights) 
 for saving LoRA weights
* [from\_single\_file()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.from_single_file) 
 for loading
 `.ckpt` 
 files



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion_img2img.py#L700)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 image
 
 : typing.Union[PIL.Image.Image, numpy.ndarray, torch.FloatTensor, typing.List[PIL.Image.Image], typing.List[numpy.ndarray], typing.List[torch.FloatTensor]] = None
 




 strength
 
 : float = 0.8
 




 num\_inference\_steps
 
 : typing.Optional[int] = 50
 




 guidance\_scale
 
 : typing.Optional[float] = 7.5
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_images\_per\_prompt
 
 : typing.Optional[int] = 1
 




 eta
 
 : typing.Optional[float] = 0.0
 




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
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 clip\_skip
 
 : int = None
 




 callback\_on\_step\_end
 
 : typing.Union[typing.Callable[[int, int, typing.Dict], NoneType], NoneType] = None
 




 callback\_on\_step\_end\_tensor\_inputs
 
 : typing.List[str] = ['latents']
 




 \*\*kwargs
 




 )
 

 →
 



 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.AltDiffusionPipelineOutput` 
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
* **image** 
 (
 `torch.FloatTensor` 
 ,
 `PIL.Image.Image` 
 ,
 `np.ndarray` 
 ,
 `List[torch.FloatTensor]` 
 ,
 `List[PIL.Image.Image]` 
 , or
 `List[np.ndarray]` 
 ) —
 `Image` 
 , numpy array or tensor representing an image batch to be used as the starting point. For both
numpy array and pytorch tensor, the expected value range is between
 `[0, 1]` 
 If it’s a tensor or a list
or tensors, the expected shape should be
 `(B, C, H, W)` 
 or
 `(C, H, W)` 
. If it is a numpy array or a
list of arrays, the expected shape should be
 `(B, H, W, C)` 
 or
 `(H, W, C)` 
 It can also accept image
latents as
 `image` 
 , but if passing latents directly it is not encoded again.
* **strength** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.8) —
Indicates extent to transform the reference
 `image` 
. Must be between 0 and 1.
 `image` 
 is used as a
starting point and more noise is added the higher the
 `strength` 
. The number of denoising steps depends
on the amount of noise initially added. When
 `strength` 
 is 1, added noise is maximum and the denoising
process runs for the full number of iterations specified in
 `num_inference_steps` 
. A value of 1
essentially ignores
 `image` 
.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 50) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference. This parameter is modulated by
 `strength` 
.
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
 or
 `List[torch.Generator]` 
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
 `~pipelines.stable_diffusion.AltDiffusionPipelineOutput` 
 instead of a
plain tuple.
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
* **callback\_on\_step\_end** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
A function that calls at the end of each denoising steps during the inference. The function is called
with the following arguments:
 `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)` 
.
 `callback_kwargs` 
 will include a list of all tensors as specified by
 `callback_on_step_end_tensor_inputs` 
.
* **callback\_on\_step\_end\_tensor\_inputs** 
 (
 `List` 
 ,
 *optional* 
 ) —
The list of tensor inputs for the
 `callback_on_step_end` 
 function. The tensors specified in the list
will be passed as
 `callback_kwargs` 
 argument. You will only be able to include variables listed in the
 `._callback_tensor_inputs` 
 attribute of your pipeine class.




 Returns
 




 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.AltDiffusionPipelineOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is
 `True` 
 ,
 `~pipelines.stable_diffusion.AltDiffusionPipelineOutput` 
 is returned,
otherwise a
 `tuple` 
 is returned where the first element is a list with the generated images and the
second element is a list of
 `bool` 
 s indicating whether the corresponding generated image contains
“not-safe-for-work” (nsfw) content.
 



 The call function to the pipeline for generation.
 


 Examples:
 



```
>>> import requests
>>> import torch
>>> from PIL import Image
>>> from io import BytesIO

>>> from diffusers import AltDiffusionImg2ImgPipeline

>>> device = "cuda"
>>> model_id_or_path = "BAAI/AltDiffusion-m9"
>>> pipe = AltDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
>>> pipe = pipe.to(device)

>>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

>>> response = requests.get(url)
>>> init_image = Image.open(BytesIO(response.content)).convert("RGB")
>>> init_image = init_image.resize((768, 512))

>>> # "A fantasy landscape, trending on artstation"
>>> prompt = "幻想风景, artstation"

>>> images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
>>> images[0].save("幻想风景.png")
```


#### 




 disable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion_img2img.py#L645)



 (
 

 )
 




 Disables the FreeU mechanism if enabled.
 




#### 




 enable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion_img2img.py#L623)



 (
 


 s1
 
 : float
 




 s2
 
 : float
 




 b1
 
 : float
 




 b2
 
 : float
 



 )
 


 Parameters
 




* **s1** 
 (
 `float` 
 ) —
Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
mitigate “oversmoothing effect” in the enhanced denoising process.
* **s2** 
 (
 `float` 
 ) —
Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
mitigate “oversmoothing effect” in the enhanced denoising process.
* **b1** 
 (
 `float` 
 ) — Scaling factor for stage 1 to amplify the contributions of backbone features.
* **b2** 
 (
 `float` 
 ) — Scaling factor for stage 2 to amplify the contributions of backbone features.


 Enables the FreeU mechanism as in
 <https://arxiv.org/abs/2309.11497>
.
 



 The suffixes after the scaling factors represent the stages where they are being applied.
 



 Please refer to the
 [official repository](https://github.com/ChenyangSi/FreeU) 
 for combinations of the values
that are known to work well for different pipelines such as Alt Diffusion v1, v2, and Alt Diffusion XL.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion_img2img.py#L278)



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




 get\_guidance\_scale\_embedding




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_alt_diffusion_img2img.py#L649)



 (
 


 w
 


 embedding\_dim
 
 = 512
 




 dtype
 
 = torch.float32
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`torch.FloatTensor` 


 Parameters
 




* **timesteps** 
 (
 `torch.Tensor` 
 ) —
generate embedding vectors at these timesteps
* **embedding\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 512) —
dimension of the embeddings to generate
dtype —
data type of the generated embeddings




 Returns
 




 export const metadata = 'undefined';
 

`torch.FloatTensor` 




 export const metadata = 'undefined';
 




 Embedding vectors with shape
 `(len(timesteps), embedding_dim)` 




 See
 <https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298>


## AltDiffusionPipelineOutput




### 




 class
 

 diffusers.pipelines.alt\_diffusion.
 

 AltDiffusionPipelineOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/alt_diffusion/pipeline_output.py#L14)



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


 Output class for Alt Diffusion pipelines.
 



#### 




 \_\_call\_\_




 (
 


 \*args
 


 \*\*kwargs
 




 )
 




 Call self as a function.