# Semantic Guidance

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/semantic_stable_diffusion>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/semantic_stable_diffusion>



 Semantic Guidance for Diffusion Models was proposed in
 [SEGA: Instructing Diffusion using Semantic Dimensions](https://huggingface.co/papers/2301.12247) 
 and provides strong semantic control over image generation.
Small changes to the text prompt usually result in entirely different output images. However, with SEGA a variety of changes to the image are enabled that can be controlled easily and intuitively, while staying true to the original image composition.
 



 The abstract from the paper is:
 



*Text-to-image diffusion models have recently received a lot of interest for their astonishing ability to produce high-fidelity images from text only. However, achieving one-shot generation that aligns with the user’s intent is nearly impossible, yet small changes to the input prompt often result in very different images. This leaves the user with little semantic control. To put the user in control, we show how to interact with the diffusion process to flexibly steer it along semantic directions. This semantic guidance (SEGA) allows for subtle and extensive edits, changes in composition and style, as well as optimizing the overall artistic conception. We demonstrate SEGA’s effectiveness on a variety of tasks and provide evidence for its versatility and flexibility.* 


 Make sure to check out the Schedulers
 [guide](../../using-diffusers/schedulers) 
 to learn how to explore the tradeoff between scheduler speed and quality, and see the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## SemanticStableDiffusionPipeline




### 




 class
 

 diffusers.
 

 SemanticStableDiffusionPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/semantic_stable_diffusion/pipeline_semantic_stable_diffusion.py#L21)



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
 to denoise the encoded image latents. Can be one of
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 ,
 [LMSDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler) 
 , or
 [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler) 
.
* **safety\_checker** 
 (
 `Q16SafetyChecker` 
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


 Pipeline for text-to-image generation using Stable Diffusion with latent editing.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
 and builds on the
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline) 
. Check the superclass
documentation for the generic methods implemented for all pipelines (downloading, saving, running on a particular
device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/semantic_stable_diffusion/pipeline_semantic_stable_diffusion.py#L210)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]]
 




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
 
 : int = 1
 




 eta
 
 : float = 0.0
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : int = 1
 




 editing\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 editing\_prompt\_embeddings
 
 : typing.Optional[torch.Tensor] = None
 




 reverse\_editing\_direction
 
 : typing.Union[bool, typing.List[bool], NoneType] = False
 




 edit\_guidance\_scale
 
 : typing.Union[float, typing.List[float], NoneType] = 5
 




 edit\_warmup\_steps
 
 : typing.Union[int, typing.List[int], NoneType] = 10
 




 edit\_cooldown\_steps
 
 : typing.Union[int, typing.List[int], NoneType] = None
 




 edit\_threshold
 
 : typing.Union[float, typing.List[float], NoneType] = 0.9
 




 edit\_momentum\_scale
 
 : typing.Optional[float] = 0.1
 




 edit\_mom\_beta
 
 : typing.Optional[float] = 0.4
 




 edit\_weights
 
 : typing.Optional[typing.List[float]] = None
 




 sem\_guidance
 
 : typing.Optional[typing.List[torch.Tensor]] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`~pipelines.semantic_stable_diffusion.SemanticStableDiffusionPipelineOutput` 
 or
 `tuple` 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ) —
The prompt or prompts to guide image generation.
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
* **editing\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to use for semantic guidance. Semantic guidance is disabled by setting
 `editing_prompt = None` 
. Guidance direction of prompt should be specified via
 `reverse_editing_direction` 
.
* **editing\_prompt\_embeddings** 
 (
 `torch.Tensor` 
 ,
 *optional* 
 ) —
Pre-computed embeddings to use for semantic guidance. Guidance direction of embedding should be
specified via
 `reverse_editing_direction` 
.
* **reverse\_editing\_direction** 
 (
 `bool` 
 or
 `List[bool]` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether the corresponding prompt in
 `editing_prompt` 
 should be increased or decreased.
* **edit\_guidance\_scale** 
 (
 `float` 
 or
 `List[float]` 
 ,
 *optional* 
 , defaults to 5) —
Guidance scale for semantic guidance. If provided as a list, values should correspond to
 `editing_prompt` 
.
* **edit\_warmup\_steps** 
 (
 `float` 
 or
 `List[float]` 
 ,
 *optional* 
 , defaults to 10) —
Number of diffusion steps (for each prompt) for which semantic guidance is not applied. Momentum is
calculated for those steps and applied once all warmup periods are over.
* **edit\_cooldown\_steps** 
 (
 `float` 
 or
 `List[float]` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Number of diffusion steps (for each prompt) after which semantic guidance is longer applied.
* **edit\_threshold** 
 (
 `float` 
 or
 `List[float]` 
 ,
 *optional* 
 , defaults to 0.9) —
Threshold of semantic guidance.
* **edit\_momentum\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.1) —
Scale of the momentum to be added to the semantic guidance at each diffusion step. If set to 0.0,
momentum is disabled. Momentum is already built up during warmup (for diffusion steps smaller than
 `sld_warmup_steps` 
 ). Momentum is only added to latent guidance once all warmup periods are finished.
* **edit\_mom\_beta** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.4) —
Defines how semantic guidance momentum builds up.
 `edit_mom_beta` 
 indicates how much of the previous
momentum is kept. Momentum is already built up during warmup (for diffusion steps smaller than
 `edit_warmup_steps` 
 ).
* **edit\_weights** 
 (
 `List[float]` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Indicates how much each individual concept should influence the overall guidance. If no weights are
provided all concepts are applied equally.
* **sem\_guidance** 
 (
 `List[torch.Tensor]` 
 ,
 *optional* 
 ) —
List of pre-generated guidance vectors to be applied at generation. Length of the list has to
correspond to
 `num_inference_steps` 
.




 Returns
 




 export const metadata = 'undefined';
 

`~pipelines.semantic_stable_diffusion.SemanticStableDiffusionPipelineOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is
 `True` 
 ,
 `~pipelines.semantic_stable_diffusion.SemanticStableDiffusionPipelineOutput` 
 is returned, otherwise a
 `tuple` 
 is returned where the first element is a list with the generated images and the second element
is a list of
 `bool` 
 s indicating whether the corresponding generated image contains “not-safe-for-work”
(nsfw) content.
 



 The call function to the pipeline for generation.
 


 Examples:
 



```
>>> import torch
>>> from diffusers import SemanticStableDiffusionPipeline

>>> pipe = SemanticStableDiffusionPipeline.from_pretrained(
...     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
... )
>>> pipe = pipe.to("cuda")

>>> out = pipe(
...     prompt="a photo of the face of a woman",
...     num_images_per_prompt=1,
...     guidance_scale=7,
...     editing_prompt=[
...         "smiling, smile",  # Concepts to apply
...         "glasses, wearing glasses",
...         "curls, wavy hair, curly hair",
...         "beard, full beard, mustache",
...     ],
...     reverse_editing_direction=[
...         False,
...         False,
...         False,
...         False,
...     ],  # Direction of guidance i.e. increase all concepts
...     edit_warmup_steps=[10, 10, 10, 10],  # Warmup period for each concept
...     edit_guidance_scale=[4, 5, 5, 5.4],  # Guidance scale for each concept
...     edit_threshold=[
...         0.99,
...         0.975,
...         0.925,
...         0.96,
...     ],  # Threshold for each concept. Threshold equals the percentile of the latent space that will be discarded. I.e. threshold=0.99 uses 1% of the latent dimensions
...     edit_momentum_scale=0.3,  # Momentum scale that will be added to the latent guidance
...     edit_mom_beta=0.6,  # Momentum beta
...     edit_weights=[1, 1, 1, 1, 1],  # Weights of the individual concepts against each other
... )
>>> image = out.images[0]
```


## StableDiffusionSafePipelineOutput




### 




 class
 

 diffusers.pipelines.semantic\_stable\_diffusion.pipeline\_output.
 

 SemanticStableDiffusionPipelineOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/semantic_stable_diffusion/pipeline_output.py#L11)



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