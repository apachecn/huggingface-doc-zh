# Würstchen

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/wuerstchen>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/wuerstchen>


![](https://github.com/dome272/Wuerstchen/assets/61938694/0617c863-165a-43ee-9303-2a17299a0cf9)

[Würstchen: Efficient Pretraining of Text-to-Image Models](https://huggingface.co/papers/2306.00637) 
 is by Pablo Pernias, Dominic Rampas, Mats L. Richter and Christopher Pal and Marc Aubreville.
 



 The abstract from the paper is:
 



*We introduce Würstchen, a novel technique for text-to-image synthesis that unites competitive performance with unprecedented cost-effectiveness and ease of training on constrained hardware. Building on recent advancements in machine learning, our approach, which utilizes latent diffusion strategies at strong latent image compression rates, significantly reduces the computational burden, typically associated with state-of-the-art models, while preserving, if not enhancing, the quality of generated images. Wuerstchen achieves notable speed improvements at inference time, thereby rendering real-time applications more viable. One of the key advantages of our method lies in its modest training requirements of only 9,200 GPU hours, slashing the usual costs significantly without compromising the end performance. In a comparison against the state-of-the-art, we found the approach to yield strong competitiveness. This paper opens the door to a new line of research that prioritizes both performance and computational accessibility, hence democratizing the use of sophisticated AI technologies. Through Wuerstchen, we demonstrate a compelling stride forward in the realm of text-to-image synthesis, offering an innovative path to explore in future research.* 


## Würstchen Overview




 Würstchen is a diffusion model, whose text-conditional model works in a highly compressed latent space of images. Why is this important? Compressing data can reduce computational costs for both training and inference by magnitudes. Training on 1024x1024 images is way more expensive than training on 32x32. Usually, other works make use of a relatively small compression, in the range of 4x - 8x spatial compression. Würstchen takes this to an extreme. Through its novel design, we achieve a 42x spatial compression. This was unseen before because common methods fail to faithfully reconstruct detailed images after 16x spatial compression. Würstchen employs a two-stage compression, what we call Stage A and Stage B. Stage A is a VQGAN, and Stage B is a Diffusion Autoencoder (more details can be found in the
 [paper](https://huggingface.co/papers/2306.00637) 
 ). A third model, Stage C, is learned in that highly compressed latent space. This training requires fractions of the compute used for current top-performing models, while also allowing cheaper and faster inference.
 


## Würstchen v2 comes to Diffusers




 After the initial paper release, we have improved numerous things in the architecture, training and sampling, making Würstchen competitive to current state-of-the-art models in many ways. We are excited to release this new version together with Diffusers. Here is a list of the improvements.
 


* Higher resolution (1024x1024 up to 2048x2048)
* Faster inference
* Multi Aspect Resolution Sampling
* Better quality



 We are releasing 3 checkpoints for the text-conditional image generation model (Stage C). Those are:
 


* v2-base
* v2-aesthetic
* **(default)** 
 v2-interpolated (50% interpolation between v2-base and v2-aesthetic)



 We recommend using v2-interpolated, as it has a nice touch of both photorealism and aesthetics. Use v2-base for finetunings as it does not have a style bias and use v2-aesthetic for very artistic generations.
A comparison can be seen here:
 


![](https://github.com/dome272/Wuerstchen/assets/61938694/2914830f-cbd3-461c-be64-d50734f4b49d)


## Text-to-Image Generation




 For the sake of usability, Würstchen can be used with a single pipeline. This pipeline can be used as follows:
 



```
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

pipe = AutoPipelineForText2Image.from_pretrained("warp-ai/wuerstchen", torch_dtype=torch.float16).to("cuda")

caption = "Anthropomorphic cat dressed as a fire fighter"
images = pipe(
    caption, 
    width=1024,
    height=1536,
    prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    prior_guidance_scale=4.0,
    num_images_per_prompt=2,
).images
```



 For explanation purposes, we can also initialize the two main pipelines of Würstchen individually. Würstchen consists of 3 stages: Stage C, Stage B, Stage A. They all have different jobs and work only together. When generating text-conditional images, Stage C will first generate the latents in a very compressed latent space. This is what happens in the
 `prior_pipeline` 
. Afterwards, the generated latents will be passed to Stage B, which decompresses the latents into a bigger latent space of a VQGAN. These latents can then be decoded by Stage A, which is a VQGAN, into the pixel-space. Stage B & Stage A are both encapsulated in the
 `decoder_pipeline` 
. For more details, take a look at the
 [paper](https://huggingface.co/papers/2306.00637) 
.
 



```
import torch
from diffusers import WuerstchenDecoderPipeline, WuerstchenPriorPipeline
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

device = "cuda"
dtype = torch.float16
num_images_per_prompt = 2

prior_pipeline = WuerstchenPriorPipeline.from_pretrained(
    "warp-ai/wuerstchen-prior", torch_dtype=dtype
).to(device)
decoder_pipeline = WuerstchenDecoderPipeline.from_pretrained(
    "warp-ai/wuerstchen", torch_dtype=dtype
).to(device)

caption = "Anthropomorphic cat dressed as a fire fighter"
negative_prompt = ""

prior_output = prior_pipeline(
    prompt=caption,
    height=1024,
    width=1536,
    timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=num_images_per_prompt,
)
decoder_output = decoder_pipeline(
    image_embeddings=prior_output.image_embeddings,
    prompt=caption,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
).images
```


## Speed-Up Inference




 You can make use of
 `torch.compile` 
 function and gain a speed-up of about 2-3x:
 



```
prior_pipeline.prior = torch.compile(prior_pipeline.prior, mode="reduce-overhead", fullgraph=True)
decoder_pipeline.decoder = torch.compile(decoder_pipeline.decoder, mode="reduce-overhead", fullgraph=True)
```


## Limitations



* Due to the high compression employed by Würstchen, generations can lack a good amount
of detail. To our human eye, this is especially noticeable in faces, hands etc.
* **Images can only be generated in 128-pixel steps** 
 , e.g. the next higher resolution
after 1024x1024 is 1152x1152
* The model lacks the ability to render correct text in images
* The model often does not achieve photorealism
* Difficult compositional prompts are hard for the model



 The original codebase, as well as experimental ideas, can be found at
 [dome272/Wuerstchen](https://github.com/dome272/Wuerstchen) 
.
 


## WuerstchenCombinedPipeline




### 




 class
 

 diffusers.
 

 WuerstchenCombinedPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen_combined.py#L43)



 (
 


 tokenizer
 
 : CLIPTokenizer
 




 text\_encoder
 
 : CLIPTextModel
 




 decoder
 
 : WuerstchenDiffNeXt
 




 scheduler
 
 : DDPMWuerstchenScheduler
 




 vqgan
 
 : PaellaVQModel
 




 prior\_tokenizer
 
 : CLIPTokenizer
 




 prior\_text\_encoder
 
 : CLIPTextModel
 




 prior\_prior
 
 : WuerstchenPrior
 




 prior\_scheduler
 
 : DDPMWuerstchenScheduler
 



 )
 


 Parameters
 




* **tokenizer** 
 (
 `CLIPTokenizer` 
 ) —
The decoder tokenizer to be used for text inputs.
* **text\_encoder** 
 (
 `CLIPTextModel` 
 ) —
The decoder text encoder to be used for text inputs.
* **decoder** 
 (
 `WuerstchenDiffNeXt` 
 ) —
The decoder model to be used for decoder image generation pipeline.
* **scheduler** 
 (
 `DDPMWuerstchenScheduler` 
 ) —
The scheduler to be used for decoder image generation pipeline.
* **vqgan** 
 (
 `PaellaVQModel` 
 ) —
The VQGAN model to be used for decoder image generation pipeline.
* **prior\_tokenizer** 
 (
 `CLIPTokenizer` 
 ) —
The prior tokenizer to be used for text inputs.
* **prior\_text\_encoder** 
 (
 `CLIPTextModel` 
 ) —
The prior text encoder to be used for text inputs.
* **prior\_prior** 
 (
 `WuerstchenPrior` 
 ) —
The prior model to be used for prior pipeline.
* **prior\_scheduler** 
 (
 `DDPMWuerstchenScheduler` 
 ) —
The scheduler to be used for prior pipeline.


 Combined Pipeline for text-to-image generation using Wuerstchen
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen_combined.py#L143)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 height
 
 : int = 512
 




 width
 
 : int = 512
 




 prior\_num\_inference\_steps
 
 : int = 60
 




 prior\_timesteps
 
 : typing.Optional[typing.List[float]] = None
 




 prior\_guidance\_scale
 
 : float = 4.0
 




 num\_inference\_steps
 
 : int = 12
 




 decoder\_timesteps
 
 : typing.Optional[typing.List[float]] = None
 




 decoder\_guidance\_scale
 
 : float = 0.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 num\_images\_per\_prompt
 
 : int = 1
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 




 prior\_callback\_on\_step\_end
 
 : typing.Union[typing.Callable[[int, int, typing.Dict], NoneType], NoneType] = None
 




 prior\_callback\_on\_step\_end\_tensor\_inputs
 
 : typing.List[str] = ['latents']
 




 callback\_on\_step\_end
 
 : typing.Union[typing.Callable[[int, int, typing.Dict], NoneType], NoneType] = None
 




 callback\_on\_step\_end\_tensor\_inputs
 
 : typing.List[str] = ['latents']
 




 \*\*kwargs
 




 )
 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ) —
The prompt or prompts to guide the image generation for the prior and decoder.
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
if
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
Pre-generated text embeddings for the prior. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings for the prior. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input argument.
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
 , defaults to 512) —
The height in pixels of the generated image.
* **width** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 512) —
The width in pixels of the generated image.
* **prior\_guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
Guidance scale as defined in
 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) 
.
 `prior_guidance_scale` 
 is defined as
 `w` 
 of equation 2. of
 [Imagen
Paper](https://arxiv.org/pdf/2205.11487.pdf) 
. Guidance scale is enabled by setting
 `prior_guidance_scale > 1` 
. Higher guidance scale encourages to generate images that are closely linked
to the text
 `prompt` 
 , usually at the expense of lower image quality.
* **prior\_num\_inference\_steps** 
 (
 `Union[int, Dict[float, int]]` 
 ,
 *optional* 
 , defaults to 60) —
The number of prior denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference. For more specific timestep spacing, you can pass customized
 `prior_timesteps`
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 12) —
The number of decoder denoising steps. More denoising steps usually lead to a higher quality image at
the expense of slower inference. For more specific timestep spacing, you can pass customized
 `timesteps`
* **prior\_timesteps** 
 (
 `List[float]` 
 ,
 *optional* 
 ) —
Custom timesteps to use for the denoising process for the prior. If not defined, equal spaced
 `prior_num_inference_steps` 
 timesteps are used. Must be in descending order.
* **decoder\_timesteps** 
 (
 `List[float]` 
 ,
 *optional* 
 ) —
Custom timesteps to use for the denoising process for the decoder. If not defined, equal spaced
 `num_inference_steps` 
 timesteps are used. Must be in descending order.
* **decoder\_guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
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
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pil"` 
 ) —
The output format of the generate image. Choose between:
 `"pil"` 
 (
 `PIL.Image.Image` 
 ),
 `"np"` 
 (
 `np.array` 
 ) or
 `"pt"` 
 (
 `torch.Tensor` 
 ).
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 instead of a plain tuple.
* **prior\_callback\_on\_step\_end** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
A function that calls at the end of each denoising steps during the inference. The function is called
with the following arguments:
 `prior_callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)` 
.
* **prior\_callback\_on\_step\_end\_tensor\_inputs** 
 (
 `List` 
 ,
 *optional* 
 ) —
The list of tensor inputs for the
 `prior_callback_on_step_end` 
 function. The tensors specified in the
list will be passed as
 `callback_kwargs` 
 argument. You will only be able to include variables listed in
the
 `._callback_tensor_inputs` 
 attribute of your pipeine class.
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


 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> from diffusions import WuerstchenCombinedPipeline

>>> pipe = WuerstchenCombinedPipeline.from_pretrained("warp-ai/Wuerstchen", torch_dtype=torch.float16).to(
...     "cuda"
... )
>>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
>>> images = pipe(prompt=prompt)
```


#### 




 enable\_model\_cpu\_offload




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen_combined.py#L115)



 (
 


 gpu\_id
 
 = 0
 



 )
 




 Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
to
 `enable_sequential_cpu_offload` 
 , this method moves one whole model at a time to the GPU when its
 `forward` 
 method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
 `enable_sequential_cpu_offload` 
 , but performance is much better due to the iterative execution of the
 `unet` 
.
 




#### 




 enable\_sequential\_cpu\_offload




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen_combined.py#L125)



 (
 


 gpu\_id
 
 = 0
 



 )
 




 Offloads all models (
 `unet` 
 ,
 `text_encoder` 
 ,
 `vae` 
 , and
 `safety checker` 
 state dicts) to CPU using 🤗
Accelerate, significantly reducing memory usage. Models are moved to a
 `torch.device('meta')` 
 and loaded on a
GPU only when their specific submodule’s
 `forward` 
 method is called. Offloading happens on a submodule basis.
Memory savings are higher than using
 `enable_model_cpu_offload` 
 , but performance is lower.
 


## WuerstchenPriorPipeline




### 




 class
 

 diffusers.
 

 WuerstchenPriorPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen_prior.py#L65)



 (
 


 tokenizer
 
 : CLIPTokenizer
 




 text\_encoder
 
 : CLIPTextModel
 




 prior
 
 : WuerstchenPrior
 




 scheduler
 
 : DDPMWuerstchenScheduler
 




 latent\_mean
 
 : float = 42.0
 




 latent\_std
 
 : float = 1.0
 




 resolution\_multiple
 
 : float = 42.67
 



 )
 


 Parameters
 




* **prior** 
 (
 `Prior` 
 ) —
The canonical unCLIP prior to approximate the image embedding from the text embedding.
* **text\_encoder** 
 (
 `CLIPTextModelWithProjection` 
 ) —
Frozen text-encoder.
* **tokenizer** 
 (
 `CLIPTokenizer` 
 ) —
Tokenizer of class
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 
.
* **scheduler** 
 (
 `DDPMWuerstchenScheduler` 
 ) —
A scheduler to be used in combination with
 `prior` 
 to generate image embedding.
* **latent\_mean** 
 (‘float’,
 *optional* 
 , defaults to 42.0) —
Mean value for latent diffusers.
* **latent\_std** 
 (‘float’,
 *optional* 
 , defaults to 1.0) —
Standard value for latent diffusers.
* **resolution\_multiple** 
 (‘float’,
 *optional* 
 , defaults to 42.67) —
Default resolution for multiple images generated.


 Pipeline for generating image prior for Wuerstchen.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen_prior.py#L276)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 height
 
 : int = 1024
 




 width
 
 : int = 1024
 




 num\_inference\_steps
 
 : int = 60
 




 timesteps
 
 : typing.List[float] = None
 




 guidance\_scale
 
 : float = 8.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 num\_images\_per\_prompt
 
 : typing.Optional[int] = 1
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 output\_type
 
 : typing.Optional[str] = 'pt'
 




 return\_dict
 
 : bool = True
 




 callback\_on\_step\_end
 
 : typing.Union[typing.Callable[[int, int, typing.Dict], NoneType], NoneType] = None
 




 callback\_on\_step\_end\_tensor\_inputs
 
 : typing.List[str] = ['latents']
 




 \*\*kwargs
 




 )
 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ) —
The prompt or prompts to guide the image generation.
* **height** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1024) —
The height in pixels of the generated image.
* **width** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1024) —
The width in pixels of the generated image.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 60) —
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
 , defaults to 8.0) —
Guidance scale as defined in
 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) 
.
 `decoder_guidance_scale` 
 is defined as
 `w` 
 of equation 2. of
 [Imagen
Paper](https://arxiv.org/pdf/2205.11487.pdf) 
. Guidance scale is enabled by setting
 `decoder_guidance_scale > 1` 
. Higher guidance scale encourages to generate images that are closely
linked to the text
 `prompt` 
 , usually at the expense of lower image quality.
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
if
 `decoder_guidance_scale` 
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
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
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
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pil"` 
 ) —
The output format of the generate image. Choose between:
 `"pil"` 
 (
 `PIL.Image.Image` 
 ),
 `"np"` 
 (
 `np.array` 
 ) or
 `"pt"` 
 (
 `torch.Tensor` 
 ).
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 instead of a plain tuple.
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


 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> import torch
>>> from diffusers import WuerstchenPriorPipeline

>>> prior_pipe = WuerstchenPriorPipeline.from_pretrained(
...     "warp-ai/wuerstchen-prior", torch_dtype=torch.float16
... ).to("cuda")

>>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
>>> prior_output = pipe(prompt)
```


## WuerstchenPriorPipelineOutput




### 




 class
 

 diffusers.pipelines.wuerstchen.pipeline\_wuerstchen\_prior.
 

 WuerstchenPriorPipelineOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen_prior.py#L52)



 (
 


 image\_embeddings
 
 : typing.Union[torch.FloatTensor, numpy.ndarray]
 



 )
 


 Parameters
 




* **image\_embeddings** 
 (
 `torch.FloatTensor` 
 or
 `np.ndarray` 
 ) —
Prior image embeddings for text prompt


 Output class for WuerstchenPriorPipeline.
 


## WuerstchenDecoderPipeline




### 




 class
 

 diffusers.
 

 WuerstchenDecoderPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen.py#L51)



 (
 


 tokenizer
 
 : CLIPTokenizer
 




 text\_encoder
 
 : CLIPTextModel
 




 decoder
 
 : WuerstchenDiffNeXt
 




 scheduler
 
 : DDPMWuerstchenScheduler
 




 vqgan
 
 : PaellaVQModel
 




 latent\_dim\_scale
 
 : float = 10.67
 



 )
 


 Parameters
 




* **tokenizer** 
 (
 `CLIPTokenizer` 
 ) —
The CLIP tokenizer.
* **text\_encoder** 
 (
 `CLIPTextModel` 
 ) —
The CLIP text encoder.
* **decoder** 
 (
 `WuerstchenDiffNeXt` 
 ) —
The WuerstchenDiffNeXt unet decoder.
* **vqgan** 
 (
 `PaellaVQModel` 
 ) —
The VQGAN model.
* **scheduler** 
 (
 `DDPMWuerstchenScheduler` 
 ) —
A scheduler to be used in combination with
 `prior` 
 to generate image embedding.
* **latent\_dim\_scale** 
 (float,
 `optional` 
 , defaults to 10.67) —
Multiplier to determine the VQ latent space size from the image embeddings. If the image embeddings are
height=24 and width=24, the VQ latent shape needs to be height=int(24
 *10.67)=256 and
width=int(24* 
 10.67)=256 in order to match the training conditions.


 Pipeline for generating images from the Wuerstchen model.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen.py#L208)



 (
 


 image\_embeddings
 
 : typing.Union[torch.FloatTensor, typing.List[torch.FloatTensor]]
 




 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 num\_inference\_steps
 
 : int = 12
 




 timesteps
 
 : typing.Optional[typing.List[float]] = None
 




 guidance\_scale
 
 : float = 0.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_images\_per\_prompt
 
 : int = 1
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 




 callback\_on\_step\_end
 
 : typing.Union[typing.Callable[[int, int, typing.Dict], NoneType], NoneType] = None
 




 callback\_on\_step\_end\_tensor\_inputs
 
 : typing.List[str] = ['latents']
 




 \*\*kwargs
 




 )
 


 Parameters
 




* **image\_embedding** 
 (
 `torch.FloatTensor` 
 or
 `List[torch.FloatTensor]` 
 ) —
Image Embeddings either extracted from an image or generated by a Prior Model.
* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ) —
The prompt or prompts to guide the image generation.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 12) —
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
 , defaults to 0.0) —
Guidance scale as defined in
 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) 
.
 `decoder_guidance_scale` 
 is defined as
 `w` 
 of equation 2. of
 [Imagen
Paper](https://arxiv.org/pdf/2205.11487.pdf) 
. Guidance scale is enabled by setting
 `decoder_guidance_scale > 1` 
. Higher guidance scale encourages to generate images that are closely
linked to the text
 `prompt` 
 , usually at the expense of lower image quality.
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
if
 `decoder_guidance_scale` 
 is less than
 `1` 
 ).
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
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
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pil"` 
 ) —
The output format of the generate image. Choose between:
 `"pil"` 
 (
 `PIL.Image.Image` 
 ),
 `"np"` 
 (
 `np.array` 
 ) or
 `"pt"` 
 (
 `torch.Tensor` 
 ).
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 instead of a plain tuple.
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


 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> import torch
>>> from diffusers import WuerstchenPriorPipeline, WuerstchenDecoderPipeline

>>> prior_pipe = WuerstchenPriorPipeline.from_pretrained(
...     "warp-ai/wuerstchen-prior", torch_dtype=torch.float16
... ).to("cuda")
>>> gen_pipe = WuerstchenDecoderPipeline.from_pretrain("warp-ai/wuerstchen", torch_dtype=torch.float16).to(
...     "cuda"
... )

>>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
>>> prior_output = pipe(prompt)
>>> images = gen_pipe(prior_output.image_embeddings, prompt=prompt)
```


## Citation




```
 @misc{pernias2023wuerstchen,
 title={Wuerstchen: Efficient Pretraining of Text-to-Image Models}, 
 author={Pablo Pernias and Dominic Rampas and Mats L. Richter and Christopher Pal and Marc Aubreville},
 year={2023},
 eprint={2306.00637},
 archivePrefix={arXiv},
 primaryClass={cs.CV}
 }
```