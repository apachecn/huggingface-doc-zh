# GLIGEN (Grounded Language-to-Image Generation)

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/stable_diffusion/gligen>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/gligen>



 The GLIGEN model was created by researchers and engineers from
 [University of Wisconsin-Madison, Columbia University, and Microsoft](https://github.com/gligen/GLIGEN) 
. The
 [StableDiffusionGLIGENPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/gligen#diffusers.StableDiffusionGLIGENPipeline) 
 and
 [StableDiffusionGLIGENTextImagePipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/gligen#diffusers.StableDiffusionGLIGENTextImagePipeline) 
 can generate photorealistic images conditioned on grounding inputs. Along with text and bounding boxes with
 [StableDiffusionGLIGENPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/gligen#diffusers.StableDiffusionGLIGENPipeline) 
 , if input images are given,
 [StableDiffusionGLIGENTextImagePipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/gligen#diffusers.StableDiffusionGLIGENTextImagePipeline) 
 can insert objects described by text at the region defined by bounding boxes. Otherwise, it’ll generate an image described by the caption/prompt and insert objects described by text at the region defined by bounding boxes. It’s trained on COCO2014D and COCO2014CD datasets, and the model uses a frozen CLIP ViT-L/14 text encoder to condition itself on grounding inputs.
 



 The abstract from the
 [paper](https://huggingface.co/papers/2301.07093) 
 is:
 



*Large-scale text-to-image diffusion models have made amazing advances. However, the status quo is to use text input alone, which can impede controllability. In this work, we propose GLIGEN, Grounded-Language-to-Image Generation, a novel approach that builds upon and extends the functionality of existing pre-trained text-to-image diffusion models by enabling them to also be conditioned on grounding inputs. To preserve the vast concept knowledge of the pre-trained model, we freeze all of its weights and inject the grounding information into new trainable layers via a gated mechanism. Our model achieves open-world grounded text2img generation with caption and bounding box condition inputs, and the grounding ability generalizes well to novel spatial configurations and concepts. GLIGEN’s zeroshot performance on COCO and LVIS outperforms existing supervised layout-to-image baselines by a large margin.* 


 Make sure to check out the Stable Diffusion
 [Tips](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/overview#tips) 
 section to learn how to explore the tradeoff between scheduler speed and quality and how to reuse pipeline components efficiently!
 



 If you want to use one of the official checkpoints for a task, explore the
 [gligen](https://huggingface.co/gligen) 
 Hub organizations!
 




[StableDiffusionGLIGENPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/gligen#diffusers.StableDiffusionGLIGENPipeline) 
 was contributed by
 [Nikhil Gajendrakumar](https://github.com/nikhil-masterful) 
 and
 [StableDiffusionGLIGENTextImagePipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/gligen#diffusers.StableDiffusionGLIGENTextImagePipeline) 
 was contributed by
 [Nguyễn Công Tú Anh](https://github.com/tuanh123789) 
.
 


## StableDiffusionGLIGENPipeline




### 




 class
 

 diffusers.
 

 StableDiffusionGLIGENPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen.py#L102)



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
 
 : CLIPFeatureExtractor
 




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


 Pipeline for text-to-image generation using Stable Diffusion with Grounded-Language-to-Image Generation (GLIGEN).
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen.py#L550)



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
 




 gligen\_scheduled\_sampling\_beta
 
 : float = 0.3
 




 gligen\_phrases
 
 : typing.List[str] = None
 




 gligen\_boxes
 
 : typing.List[typing.List[float]] = None
 




 gligen\_inpaint\_image
 
 : typing.Optional[PIL.Image.Image] = None
 




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
* **gligen\_phrases** 
 (
 `List[str]` 
 ) —
The phrases to guide what to include in each of the regions defined by the corresponding
 `gligen_boxes` 
. There should only be one phrase per bounding box.
* **gligen\_boxes** 
 (
 `List[List[float]]` 
 ) —
The bounding boxes that identify rectangular regions of the image that are going to be filled with the
content described by the corresponding
 `gligen_phrases` 
. Each rectangular box is defined as a
 `List[float]` 
 of 4 elements
 `[xmin, ymin, xmax, ymax]` 
 where each value is between [0,1].
* **gligen\_inpaint\_image** 
 (
 `PIL.Image.Image` 
 ,
 *optional* 
 ) —
The input image, if provided, is inpainted with objects described by the
 `gligen_boxes` 
 and
 `gligen_phrases` 
. Otherwise, it is treated as a generation task on a blank input image.
* **gligen\_scheduled\_sampling\_beta** 
 (
 `float` 
 , defaults to 0.3) —
Scheduled Sampling factor from
 [GLIGEN: Open-Set Grounded Text-to-Image
Generation](https://arxiv.org/pdf/2301.07093.pdf) 
. Scheduled Sampling factor is only varied for
scheduled sampling during inference for improved quality and controllability.
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
 


 Examples:
 



```
>>> import torch
>>> from diffusers import StableDiffusionGLIGENPipeline
>>> from diffusers.utils import load_image

>>> # Insert objects described by text at the region defined by bounding boxes
>>> pipe = StableDiffusionGLIGENPipeline.from_pretrained(
...     "masterful/gligen-1-4-inpainting-text-box", variant="fp16", torch_dtype=torch.float16
... )
>>> pipe = pipe.to("cuda")

>>> input_image = load_image(
...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/livingroom\_modern.png"
... )
>>> prompt = "a birthday cake"
>>> boxes = [[0.2676, 0.6088, 0.4773, 0.7183]]
>>> phrases = ["a birthday cake"]

>>> images = pipe(
...     prompt=prompt,
...     gligen_phrases=phrases,
...     gligen_inpaint_image=input_image,
...     gligen_boxes=boxes,
...     gligen_scheduled_sampling_beta=1,
...     output_type="pil",
...     num_inference_steps=50,
... ).images

>>> images[0].save("./gligen-1-4-inpainting-text-box.jpg")

>>> # Generate an image described by the prompt and
>>> # insert objects described by text at the region defined by bounding boxes
>>> pipe = StableDiffusionGLIGENPipeline.from_pretrained(
...     "masterful/gligen-1-4-generation-text-box", variant="fp16", torch_dtype=torch.float16
... )
>>> pipe = pipe.to("cuda")

>>> prompt = "a waterfall and a modern high speed train running through the tunnel in a beautiful forest with fall foliage"
>>> boxes = [[0.1387, 0.2051, 0.4277, 0.7090], [0.4980, 0.4355, 0.8516, 0.7266]]
>>> phrases = ["a waterfall", "a modern high speed train running through the tunnel"]

>>> images = pipe(
...     prompt=prompt,
...     gligen_phrases=phrases,
...     gligen_boxes=boxes,
...     gligen_scheduled_sampling_beta=1,
...     output_type="pil",
...     num_inference_steps=50,
... ).images

>>> images[0].save("./gligen-1-4-generation-text-box.jpg")
```


#### 




 enable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen.py#L174)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 disable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen.py#L181)



 (
 

 )
 




 Disable sliced VAE decoding. If
 `enable_vae_slicing` 
 was previously enabled, this method will go back to
computing decoding in one step.
 




#### 




 enable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen.py#L188)



 (
 

 )
 




 Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
 




#### 




 disable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen.py#L196)



 (
 

 )
 




 Disable tiled VAE decoding. If
 `enable_vae_tiling` 
 was previously enabled, this method will go back to
computing decoding in one step.
 




#### 




 enable\_model\_cpu\_offload




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L1377)



 (
 


 gpu\_id
 
 : typing.Optional[int] = None
 




 device
 
 : typing.Union[torch.device, str] = 'cuda'
 



 )
 


 Parameters
 




* **gpu\_id** 
 (
 `int` 
 ,
 *optional* 
 ) —
The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
* **device** 
 (
 `torch.Device` 
 or
 `str` 
 ,
 *optional* 
 , defaults to “cuda”) —
The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
default to “cuda”.


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




 prepare\_latents




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen.py#L506)



 (
 


 batch\_size
 


 num\_channels\_latents
 


 height
 


 width
 


 dtype
 


 device
 


 generator
 


 latents
 
 = None
 



 )
 


#### 




 enable\_fuser




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen.py#L523)



 (
 


 enabled
 
 = True
 



 )
 


#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen.py#L237)



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
 


## StableDiffusionGLIGENTextImagePipeline




### 




 class
 

 diffusers.
 

 StableDiffusionGLIGENTextImagePipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L148)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : CLIPTextModel
 




 tokenizer
 
 : CLIPTokenizer
 




 processor
 
 : CLIPProcessor
 




 image\_encoder
 
 : CLIPVisionModelWithProjection
 




 image\_project
 
 : CLIPImageProjection
 




 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : KarrasDiffusionSchedulers
 




 safety\_checker
 
 : StableDiffusionSafetyChecker
 




 feature\_extractor
 
 : CLIPFeatureExtractor
 




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
* **processor** 
 (
 [CLIPProcessor](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPProcessor) 
 ) —
A
 `CLIPProcessor` 
 to procces reference image.
* **image\_encoder** 
 (
 [CLIPVisionModelWithProjection](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPVisionModelWithProjection) 
 ) —
Frozen image-encoder (
 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 
 ).
* **image\_project** 
 (
 `CLIPImageProjection` 
 ) —
A
 `CLIPImageProjection` 
 to project image embedding into phrases embedding space.
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


 Pipeline for text-to-image generation using Stable Diffusion with Grounded-Language-to-Image Generation (GLIGEN).
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L711)



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
 




 gligen\_scheduled\_sampling\_beta
 
 : float = 0.3
 




 gligen\_phrases
 
 : typing.List[str] = None
 




 gligen\_images
 
 : typing.List[PIL.Image.Image] = None
 




 input\_phrases\_mask
 
 : typing.Union[int, typing.List[int]] = None
 




 input\_images\_mask
 
 : typing.Union[int, typing.List[int]] = None
 




 gligen\_boxes
 
 : typing.List[typing.List[float]] = None
 




 gligen\_inpaint\_image
 
 : typing.Optional[PIL.Image.Image] = None
 




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
 




 gligen\_normalize\_constant
 
 : float = 28.7
 




 clip\_skip
 
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
* **gligen\_phrases** 
 (
 `List[str]` 
 ) —
The phrases to guide what to include in each of the regions defined by the corresponding
 `gligen_boxes` 
. There should only be one phrase per bounding box.
* **gligen\_images** 
 (
 `List[PIL.Image.Image]` 
 ) —
The images to guide what to include in each of the regions defined by the corresponding
 `gligen_boxes` 
.
There should only be one image per bounding box
* **input\_phrases\_mask** 
 (
 `int` 
 or
 `List[int]` 
 ) —
pre phrases mask input defined by the correspongding
 `input_phrases_mask`
* **input\_images\_mask** 
 (
 `int` 
 or
 `List[int]` 
 ) —
pre images mask input defined by the correspongding
 `input_images_mask`
* **gligen\_boxes** 
 (
 `List[List[float]]` 
 ) —
The bounding boxes that identify rectangular regions of the image that are going to be filled with the
content described by the corresponding
 `gligen_phrases` 
. Each rectangular box is defined as a
 `List[float]` 
 of 4 elements
 `[xmin, ymin, xmax, ymax]` 
 where each value is between [0,1].
* **gligen\_inpaint\_image** 
 (
 `PIL.Image.Image` 
 ,
 *optional* 
 ) —
The input image, if provided, is inpainted with objects described by the
 `gligen_boxes` 
 and
 `gligen_phrases` 
. Otherwise, it is treated as a generation task on a blank input image.
* **gligen\_scheduled\_sampling\_beta** 
 (
 `float` 
 , defaults to 0.3) —
Scheduled Sampling factor from
 [GLIGEN: Open-Set Grounded Text-to-Image
Generation](https://arxiv.org/pdf/2301.07093.pdf) 
. Scheduled Sampling factor is only varied for
scheduled sampling during inference for improved quality and controllability.
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
* **gligen\_normalize\_constant** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 28.7) —
The normalize value of the image embedding.
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
 


 Examples:
 



```
>>> import torch
>>> from diffusers import StableDiffusionGLIGENTextImagePipeline
>>> from diffusers.utils import load_image

>>> # Insert objects described by image at the region defined by bounding boxes
>>> pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
...     "anhnct/Gligen\_Inpainting\_Text\_Image", torch_dtype=torch.float16
... )
>>> pipe = pipe.to("cuda")

>>> input_image = load_image(
...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/livingroom\_modern.png"
... )
>>> prompt = "a backpack"
>>> boxes = [[0.2676, 0.4088, 0.4773, 0.7183]]
>>> phrases = None
>>> gligen_image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/backpack.jpeg"
... )

>>> images = pipe(
...     prompt=prompt,
...     gligen_phrases=phrases,
...     gligen_inpaint_image=input_image,
...     gligen_boxes=boxes,
...     gligen_images=[gligen_image],
...     gligen_scheduled_sampling_beta=1,
...     output_type="pil",
...     num_inference_steps=50,
... ).images

>>> images[0].save("./gligen-inpainting-text-image-box.jpg")

>>> # Generate an image described by the prompt and
>>> # insert objects described by text and image at the region defined by bounding boxes
>>> pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
...     "anhnct/Gligen\_Text\_Image", torch_dtype=torch.float16
... )
>>> pipe = pipe.to("cuda")

>>> prompt = "a flower sitting on the beach"
>>> boxes = [[0.0, 0.09, 0.53, 0.76]]
>>> phrases = ["flower"]
>>> gligen_image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/pexels-pixabay-60597.jpg"
... )

>>> images = pipe(
...     prompt=prompt,
...     gligen_phrases=phrases,
...     gligen_images=[gligen_image],
...     gligen_boxes=boxes,
...     gligen_scheduled_sampling_beta=1,
...     output_type="pil",
...     num_inference_steps=50,
... ).images

>>> images[0].save("./gligen-generation-text-image-box.jpg")

>>> # Generate an image described by the prompt and
>>> # transfer style described by image at the region defined by bounding boxes
>>> pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
...     "anhnct/Gligen\_Text\_Image", torch_dtype=torch.float16
... )
>>> pipe = pipe.to("cuda")

>>> prompt = "a dragon flying on the sky"
>>> boxes = [[0.4, 0.2, 1.0, 0.8], [0.0, 1.0, 0.0, 1.0]]  # Set `[0.0, 1.0, 0.0, 1.0]` for the style

>>> gligen_image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
... )

>>> gligen_placeholder = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
... )

>>> images = pipe(
...     prompt=prompt,
...     gligen_phrases=[
...         "dragon",
...         "placeholder",
...     ],  # Can use any text instead of `placeholder` token, because we will use mask here
...     gligen_images=[
...         gligen_placeholder,
...         gligen_image,
...     ],  # Can use any image in gligen\_placeholder, because we will use mask here
...     input_phrases_mask=[1, 0],  # Set 0 for the placeholder token
...     input_images_mask=[0, 1],  # Set 0 for the placeholder image
...     gligen_boxes=boxes,
...     gligen_scheduled_sampling_beta=1,
...     output_type="pil",
...     num_inference_steps=50,
... ).images

>>> images[0].save("./gligen-generation-text-image-box-style-transfer.jpg")
```


#### 




 enable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L232)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 disable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L239)



 (
 

 )
 




 Disable sliced VAE decoding. If
 `enable_vae_slicing` 
 was previously enabled, this method will go back to
computing decoding in one step.
 




#### 




 enable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L246)



 (
 

 )
 




 Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
 




#### 




 disable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L254)



 (
 

 )
 




 Disable tiled VAE decoding. If
 `enable_vae_tiling` 
 was previously enabled, this method will go back to
computing decoding in one step.
 




#### 




 enable\_model\_cpu\_offload




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L1377)



 (
 


 gpu\_id
 
 : typing.Optional[int] = None
 




 device
 
 : typing.Union[torch.device, str] = 'cuda'
 



 )
 


 Parameters
 




* **gpu\_id** 
 (
 `int` 
 ,
 *optional* 
 ) —
The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
* **device** 
 (
 `torch.Device` 
 or
 `str` 
 ,
 *optional* 
 , defaults to “cuda”) —
The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
default to “cuda”.


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




 prepare\_latents




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L530)



 (
 


 batch\_size
 


 num\_channels\_latents
 


 height
 


 width
 


 dtype
 


 device
 


 generator
 


 latents
 
 = None
 



 )
 


#### 




 enable\_fuser




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L547)



 (
 


 enabled
 
 = True
 



 )
 


#### 




 complete\_mask




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L584)



 (
 


 has\_mask
 


 max\_objs
 


 device
 




 )
 




 Based on the input mask corresponding value
 `0 or 1` 
 for each phrases and image, mask the features
corresponding to phrases and images.
 




#### 




 crop




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L564)



 (
 


 im
 


 new\_width
 


 new\_height
 




 )
 




 Crop the input image to the specified dimensions.
 




#### 




 draw\_inpaint\_mask\_from\_boxes




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L552)



 (
 


 boxes
 


 size
 




 )
 




 Create an inpainting mask based on given boxes. This function generates an inpainting mask using the provided
boxes to mark regions that need to be inpainted.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L262)



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




 get\_clip\_feature




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L600)



 (
 


 input
 


 normalize\_constant
 


 device
 


 is\_image
 
 = False
 



 )
 




 Get image and phrases embedding by using CLIP pretrain model. The image embedding is transformed into the
phrases embedding space through a projection.
 




#### 




 get\_cross\_attention\_kwargs\_with\_grounded




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L624)



 (
 


 hidden\_size
 


 gligen\_phrases
 


 gligen\_images
 


 gligen\_boxes
 


 input\_phrases\_mask
 


 input\_images\_mask
 


 repeat\_batch
 


 normalize\_constant
 


 max\_objs
 


 device
 




 )
 




 Prepare the cross-attention kwargs containing information about the grounded input (boxes, mask, image
embedding, phrases embedding).
 




#### 




 get\_cross\_attention\_kwargs\_without\_grounded




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L688)



 (
 


 hidden\_size
 


 repeat\_batch
 


 max\_objs
 


 device
 




 )
 




 Prepare the cross-attention kwargs without information about the grounded input (boxes, mask, image embedding,
phrases embedding) (All are zero tensor).
 




#### 




 target\_size\_center\_crop




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_gligen_text_image.py#L575)



 (
 


 im
 


 new\_hw
 




 )
 




 Crop and resize the image to the target size while keeping the center.
 


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