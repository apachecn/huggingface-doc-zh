# ControlNet with Stable Diffusion XL

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/controlnet_sdxl>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl>



 ControlNet was introduced in
 [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) 
 by Lvmin Zhang and Maneesh Agrawala.
 



 With a ControlNet model, you can provide an additional control image to condition and control Stable Diffusion generation. For example, if you provide a depth map, the ControlNet model generates an image that’ll preserve the spatial information from the depth map. It is a more flexible and accurate way to control the image generation process.
 



 The abstract from the paper is:
 



*We present a neural network structure, ControlNet, to control pretrained large diffusion models to support additional input conditions. The ControlNet learns task-specific conditions in an end-to-end way, and the learning is robust even when the training dataset is small (< 50k). Moreover, training a ControlNet is as fast as fine-tuning a diffusion model, and the model can be trained on a personal devices. Alternatively, if powerful computation clusters are available, the model can scale to large amounts (millions to billions) of data. We report that large diffusion models like Stable Diffusion can be augmented with ControlNets to enable conditional inputs like edge maps, segmentation maps, keypoints, etc. This may enrich the methods to control large diffusion models and further facilitate related applications.* 




 You can find additional smaller Stable Diffusion XL (SDXL) ControlNet checkpoints from the 🤗
 [Diffusers](https://huggingface.co/diffusers) 
 Hub organization, and browse
 [community-trained](https://huggingface.co/models?other=stable-diffusion-xl&other=controlnet) 
 checkpoints on the Hub.
 




 🧪 Many of the SDXL ControlNet checkpoints are experimental, and there is a lot of room for improvement. Feel free to open an
 [Issue](https://github.com/huggingface/diffusers/issues/new/choose) 
 and leave us feedback on how we can improve!
 




 If you don’t see a checkpoint you’re interested in, you can train your own SDXL ControlNet with our
 [training script](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/README_sdxl.md) 
.
 




 Make sure to check out the Schedulers
 [guide](../../using-diffusers/schedulers) 
 to learn how to explore the tradeoff between scheduler speed and quality, and see the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## StableDiffusionXLControlNetPipeline




### 




 class
 

 diffusers.
 

 StableDiffusionXLControlNetPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py#L99)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : CLIPTextModel
 




 text\_encoder\_2
 
 : CLIPTextModelWithProjection
 




 tokenizer
 
 : CLIPTokenizer
 




 tokenizer\_2
 
 : CLIPTokenizer
 




 unet
 
 : UNet2DConditionModel
 




 controlnet
 
 : typing.Union[diffusers.models.controlnet.ControlNetModel, typing.List[diffusers.models.controlnet.ControlNetModel], typing.Tuple[diffusers.models.controlnet.ControlNetModel], diffusers.pipelines.controlnet.multicontrolnet.MultiControlNetModel]
 




 scheduler
 
 : KarrasDiffusionSchedulers
 




 force\_zeros\_for\_empty\_prompt
 
 : bool = True
 




 add\_watermarker
 
 : typing.Optional[bool] = None
 



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
* **text\_encoder\_2** 
 (
 [CLIPTextModelWithProjection](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPTextModelWithProjection) 
 ) —
Second frozen text-encoder
(
 [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) 
 ).
* **tokenizer** 
 (
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPTokenizer) 
 ) —
A
 `CLIPTokenizer` 
 to tokenize text.
* **tokenizer\_2** 
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
* **controlnet** 
 (
 [ControlNetModel](/docs/diffusers/v0.23.0/en/api/models/controlnet#diffusers.ControlNetModel) 
 or
 `List[ControlNetModel]` 
 ) —
Provides additional conditioning to the
 `unet` 
 during the denoising process. If you set multiple
ControlNets as a list, the outputs from each ControlNet are added together to create one combined
additional conditioning.
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
* **force\_zeros\_for\_empty\_prompt** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `"True"` 
 ) —
Whether the negative prompt embeddings should always be set to 0. Also see the config of
 `stabilityai/stable-diffusion-xl-base-1-0` 
.
* **add\_watermarker** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Whether to use the
 [invisible\_watermark](https://github.com/ShieldMnt/invisible-watermark/) 
 library to
watermark output images. If not defined, it defaults to
 `True` 
 if the package is installed; otherwise no
watermarker is used.


 Pipeline for text-to-image generation using Stable Diffusion XL with ControlNet guidance.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



 The pipeline also inherits the following loading methods:
 


* [load\_textual\_inversion()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.load_textual_inversion) 
 for loading textual inversion embeddings
* [loaders.StableDiffusionXLLoraLoaderMixin.load\_lora\_weights()](/docs/diffusers/v0.23.0/en/api/loaders#diffusers.loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights) 
 for loading LoRA weights
* [loaders.FromSingleFileMixin.from\_single\_file()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.from_single_file) 
 for loading
 `.ckpt` 
 files



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py#L794)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 prompt\_2
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 image
 
 : typing.Union[PIL.Image.Image, numpy.ndarray, torch.FloatTensor, typing.List[PIL.Image.Image], typing.List[numpy.ndarray], typing.List[torch.FloatTensor]] = None
 




 height
 
 : typing.Optional[int] = None
 




 width
 
 : typing.Optional[int] = None
 




 num\_inference\_steps
 
 : int = 50
 




 guidance\_scale
 
 : float = 5.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 negative\_prompt\_2
 
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
 




 pooled\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_pooled\_prompt\_embeds
 
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
 




 controlnet\_conditioning\_scale
 
 : typing.Union[float, typing.List[float]] = 1.0
 




 guess\_mode
 
 : bool = False
 




 control\_guidance\_start
 
 : typing.Union[float, typing.List[float]] = 0.0
 




 control\_guidance\_end
 
 : typing.Union[float, typing.List[float]] = 1.0
 




 original\_size
 
 : typing.Tuple[int, int] = None
 




 crops\_coords\_top\_left
 
 : typing.Tuple[int, int] = (0, 0)
 




 target\_size
 
 : typing.Tuple[int, int] = None
 




 negative\_original\_size
 
 : typing.Union[typing.Tuple[int, int], NoneType] = None
 




 negative\_crops\_coords\_top\_left
 
 : typing.Tuple[int, int] = (0, 0)
 




 negative\_target\_size
 
 : typing.Union[typing.Tuple[int, int], NoneType] = None
 




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
* **prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to be sent to
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `prompt` 
 is
used in both text-encoders.
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
 ,
 `List[np.ndarray]` 
 , —
 `List[List[torch.FloatTensor]]` 
 ,
 `List[List[np.ndarray]]` 
 or
 `List[List[PIL.Image.Image]]` 
 ):
The ControlNet input condition to provide guidance to the
 `unet` 
 for generation. If the type is
specified as
 `torch.FloatTensor` 
 , it is passed to ControlNet as is.
 `PIL.Image.Image` 
 can also be
accepted as an image. The dimensions of the output image defaults to
 `image` 
 ’s dimensions. If height
and/or width are passed,
 `image` 
 is resized accordingly. If multiple ControlNets are specified in
 `init` 
 , images must be passed as a list such that each element of the list can be correctly batched for
input to a single ControlNet.
* **height** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `self.unet.config.sample_size * self.vae_scale_factor` 
 ) —
The height in pixels of the generated image. Anything below 512 pixels won’t work well for
 [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 
 and checkpoints that are not specifically fine-tuned on low resolutions.
* **width** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `self.unet.config.sample_size * self.vae_scale_factor` 
 ) —
The width in pixels of the generated image. Anything below 512 pixels won’t work well for
 [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 
 and checkpoints that are not specifically fine-tuned on low resolutions.
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
 , defaults to 5.0) —
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
* **negative\_prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to guide what to not include in image generation. This is sent to
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `negative_prompt` 
 is used in both text-encoders.
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
* **pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated pooled text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
not provided, pooled text embeddings are generated from
 `prompt` 
 input argument.
* **negative\_pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs (prompt
weighting). If not provided, pooled
 `negative_prompt_embeds` 
 are generated from
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
* **controlnet\_conditioning\_scale** 
 (
 `float` 
 or
 `List[float]` 
 ,
 *optional* 
 , defaults to 1.0) —
The outputs of the ControlNet are multiplied by
 `controlnet_conditioning_scale` 
 before they are added
to the residual in the original
 `unet` 
. If multiple ControlNets are specified in
 `init` 
 , you can set
the corresponding scale as a list.
* **guess\_mode** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
The ControlNet encoder tries to recognize the content of the input image even if you remove all
prompts. A
 `guidance_scale` 
 value between 3.0 and 5.0 is recommended.
* **control\_guidance\_start** 
 (
 `float` 
 or
 `List[float]` 
 ,
 *optional* 
 , defaults to 0.0) —
The percentage of total steps at which the ControlNet starts applying.
* **control\_guidance\_end** 
 (
 `float` 
 or
 `List[float]` 
 ,
 *optional* 
 , defaults to 1.0) —
The percentage of total steps at which the ControlNet stops applying.
* **original\_size** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (1024, 1024)) —
If
 `original_size` 
 is not the same as
 `target_size` 
 the image will appear to be down- or upsampled.
 `original_size` 
 defaults to
 `(height, width)` 
 if not specified. Part of SDXL’s micro-conditioning as
explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
.
* **crops\_coords\_top\_left** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (0, 0)) —
 `crops_coords_top_left` 
 can be used to generate an image that appears to be “cropped” from the position
 `crops_coords_top_left` 
 downwards. Favorable, well-centered images are usually achieved by setting
 `crops_coords_top_left` 
 to (0, 0). Part of SDXL’s micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
.
* **target\_size** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (1024, 1024)) —
For most cases,
 `target_size` 
 should be set to the desired height and width of the generated image. If
not specified it will default to
 `(height, width)` 
. Part of SDXL’s micro-conditioning as explained in
section 2.2 of
 <https://huggingface.co/papers/2307.01952>
.
* **negative\_original\_size** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (1024, 1024)) —
To negatively condition the generation process based on a specific image resolution. Part of SDXL’s
micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
. For more
information, refer to this issue thread:
 <https://github.com/huggingface/diffusers/issues/4208>
.
* **negative\_crops\_coords\_top\_left** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (0, 0)) —
To negatively condition the generation process based on a specific crop coordinates. Part of SDXL’s
micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
. For more
information, refer to this issue thread:
 <https://github.com/huggingface/diffusers/issues/4208>
.
* **negative\_target\_size** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (1024, 1024)) —
To negatively condition the generation process based on a target image resolution. It should be as same
as the
 `target_size` 
 for most cases. Part of SDXL’s micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
. For more
information, refer to this issue thread:
 <https://github.com/huggingface/diffusers/issues/4208>
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
 is returned containing the output images.
 



 The call function to the pipeline for generation.
 


 Examples:
 



```
>>> # !pip install opencv-python transformers accelerate
>>> from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
>>> from diffusers.utils import load_image
>>> import numpy as np
>>> import torch

>>> import cv2
>>> from PIL import Image

>>> prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
>>> negative_prompt = "low quality, bad quality, sketches"

>>> # download an image
>>> image = load_image(
...     "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd\_controlnet/hf-logo.png"
... )

>>> # initialize the models and pipeline
>>> controlnet_conditioning_scale = 0.5  # recommended for good generalization
>>> controlnet = ControlNetModel.from_pretrained(
...     "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
... )
>>> vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
>>> pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
...     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
... )
>>> pipe.enable_model_cpu_offload()

>>> # get canny image
>>> image = np.array(image)
>>> image = cv2.Canny(image, 100, 200)
>>> image = image[:, :, None]
>>> image = np.concatenate([image, image, image], axis=2)
>>> canny_image = Image.fromarray(image)

>>> # generate image
>>> image = pipe(
...     prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image
... ).images[0]
```


#### 




 disable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py#L790)



 (
 

 )
 




 Disables the FreeU mechanism if enabled.
 




#### 




 disable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py#L197)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py#L214)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py#L767)



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
that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.
 




#### 




 enable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py#L189)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 enable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py#L205)



 (
 

 )
 




 Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py#L222)



 (
 


 prompt
 
 : str
 




 prompt\_2
 
 : typing.Optional[str] = None
 




 device
 
 : typing.Optional[torch.device] = None
 




 num\_images\_per\_prompt
 
 : int = 1
 




 do\_classifier\_free\_guidance
 
 : bool = True
 




 negative\_prompt
 
 : typing.Optional[str] = None
 




 negative\_prompt\_2
 
 : typing.Optional[str] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 pooled\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_pooled\_prompt\_embeds
 
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
* **prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to be sent to the
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `prompt` 
 is
used in both text-encoders
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
* **negative\_prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation to be sent to
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `negative_prompt` 
 is used in both text-encoders
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
* **pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated pooled text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting.
If not provided, pooled text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, pooled negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input argument.
* **lora\_scale** 
 (
 `float` 
 ,
 *optional* 
 ) —
A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
* **clip\_skip** 
 (
 `int` 
 ,
 *optional* 
 ) —
Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
the output of the pre-final layer will be used for computing the prompt embeddings.


 Encodes the prompt into text encoder hidden states.
 


## StableDiffusionXLControlNetImg2ImgPipeline




### 




 class
 

 diffusers.
 

 StableDiffusionXLControlNetImg2ImgPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py#L144)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : CLIPTextModel
 




 text\_encoder\_2
 
 : CLIPTextModelWithProjection
 




 tokenizer
 
 : CLIPTokenizer
 




 tokenizer\_2
 
 : CLIPTokenizer
 




 unet
 
 : UNet2DConditionModel
 




 controlnet
 
 : typing.Union[diffusers.models.controlnet.ControlNetModel, typing.List[diffusers.models.controlnet.ControlNetModel], typing.Tuple[diffusers.models.controlnet.ControlNetModel], diffusers.pipelines.controlnet.multicontrolnet.MultiControlNetModel]
 




 scheduler
 
 : KarrasDiffusionSchedulers
 




 requires\_aesthetics\_score
 
 : bool = False
 




 force\_zeros\_for\_empty\_prompt
 
 : bool = True
 




 add\_watermarker
 
 : typing.Optional[bool] = None
 



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
* **text\_encoder\_2** 
 (
 `CLIPTextModelWithProjection` 
 ) —
Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
 [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection) 
 ,
specifically the
 [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) 
 variant.
* **tokenizer** 
 (
 `CLIPTokenizer` 
 ) —
Tokenizer of class
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 
.
* **tokenizer\_2** 
 (
 `CLIPTokenizer` 
 ) —
Second Tokenizer of class
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 
.
* **unet** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) — Conditional U-Net architecture to denoise the encoded image latents.
* **controlnet** 
 (
 [ControlNetModel](/docs/diffusers/v0.23.0/en/api/models/controlnet#diffusers.ControlNetModel) 
 or
 `List[ControlNetModel]` 
 ) —
Provides additional conditioning to the unet during the denoising process. If you set multiple ControlNets
as a list, the outputs from each ControlNet are added together to create one combined additional
conditioning.
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
* **requires\_aesthetics\_score** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `"False"` 
 ) —
Whether the
 `unet` 
 requires an
 `aesthetic_score` 
 condition to be passed during inference. Also see the
config of
 `stabilityai/stable-diffusion-xl-refiner-1-0` 
.
* **force\_zeros\_for\_empty\_prompt** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `"True"` 
 ) —
Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
 `stabilityai/stable-diffusion-xl-base-1-0` 
.
* **add\_watermarker** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Whether to use the
 [invisible\_watermark library](https://github.com/ShieldMnt/invisible-watermark/) 
 to
watermark output images. If not defined, it will default to True if the package is installed, otherwise no
watermarker will be used.


 Pipeline for image-to-image generation using Stable Diffusion XL with ControlNet guidance.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



 In addition the pipeline inherits the following loading methods:
 


* *Textual-Inversion* 
 :
 [loaders.TextualInversionLoaderMixin.load\_textual\_inversion()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.load_textual_inversion)
* *LoRA* 
 :
 [loaders.StableDiffusionXLLoraLoaderMixin.load\_lora\_weights()](/docs/diffusers/v0.23.0/en/api/loaders#diffusers.loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights)



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py#L953)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 prompt\_2
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 image
 
 : typing.Union[PIL.Image.Image, numpy.ndarray, torch.FloatTensor, typing.List[PIL.Image.Image], typing.List[numpy.ndarray], typing.List[torch.FloatTensor]] = None
 




 control\_image
 
 : typing.Union[PIL.Image.Image, numpy.ndarray, torch.FloatTensor, typing.List[PIL.Image.Image], typing.List[numpy.ndarray], typing.List[torch.FloatTensor]] = None
 




 height
 
 : typing.Optional[int] = None
 




 width
 
 : typing.Optional[int] = None
 




 strength
 
 : float = 0.8
 




 num\_inference\_steps
 
 : int = 50
 




 guidance\_scale
 
 : float = 5.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 negative\_prompt\_2
 
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
 




 pooled\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_pooled\_prompt\_embeds
 
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
 




 controlnet\_conditioning\_scale
 
 : typing.Union[float, typing.List[float]] = 0.8
 




 guess\_mode
 
 : bool = False
 




 control\_guidance\_start
 
 : typing.Union[float, typing.List[float]] = 0.0
 




 control\_guidance\_end
 
 : typing.Union[float, typing.List[float]] = 1.0
 




 original\_size
 
 : typing.Tuple[int, int] = None
 




 crops\_coords\_top\_left
 
 : typing.Tuple[int, int] = (0, 0)
 




 target\_size
 
 : typing.Tuple[int, int] = None
 




 negative\_original\_size
 
 : typing.Union[typing.Tuple[int, int], NoneType] = None
 




 negative\_crops\_coords\_top\_left
 
 : typing.Tuple[int, int] = (0, 0)
 




 negative\_target\_size
 
 : typing.Union[typing.Tuple[int, int], NoneType] = None
 




 aesthetic\_score
 
 : float = 6.0
 




 negative\_aesthetic\_score
 
 : float = 2.5
 




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
* **prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to be sent to the
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `prompt` 
 is
used in both text-encoders
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
 ,
 `List[np.ndarray]` 
 , —
 `List[List[torch.FloatTensor]]` 
 ,
 `List[List[np.ndarray]]` 
 or
 `List[List[PIL.Image.Image]]` 
 ):
The initial image will be used as the starting point for the image generation process. Can also accept
image latents as
 `image` 
 , if passing latents directly, it will not be encoded again.
* **control\_image** 
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
 ,
 `List[np.ndarray]` 
 , —
 `List[List[torch.FloatTensor]]` 
 ,
 `List[List[np.ndarray]]` 
 or
 `List[List[PIL.Image.Image]]` 
 ):
The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
the type is specified as
 `Torch.FloatTensor` 
 , it is passed to ControlNet as is.
 `PIL.Image.Image` 
 can
also be accepted as an image. The dimensions of the output image defaults to
 `image` 
 ’s dimensions. If
height and/or width are passed,
 `image` 
 is resized according to them. If multiple ControlNets are
specified in init, images must be passed as a list such that each element of the list can be correctly
batched for input to a single controlnet.
* **height** 
 (
 `int` 
 ,
 *optional* 
 , defaults to the size of control\_image) —
The height in pixels of the generated image. Anything below 512 pixels won’t work well for
 [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 
 and checkpoints that are not specifically fine-tuned on low resolutions.
* **width** 
 (
 `int` 
 ,
 *optional* 
 , defaults to the size of control\_image) —
The width in pixels of the generated image. Anything below 512 pixels won’t work well for
 [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 
 and checkpoints that are not specifically fine-tuned on low resolutions.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 50) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **strength** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.3) —
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
.
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
* **negative\_prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation to be sent to
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `negative_prompt` 
 is used in both text-encoders
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
* **pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated pooled text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting.
If not provided, pooled text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, pooled negative\_prompt\_embeds will be generated from
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
* **controlnet\_conditioning\_scale** 
 (
 `float` 
 or
 `List[float]` 
 ,
 *optional* 
 , defaults to 1.0) —
The outputs of the controlnet are multiplied by
 `controlnet_conditioning_scale` 
 before they are added
to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
corresponding scale as a list.
* **guess\_mode** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
In this mode, the ControlNet encoder will try best to recognize the content of the input image even if
you remove all prompts. The
 `guidance_scale` 
 between 3.0 and 5.0 is recommended.
* **control\_guidance\_start** 
 (
 `float` 
 or
 `List[float]` 
 ,
 *optional* 
 , defaults to 0.0) —
The percentage of total steps at which the controlnet starts applying.
* **control\_guidance\_end** 
 (
 `float` 
 or
 `List[float]` 
 ,
 *optional* 
 , defaults to 1.0) —
The percentage of total steps at which the controlnet stops applying.
* **original\_size** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (1024, 1024)) —
If
 `original_size` 
 is not the same as
 `target_size` 
 the image will appear to be down- or upsampled.
 `original_size` 
 defaults to
 `(height, width)` 
 if not specified. Part of SDXL’s micro-conditioning as
explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
.
* **crops\_coords\_top\_left** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (0, 0)) —
 `crops_coords_top_left` 
 can be used to generate an image that appears to be “cropped” from the position
 `crops_coords_top_left` 
 downwards. Favorable, well-centered images are usually achieved by setting
 `crops_coords_top_left` 
 to (0, 0). Part of SDXL’s micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
.
* **target\_size** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (1024, 1024)) —
For most cases,
 `target_size` 
 should be set to the desired height and width of the generated image. If
not specified it will default to
 `(height, width)` 
. Part of SDXL’s micro-conditioning as explained in
section 2.2 of
 <https://huggingface.co/papers/2307.01952>
.
* **negative\_original\_size** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (1024, 1024)) —
To negatively condition the generation process based on a specific image resolution. Part of SDXL’s
micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
. For more
information, refer to this issue thread:
 <https://github.com/huggingface/diffusers/issues/4208>
.
* **negative\_crops\_coords\_top\_left** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (0, 0)) —
To negatively condition the generation process based on a specific crop coordinates. Part of SDXL’s
micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
. For more
information, refer to this issue thread:
 <https://github.com/huggingface/diffusers/issues/4208>
.
* **negative\_target\_size** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (1024, 1024)) —
To negatively condition the generation process based on a target image resolution. It should be as same
as the
 `target_size` 
 for most cases. Part of SDXL’s micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
. For more
information, refer to this issue thread:
 <https://github.com/huggingface/diffusers/issues/4208>
.
* **aesthetic\_score** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 6.0) —
Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
Part of SDXL’s micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
.
* **negative\_aesthetic\_score** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 2.5) —
Part of SDXL’s micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
. Can be used to
simulate an aesthetic score of the generated image by influencing the negative text condition.
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
 `tuple` 
 containing the output images.
 



 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> # pip install accelerate transformers safetensors diffusers

>>> import torch
>>> import numpy as np
>>> from PIL import Image

>>> from transformers import DPTFeatureExtractor, DPTForDepthEstimation
>>> from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
>>> from diffusers.utils import load_image


>>> depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
>>> feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
>>> controlnet = ControlNetModel.from_pretrained(
...     "diffusers/controlnet-depth-sdxl-1.0-small",
...     variant="fp16",
...     use_safetensors=True,
...     torch_dtype=torch.float16,
... ).to("cuda")
>>> vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
>>> pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
...     "stabilityai/stable-diffusion-xl-base-1.0",
...     controlnet=controlnet,
...     vae=vae,
...     variant="fp16",
...     use_safetensors=True,
...     torch_dtype=torch.float16,
... ).to("cuda")
>>> pipe.enable_model_cpu_offload()


>>> def get\_depth\_map(image):
...     image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
...     with torch.no_grad(), torch.autocast("cuda"):
...         depth_map = depth_estimator(image).predicted_depth

...     depth_map = torch.nn.functional.interpolate(
...         depth_map.unsqueeze(1),
...         size=(1024, 1024),
...         mode="bicubic",
...         align_corners=False,
...     )
...     depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
...     depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
...     depth_map = (depth_map - depth_min) /(depth_max - depth_min)
...     image = torch.cat([depth_map] * 3, dim=1)
...     image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
...     image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
...     return image


>>> prompt = "A robot, 4k photo"
>>> image = load_image(
...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
...     "/kandinsky/cat.png"
... ).resize((1024, 1024))
>>> controlnet_conditioning_scale = 0.5  # recommended for good generalization
>>> depth_image = get_depth_map(image)

>>> images = pipe(
...     prompt,
...     image=image,
...     control_image=depth_image,
...     strength=0.99,
...     num_inference_steps=50,
...     controlnet_conditioning_scale=controlnet_conditioning_scale,
... ).images
>>> images[0].save(f"robot\_cat.png")
```


#### 




 disable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py#L949)



 (
 

 )
 




 Disables the FreeU mechanism if enabled.
 




#### 




 disable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py#L251)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py#L268)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py#L926)



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
that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.
 




#### 




 enable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py#L243)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 enable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py#L259)



 (
 

 )
 




 Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py#L276)



 (
 


 prompt
 
 : str
 




 prompt\_2
 
 : typing.Optional[str] = None
 




 device
 
 : typing.Optional[torch.device] = None
 




 num\_images\_per\_prompt
 
 : int = 1
 




 do\_classifier\_free\_guidance
 
 : bool = True
 




 negative\_prompt
 
 : typing.Optional[str] = None
 




 negative\_prompt\_2
 
 : typing.Optional[str] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 pooled\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_pooled\_prompt\_embeds
 
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
* **prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to be sent to the
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `prompt` 
 is
used in both text-encoders
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
* **negative\_prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation to be sent to
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `negative_prompt` 
 is used in both text-encoders
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
* **pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated pooled text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting.
If not provided, pooled text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, pooled negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input argument.
* **lora\_scale** 
 (
 `float` 
 ,
 *optional* 
 ) —
A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
* **clip\_skip** 
 (
 `int` 
 ,
 *optional* 
 ) —
Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
the output of the pre-final layer will be used for computing the prompt embeddings.


 Encodes the prompt into text encoder hidden states.
 


## StableDiffusionXLControlNetInpaintPipeline




### 




 class
 

 diffusers.
 

 StableDiffusionXLControlNetInpaintPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint_sd_xl.py#L127)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : CLIPTextModel
 




 text\_encoder\_2
 
 : CLIPTextModelWithProjection
 




 tokenizer
 
 : CLIPTokenizer
 




 tokenizer\_2
 
 : CLIPTokenizer
 




 unet
 
 : UNet2DConditionModel
 




 controlnet
 
 : ControlNetModel
 




 scheduler
 
 : KarrasDiffusionSchedulers
 




 requires\_aesthetics\_score
 
 : bool = False
 




 force\_zeros\_for\_empty\_prompt
 
 : bool = True
 




 add\_watermarker
 
 : typing.Optional[bool] = None
 



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
Frozen text-encoder. Stable Diffusion XL uses the text portion of
 [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel) 
 , specifically
the
 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 
 variant.
* **text\_encoder\_2** 
 (
 `CLIPTextModelWithProjection` 
 ) —
Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
 [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection) 
 ,
specifically the
 [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) 
 variant.
* **tokenizer** 
 (
 `CLIPTokenizer` 
 ) —
Tokenizer of class
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 
.
* **tokenizer\_2** 
 (
 `CLIPTokenizer` 
 ) —
Second Tokenizer of class
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
 , or
 [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler) 
.


 Pipeline for text-to-image generation using Stable Diffusion XL.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



 In addition the pipeline inherits the following loading methods:
 


* *LoRA* 
 :
 [loaders.StableDiffusionXLLoraLoaderMixin.load\_lora\_weights()](/docs/diffusers/v0.23.0/en/api/loaders#diffusers.loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights)
* *Ckpt* 
 :
 [loaders.FromSingleFileMixin.from\_single\_file()](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.from_single_file)



 as well as the following saving methods:
 


* *LoRA* 
 :
 `loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights()`



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint_sd_xl.py#L1010)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 prompt\_2
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 image
 
 : typing.Union[PIL.Image.Image, numpy.ndarray, torch.FloatTensor, typing.List[PIL.Image.Image], typing.List[numpy.ndarray], typing.List[torch.FloatTensor]] = None
 




 mask\_image
 
 : typing.Union[PIL.Image.Image, numpy.ndarray, torch.FloatTensor, typing.List[PIL.Image.Image], typing.List[numpy.ndarray], typing.List[torch.FloatTensor]] = None
 




 control\_image
 
 : typing.Union[PIL.Image.Image, numpy.ndarray, torch.FloatTensor, typing.List[PIL.Image.Image], typing.List[numpy.ndarray], typing.List[torch.FloatTensor], typing.List[typing.Union[PIL.Image.Image, numpy.ndarray, torch.FloatTensor, typing.List[PIL.Image.Image], typing.List[numpy.ndarray], typing.List[torch.FloatTensor]]]] = None
 




 height
 
 : typing.Optional[int] = None
 




 width
 
 : typing.Optional[int] = None
 




 strength
 
 : float = 0.9999
 




 num\_inference\_steps
 
 : int = 50
 




 denoising\_start
 
 : typing.Optional[float] = None
 




 denoising\_end
 
 : typing.Optional[float] = None
 




 guidance\_scale
 
 : float = 5.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 negative\_prompt\_2
 
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
 




 pooled\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_pooled\_prompt\_embeds
 
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
 




 controlnet\_conditioning\_scale
 
 : typing.Union[float, typing.List[float]] = 1.0
 




 guess\_mode
 
 : bool = False
 




 control\_guidance\_start
 
 : typing.Union[float, typing.List[float]] = 0.0
 




 control\_guidance\_end
 
 : typing.Union[float, typing.List[float]] = 1.0
 




 guidance\_rescale
 
 : float = 0.0
 




 original\_size
 
 : typing.Tuple[int, int] = None
 




 crops\_coords\_top\_left
 
 : typing.Tuple[int, int] = (0, 0)
 




 target\_size
 
 : typing.Tuple[int, int] = None
 




 aesthetic\_score
 
 : float = 6.0
 




 negative\_aesthetic\_score
 
 : float = 2.5
 




 clip\_skip
 
 : typing.Optional[int] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput` 
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
* **prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to be sent to the
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `prompt` 
 is
used in both text-encoders
* **image** 
 (
 `PIL.Image.Image` 
 ) —
 `Image` 
 , or tensor representing an image batch which will be inpainted,
 *i.e.* 
 parts of the image will
be masked out with
 `mask_image` 
 and repainted according to
 `prompt` 
.
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
* **strength** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.9999) —
Conceptually, indicates how much to transform the masked portion of the reference
 `image` 
. Must be
between 0 and 1.
 `image` 
 will be used as a starting point, adding more noise to it the larger the
 `strength` 
. The number of denoising steps depends on the amount of noise initially added. When
 `strength` 
 is 1, added noise will be maximum and the denoising process will run for the full number of
iterations specified in
 `num_inference_steps` 
. A value of 1, therefore, essentially ignores the masked
portion of the reference
 `image` 
. Note that in the case of
 `denoising_start` 
 being declared as an
integer, the value of
 `strength` 
 will be ignored.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 50) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **denoising\_start** 
 (
 `float` 
 ,
 *optional* 
 ) —
When specified, indicates the fraction (between 0.0 and 1.0) of the total denoising process to be
bypassed before it is initiated. Consequently, the initial part of the denoising process is skipped and
it is assumed that the passed
 `image` 
 is a partly denoised image. Note that when this is specified,
strength will be ignored. The
 `denoising_start` 
 parameter is particularly beneficial when this pipeline
is integrated into a “Mixture of Denoisers” multi-pipeline setup, as detailed in
 [**Refining the Image
Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
.
* **denoising\_end** 
 (
 `float` 
 ,
 *optional* 
 ) —
When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
completed before it is intentionally prematurely terminated. As a result, the returned sample will
still retain a substantial amount of noise (ca. final 20% of timesteps still needed) and should be
denoised by a successor pipeline that has
 `denoising_start` 
 set to 0.8 so that it only denoises the
final 20% of the scheduler. The denoising\_end parameter should ideally be utilized when this pipeline
forms a part of a “Mixture of Denoisers” multi-pipeline setup, as elaborated in
 [**Refining the Image
Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
.
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
* **negative\_prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation to be sent to
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `negative_prompt` 
 is used in both text-encoders
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
* **pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated pooled text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting.
If not provided, pooled text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, pooled negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input argument.
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
* **original\_size** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (1024, 1024)) —
If
 `original_size` 
 is not the same as
 `target_size` 
 the image will appear to be down- or upsampled.
 `original_size` 
 defaults to
 `(width, height)` 
 if not specified. Part of SDXL’s micro-conditioning as
explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
.
* **crops\_coords\_top\_left** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (0, 0)) —
 `crops_coords_top_left` 
 can be used to generate an image that appears to be “cropped” from the position
 `crops_coords_top_left` 
 downwards. Favorable, well-centered images are usually achieved by setting
 `crops_coords_top_left` 
 to (0, 0). Part of SDXL’s micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
.
* **target\_size** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (1024, 1024)) —
For most cases,
 `target_size` 
 should be set to the desired height and width of the generated image. If
not specified it will default to
 `(width, height)` 
. Part of SDXL’s micro-conditioning as explained in
section 2.2 of
 <https://huggingface.co/papers/2307.01952>
.
* **aesthetic\_score** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 6.0) —
Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
Part of SDXL’s micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
.
* **negative\_aesthetic\_score** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 2.5) —
Part of SDXL’s micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
. Can be used to
simulate an aesthetic score of the generated image by influencing the negative text condition.
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
 

`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput` 
 if
 `return_dict` 
 is True, otherwise a
 `tuple.` 
 tuple. When returning a tuple, the first element is a list with the generated images.
 



 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> # !pip install transformers accelerate
>>> from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
>>> from diffusers.utils import load_image
>>> import numpy as np
>>> import torch

>>> init_image = load_image(
...     "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable\_diffusion\_inpaint/boy.png"
... )
>>> init_image = init_image.resize((1024, 1024))

>>> generator = torch.Generator(device="cpu").manual_seed(1)

>>> mask_image = load_image(
...     "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable\_diffusion\_inpaint/boy\_mask.png"
... )
>>> mask_image = mask_image.resize((1024, 1024))


>>> def make\_canny\_condition(image):
...     image = np.array(image)
...     image = cv2.Canny(image, 100, 200)
...     image = image[:, :, None]
...     image = np.concatenate([image, image, image], axis=2)
...     image = Image.fromarray(image)
...     return image


>>> control_image = make_canny_condition(init_image)

>>> controlnet = ControlNetModel.from_pretrained(
...     "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
... )
>>> pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
...     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
... )

>>> pipe.enable_model_cpu_offload()

>>> # generate image
>>> image = pipe(
...     "a handsome man with ray-ban sunglasses",
...     num_inference_steps=20,
...     generator=generator,
...     eta=1.0,
...     image=init_image,
...     mask_image=mask_image,
...     control_image=control_image,
... ).images[0]
```


#### 




 disable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint_sd_xl.py#L1006)



 (
 

 )
 




 Disables the FreeU mechanism if enabled.
 




#### 




 disable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint_sd_xl.py#L226)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint_sd_xl.py#L243)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint_sd_xl.py#L983)



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
that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.
 




#### 




 enable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint_sd_xl.py#L218)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 enable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint_sd_xl.py#L234)



 (
 

 )
 




 Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint_sd_xl.py#L251)



 (
 


 prompt
 
 : str
 




 prompt\_2
 
 : typing.Optional[str] = None
 




 device
 
 : typing.Optional[torch.device] = None
 




 num\_images\_per\_prompt
 
 : int = 1
 




 do\_classifier\_free\_guidance
 
 : bool = True
 




 negative\_prompt
 
 : typing.Optional[str] = None
 




 negative\_prompt\_2
 
 : typing.Optional[str] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 pooled\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_pooled\_prompt\_embeds
 
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
* **prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to be sent to the
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `prompt` 
 is
used in both text-encoders
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
* **negative\_prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation to be sent to
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `negative_prompt` 
 is used in both text-encoders
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
* **pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated pooled text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting.
If not provided, pooled text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, pooled negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input argument.
* **lora\_scale** 
 (
 `float` 
 ,
 *optional* 
 ) —
A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
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