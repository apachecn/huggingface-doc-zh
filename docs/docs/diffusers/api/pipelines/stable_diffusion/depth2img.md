# Depth-to-image

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/stable_diffusion/depth2img>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/depth2img>



 The Stable Diffusion model can also infer depth based on an image using
 [MiDas](https://github.com/isl-org/MiDaS) 
. This allows you to pass a text prompt and an initial image to condition the generation of new images as well as a
 `depth_map` 
 to preserve the image structure.
 




 Make sure to check out the Stable Diffusion
 [Tips](overview#tips) 
 section to learn how to explore the tradeoff between scheduler speed and quality, and how to reuse pipeline components efficiently!
 



 If you’re interested in using one of the official checkpoints for a task, explore the
 [CompVis](https://huggingface.co/CompVis) 
 ,
 [Runway](https://huggingface.co/runwayml) 
 , and
 [Stability AI](https://huggingface.co/stabilityai) 
 Hub organizations!
 


## StableDiffusionDepth2ImgPipeline




### 




 class
 

 diffusers.
 

 StableDiffusionDepth2ImgPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_depth2img.py#L73)



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
 




 depth\_estimator
 
 : DPTForDepthEstimation
 




 feature\_extractor
 
 : DPTFeatureExtractor
 



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


 Pipeline for text-guided depth-based image-to-image generation using Stable Diffusion.
 



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



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_depth2img.py#L594)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 image
 
 : typing.Union[PIL.Image.Image, numpy.ndarray, torch.FloatTensor, typing.List[PIL.Image.Image], typing.List[numpy.ndarray], typing.List[torch.FloatTensor]] = None
 




 depth\_map
 
 : typing.Optional[torch.FloatTensor] = None
 




 strength
 
 : float = 0.8
 




 num\_inference\_steps
 
 : typing.Optional[int] = 50
 




 guidance\_scale
 
 : typing.Optional[float] = 7.5
 




 negative\_prompt
 
 : typing.Union[typing.List[str], str, NoneType] = None
 




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
 
 : typing.Optional[int] = None
 




 callback\_on\_step\_end
 
 : typing.Union[typing.Callable[[int, int, typing.Dict], NoneType], NoneType] = None
 




 callback\_on\_step\_end\_tensor\_inputs
 
 : typing.List[str] = ['latents']
 




 \*\*kwargs
 




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
 or tensor representing an image batch to be used as the starting point. Can accept image
latents as
 `image` 
 only if
 `depth_map` 
 is not
 `None` 
.
* **depth\_map** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Depth prediction to be used as additional conditioning for the image generation process. If not
defined, it automatically predicts the depth with
 `self.depth_estimator` 
.
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
 [StableDiffusionPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) 
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
 is returned where the first element is a list with the generated images.
 



 The call function to the pipeline for generation.
 


 Examples:
 



```
>>> import torch
>>> import requests
>>> from PIL import Image

>>> from diffusers import StableDiffusionDepth2ImgPipeline

>>> pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
...     "stabilityai/stable-diffusion-2-depth",
...     torch_dtype=torch.float16,
... )
>>> pipe.to("cuda")


>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> init_image = Image.open(requests.get(url, stream=True).raw)
>>> prompt = "two tigers"
>>> n_propmt = "bad, deformed, ugly, bad anotomy"
>>> image = pipe(prompt=prompt, image=init_image, negative_prompt=n_propmt, strength=0.7).images[0]
```


#### 




 enable\_attention\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L2027)



 (
 


 slice\_size
 
 : typing.Union[str, int, NoneType] = 'auto'
 



 )
 


 Parameters
 




* **slice\_size** 
 (
 `str` 
 or
 `int` 
 ,
 *optional* 
 , defaults to
 `"auto"` 
 ) —
When
 `"auto"` 
 , halves the input to the attention heads, so attention will be computed in two steps. If
 `"max"` 
 , maximum amount of memory will be saved by running only one slice at a time. If a number is
provided, uses as many slices as
 `attention_head_dim //slice_size` 
. In this case,
 `attention_head_dim` 
 must be a multiple of
 `slice_size` 
.


 Enable sliced attention computation. When this option is enabled, the attention module splits the input tensor
in slices to compute attention in several steps. For more than one attention head, the computation is performed
sequentially over each head. This is useful to save some memory in exchange for a small speed decrease.
 




 ⚠️ Don’t enable attention slicing if you’re already using
 `scaled_dot_product_attention` 
 (SDPA) from PyTorch
2.0 or xFormers. These attention computations are already very memory efficient so you won’t need to enable
this function. If you enable attention slicing with SDPA or xFormers, it can lead to serious slow downs!
 



 Examples:
 



```
>>> import torch
>>> from diffusers import StableDiffusionPipeline

>>> pipe = StableDiffusionPipeline.from_pretrained(
...     "runwayml/stable-diffusion-v1-5",
...     torch_dtype=torch.float16,
...     use_safetensors=True,
... )

>>> prompt = "a photo of an astronaut riding a horse on mars"
>>> pipe.enable_attention_slicing()
>>> image = pipe(prompt).images[0]
```


#### 




 disable\_attention\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L2067)



 (
 

 )
 




 Disable sliced attention computation. If
 `enable_attention_slicing` 
 was previously called, attention is
computed in one step.
 




#### 




 enable\_xformers\_memory\_efficient\_attention




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L1966)



 (
 


 attention\_op
 
 : typing.Optional[typing.Callable] = None
 



 )
 


 Parameters
 




* **attention\_op** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
Override the default
 `None` 
 operator for use as
 `op` 
 argument to the
 [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
 function of xFormers.


 Enable memory efficient attention from
 [xFormers](https://facebookresearch.github.io/xformers/) 
. When this
option is enabled, you should observe lower GPU memory usage and a potential speed up during inference. Speed
up during training is not guaranteed.
 




 ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient attention takes
precedent.
 



 Examples:
 



```
>>> import torch
>>> from diffusers import DiffusionPipeline
>>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

>>> pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
>>> pipe = pipe.to("cuda")
>>> pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
>>> # Workaround for not accepting attention shape using VAE for Flash Attention
>>> pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
```


#### 




 disable\_xformers\_memory\_efficient\_attention




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L2001)



 (
 

 )
 




 Disable memory efficient attention from
 [xFormers](https://facebookresearch.github.io/xformers/) 
.
 




#### 




 load\_textual\_inversion




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L984)



 (
 


 pretrained\_model\_name\_or\_path
 
 : typing.Union[str, typing.List[str], typing.Dict[str, torch.Tensor], typing.List[typing.Dict[str, torch.Tensor]]]
 




 token
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 tokenizer
 
 : typing.Optional[ForwardRef('PreTrainedTokenizer')] = None
 




 text\_encoder
 
 : typing.Optional[ForwardRef('PreTrainedModel')] = None
 




 \*\*kwargs
 




 )
 


 Parameters
 




* **pretrained\_model\_name\_or\_path** 
 (
 `str` 
 or
 `os.PathLike` 
 or
 `List[str or os.PathLike]` 
 or
 `Dict` 
 or
 `List[Dict]` 
 ) —
Can be either one of the following or a list of them:
 
	+ A string, the
	 *model id* 
	 (for example
	 `sd-concepts-library/low-poly-hd-logos-icons` 
	 ) of a
	pretrained model hosted on the Hub.
	+ A path to a
	 *directory* 
	 (for example
	 `./my_text_inversion_directory/` 
	 ) containing the textual
	inversion weights.
	+ A path to a
	 *file* 
	 (for example
	 `./my_text_inversions.pt` 
	 ) containing textual inversion weights.
	+ A
	 [torch state
	dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict) 
	.
* **token** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
Override the token to use for the textual inversion weights. If
 `pretrained_model_name_or_path` 
 is a
list, then
 `token` 
 must also be a list of equal length.
* **text\_encoder** 
 (
 [CLIPTextModel](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPTextModel) 
 ,
 *optional* 
 ) —
Frozen text-encoder (
 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 
 ).
If not specified, function will take self.tokenizer.
* **tokenizer** 
 (
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPTokenizer) 
 ,
 *optional* 
 ) —
A
 `CLIPTokenizer` 
 to tokenize text. If not specified, function will take self.tokenizer.
* **weight\_name** 
 (
 `str` 
 ,
 *optional* 
 ) —
Name of a custom weight file. This should be used when:
 
	+ The saved textual inversion file is in 🤗 Diffusers format, but was saved under a specific weight
	name such as
	 `text_inv.bin` 
	.
	+ The saved textual inversion file is in the Automatic1111 format.
* **cache\_dir** 
 (
 `Union[str, os.PathLike]` 
 ,
 *optional* 
 ) —
Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
is not used.
* **force\_download** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether or not to force the (re-)download of the model weights and configuration files, overriding the
cached versions if they exist.
* **resume\_download** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether or not to resume downloading the model weights and configuration files. If set to
 `False` 
 , any
incompletely downloaded files are deleted.
* **proxies** 
 (
 `Dict[str, str]` 
 ,
 *optional* 
 ) —
A dictionary of proxy servers to use by protocol or endpoint, for example,
 `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}` 
. The proxies are used on each request.
* **local\_files\_only** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether to only load local model weights and configuration files or not. If set to
 `True` 
 , the model
won’t be downloaded from the Hub.
* **use\_auth\_token** 
 (
 `str` 
 or
 *bool* 
 ,
 *optional* 
 ) —
The token to use as HTTP bearer authorization for remote files. If
 `True` 
 , the token generated from
 `diffusers-cli login` 
 (stored in
 `~/.huggingface` 
 ) is used.
* **revision** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"main"` 
 ) —
The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
allowed by Git.
* **subfolder** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `""` 
 ) —
The subfolder location of a model file within a larger model repository on the Hub or locally.
* **mirror** 
 (
 `str` 
 ,
 *optional* 
 ) —
Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
information.


 Load textual inversion embeddings into the text encoder of
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline) 
 (both 🤗 Diffusers and
Automatic1111 formats are supported).
 



 Example:
 


 To load a textual inversion embedding vector in 🤗 Diffusers format:
 



```
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

pipe.load_textual_inversion("sd-concepts-library/cat-toy")

prompt = "A <cat-toy> backpack"

image = pipe(prompt, num_inference_steps=50).images[0]
image.save("cat-backpack.png")
```




 To load a textual inversion embedding vector in Automatic1111 format, make sure to download the vector first
(for example from
 [civitAI](https://civitai.com/models/3036?modelVersionId=9857) 
 ) and then load the vector
 


 locally:
 



```
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

pipe.load_textual_inversion("./charturnerv2.pt", token="charturnerv2")

prompt = "charturnerv2, multiple views of the same character in the same outfit, a character turnaround of a woman wearing a black jacket and red shirt, best quality, intricate details."

image = pipe(prompt, num_inference_steps=50).images[0]
image.save("character.png")
```


#### 




 load\_lora\_weights




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L1173)



 (
 


 pretrained\_model\_name\_or\_path\_or\_dict
 
 : typing.Union[str, typing.Dict[str, torch.Tensor]]
 




 adapter\_name
 
 = None
 




 \*\*kwargs
 




 )
 


 Parameters
 




* **pretrained\_model\_name\_or\_path\_or\_dict** 
 (
 `str` 
 or
 `os.PathLike` 
 or
 `dict` 
 ) —
See
 [lora\_state\_dict()](/docs/diffusers/v0.23.0/en/api/loaders#diffusers.loaders.LoraLoaderMixin.lora_state_dict) 
.
* **kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
See
 [lora\_state\_dict()](/docs/diffusers/v0.23.0/en/api/loaders#diffusers.loaders.LoraLoaderMixin.lora_state_dict) 
.
* **adapter\_name** 
 (
 `str` 
 ,
 *optional* 
 ) —
Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
 `default_{i}` 
 where i is the total number of adapters being loaded.


 Load LoRA weights specified in
 `pretrained_model_name_or_path_or_dict` 
 into
 `self.unet` 
 and
 `self.text_encoder` 
.
 



 All kwargs are forwarded to
 `self.lora_state_dict` 
.
 



 See
 [lora\_state\_dict()](/docs/diffusers/v0.23.0/en/api/loaders#diffusers.loaders.LoraLoaderMixin.lora_state_dict) 
 for more details on how the state dict is loaded.
 



 See
 [load\_lora\_into\_unet()](/docs/diffusers/v0.23.0/en/api/loaders#diffusers.loaders.LoraLoaderMixin.load_lora_into_unet) 
 for more details on how the state dict is loaded into
 `self.unet` 
.
 



 See
 [load\_lora\_into\_text\_encoder()](/docs/diffusers/v0.23.0/en/api/loaders#diffusers.loaders.LoraLoaderMixin.load_lora_into_text_encoder) 
 for more details on how the state dict is loaded
into
 `self.text_encoder` 
.
 




#### 




 save\_lora\_weights




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L1969)



 (
 


 save\_directory
 
 : typing.Union[str, os.PathLike]
 




 unet\_lora\_layers
 
 : typing.Dict[str, typing.Union[torch.nn.modules.module.Module, torch.Tensor]] = None
 




 text\_encoder\_lora\_layers
 
 : typing.Dict[str, torch.nn.modules.module.Module] = None
 




 is\_main\_process
 
 : bool = True
 




 weight\_name
 
 : str = None
 




 save\_function
 
 : typing.Callable = None
 




 safe\_serialization
 
 : bool = True
 



 )
 


 Parameters
 




* **save\_directory** 
 (
 `str` 
 or
 `os.PathLike` 
 ) —
Directory to save LoRA parameters to. Will be created if it doesn’t exist.
* **unet\_lora\_layers** 
 (
 `Dict[str, torch.nn.Module]` 
 or
 `Dict[str, torch.Tensor]` 
 ) —
State dict of the LoRA layers corresponding to the
 `unet` 
.
* **text\_encoder\_lora\_layers** 
 (
 `Dict[str, torch.nn.Module]` 
 or
 `Dict[str, torch.Tensor]` 
 ) —
State dict of the LoRA layers corresponding to the
 `text_encoder` 
. Must explicitly pass the text
encoder LoRA state dict because it comes from 🤗 Transformers.
* **is\_main\_process** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether the process calling this is the main process or not. Useful during distributed training and you
need to call this function on all processes. In this case, set
 `is_main_process=True` 
 only on the main
process to avoid race conditions.
* **save\_function** 
 (
 `Callable` 
 ) —
The function to use to save the state dictionary. Useful during distributed training when you need to
replace
 `torch.save` 
 with another method. Can be configured with the environment variable
 `DIFFUSERS_SAVE_MODE` 
.
* **safe\_serialization** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether to save the model using
 `safetensors` 
 or the traditional PyTorch way with
 `pickle` 
.


 Save the LoRA parameters corresponding to the UNet and text encoder.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_depth2img.py#L180)



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