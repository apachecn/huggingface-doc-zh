# Pipelines

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/overview>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/overview>



 Pipelines provide a simple way to run state-of-the-art diffusion models in inference by bundling all of the necessary components (multiple independently-trained models, schedulers, and processors) into a single end-to-end class. Pipelines are flexible and they can be adapted to use different schedulers or even model components.
 



 All pipelines are built from the base
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
 class which provides basic functionality for loading, downloading, and saving all the components. Specific pipeline types (for example
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline) 
 ) loaded with
 [from\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained) 
 are automatically detected and the pipeline components are loaded and passed to the
 `__init__` 
 function of the pipeline.
 




 You shouldn’t use the
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
 class for training. Individual components (for example,
 [UNet2DModel](/docs/diffusers/v0.23.0/en/api/models/unet2d#diffusers.UNet2DModel) 
 and
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) of diffusion pipelines are usually trained individually, so we suggest directly working with them instead.
 


 Pipelines do not offer any training functionality. You’ll notice PyTorch’s autograd is disabled by decorating the
 `__call__()` 
 method with a
 [`torch.no_grad`](https://pytorch.org/docs/stable/generated/torch.no_grad.html)
 decorator because pipelines should not be used for training. If you’re interested in training, please take a look at the
 [Training](../../training/overview) 
 guides instead!
 




 The table below lists all the pipelines currently available in 🤗 Diffusers and the tasks they support. Click on a pipeline to view its abstract and published paper.
 


| 	 Pipeline	  | 	 Tasks	  |
| --- | --- |
| [AltDiffusion](alt_diffusion)  | 	 image2image	  |
| [Attend-and-Excite](attend_and_excite)  | 	 text2image	  |
| [Audio Diffusion](audio_diffusion)  | 	 image2audio	  |
| [AudioLDM](audioldm)  | 	 text2audio	  |
| [AudioLDM2](audioldm2)  | 	 text2audio	  |
| [BLIP Diffusion](blip_diffusion)  | 	 text2image	  |
| [Consistency Models](consistency_models)  | 	 unconditional image generation	  |
| [ControlNet](controlnet)  | 	 text2image, image2image, inpainting	  |
| [ControlNet with Stable Diffusion XL](controlnet_sdxl)  | 	 text2image	  |
| [Cycle Diffusion](cycle_diffusion)  | 	 image2image	  |
| [Dance Diffusion](dance_diffusion)  | 	 unconditional audio generation	  |
| [DDIM](ddim)  | 	 unconditional image generation	  |
| [DDPM](ddpm)  | 	 unconditional image generation	  |
| [DeepFloyd IF](deepfloyd_if)  | 	 text2image, image2image, inpainting, super-resolution	  |
| [DiffEdit](diffedit)  | 	 inpainting	  |
| [DiT](dit)  | 	 text2image	  |
| [GLIGEN](gligen)  | 	 text2image	  |
| [InstructPix2Pix](pix2pix)  | 	 image editing	  |
| [Kandinsky](kandinsky)  | 	 text2image, image2image, inpainting, interpolation	  |
| [Kandinsky 2.2](kandinsky_v22)  | 	 text2image, image2image, inpainting	  |
| [Latent Diffusion](latent_diffusion)  | 	 text2image, super-resolution	  |
| [LDM3D](ldm3d_diffusion)  | 	 text2image, text-to-3D	  |
| [MultiDiffusion](panorama)  | 	 text2image	  |
| [MusicLDM](musicldm)  | 	 text2audio	  |
| [PaintByExample](paint_by_example)  | 	 inpainting	  |
| [ParaDiGMS](paradigms)  | 	 text2image	  |
| [Pix2Pix Zero](pix2pix_zero)  | 	 image editing	  |
| [PNDM](pndm)  | 	 unconditional image generation	  |
| [RePaint](repaint)  | 	 inpainting	  |
| [ScoreSdeVe](score_sde_ve)  | 	 unconditional image generation	  |
| [Self-Attention Guidance](self_attention_guidance)  | 	 text2image	  |
| [Semantic Guidance](semantic_stable_diffusion)  | 	 text2image	  |
| [Shap-E](shap_e)  | 	 text-to-3D, image-to-3D	  |
| [Spectrogram Diffusion](spectrogram_diffusion)  |  |
| [Stable Diffusion](stable_diffusion/overview)  | 	 text2image, image2image, depth2image, inpainting, image variation, latent upscaler, super-resolution	  |
| [Stable Diffusion Model Editing](model_editing)  | 	 model editing	  |
| [Stable Diffusion XL](stable_diffusion_xl)  | 	 text2image, image2image, inpainting	  |
| [Stable unCLIP](stable_unclip)  | 	 text2image, image variation	  |
| [KarrasVe](karras_ve)  | 	 unconditional image generation	  |
| [T2I Adapter](adapter)  | 	 text2image	  |
| [Text2Video](text_to_video)  | 	 text2video, video2video	  |
| [Text2Video Zero](text_to_video_zero)  | 	 text2video	  |
| [UnCLIP](unclip)  | 	 text2image, image variation	  |
| [Unconditional Latent Diffusion](latent_diffusion_uncond)  | 	 unconditional image generation	  |
| [UniDiffuser](unidiffuser)  | 	 text2image, image2text, image variation, text variation, unconditional image generation, unconditional audio generation	  |
| [Value-guided planning](value_guided_sampling)  | 	 value guided sampling	  |
| [Versatile Diffusion](versatile_diffusion)  | 	 text2image, image variation	  |
| [VQ Diffusion](vq_diffusion)  | 	 text2image	  |
| [Wuerstchen](wuerstchen)  | 	 text2image	  |


## DiffusionPipeline




### 




 class
 

 diffusers.
 

 DiffusionPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L514)



 (
 

 )
 




 Base class for all pipelines.
 



[DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
 stores all components (models, schedulers, and processors) for diffusion pipelines and
provides methods for loading, downloading and saving models. It also includes methods to:
 


* move all PyTorch modules to the device of your choice
* enable/disable the progress bar for the denoising iteration



 Class attributes:
 


* **config\_name** 
 (
 `str` 
 ) — The configuration filename that stores the class and module names of all the
diffusion pipeline’s components.
* **\_optional\_components** 
 (
 `List[str]` 
 ) — List of all optional components that don’t have to be passed to the
pipeline to function (should be overridden by subclasses).



#### 




 \_\_call\_\_




 (
 


 \*args
 


 \*\*kwargs
 




 )
 




 Call self as a function.
 




#### 




 device




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L869)



 (
 

 )
 

 →
 



 export const metadata = 'undefined';
 

`torch.device` 



 Returns
 




 export const metadata = 'undefined';
 

`torch.device` 




 export const metadata = 'undefined';
 




 The torch device on which the pipeline is located.
 




#### 




 to




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L710)



 (
 


 \*args
 


 \*\*kwargs
 




 )
 

 →
 



 export const metadata = 'undefined';
 

[DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 


 Parameters
 




* **dtype** 
 (
 `torch.dtype` 
 ,
 *optional* 
 ) —
Returns a pipeline with the specified
 [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)
* **device** 
 (
 `torch.Device` 
 ,
 *optional* 
 ) —
Returns a pipeline with the specified
 [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
* **silence\_dtype\_warnings** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether to omit warnings if the target
 `dtype` 
 is not compatible with the target
 `device` 
.




 Returns
 




 export const metadata = 'undefined';
 

[DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 




 export const metadata = 'undefined';
 




 The pipeline converted to specified
 `dtype` 
 and/or
 `dtype` 
.
 



 Performs Pipeline dtype and/or device conversion. A torch.dtype and torch.device are inferred from the
arguments of
 `self.to(*args, **kwargs).` 


 If the pipeline already has the correct torch.dtype and torch.device, then it is returned as is. Otherwise,
the returned pipeline is a copy of self with the desired torch.dtype and torch.device.
 




 Here are the ways to call
 `to` 
 :
 


* `to(dtype, silence_dtype_warnings=False) → DiffusionPipeline` 
 to return a pipeline with the specified
 [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)
* `to(device, silence_dtype_warnings=False) → DiffusionPipeline` 
 to return a pipeline with the specified
 [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
* `to(device=None, dtype=None, silence_dtype_warnings=False) → DiffusionPipeline` 
 to return a pipeline with the
specified
 [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
 and
 [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)




#### 




 components




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L1905)



 (
 

 )
 




 The
 `self.components` 
 property can be useful to run different pipelines with the same weights and
configurations without reallocating additional memory.
 



 Returns (
 `dict` 
 ):
A dictionary containing all the modules needed to initialize the pipeline.
 


 Examples:
 



```
>>> from diffusers import (
...     StableDiffusionPipeline,
...     StableDiffusionImg2ImgPipeline,
...     StableDiffusionInpaintPipeline,
... )

>>> text2img = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
>>> img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
>>> inpaint = StableDiffusionInpaintPipeline(**text2img.components)
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




 download




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L1519)



 (
 


 pretrained\_model\_name
 


 \*\*kwargs
 




 )
 

 →
 



 export const metadata = 'undefined';
 

`os.PathLike` 


 Parameters
 




* **pretrained\_model\_name** 
 (
 `str` 
 or
 `os.PathLike` 
 ,
 *optional* 
 ) —
A string, the
 *repository id* 
 (for example
 `CompVis/ldm-text2im-large-256` 
 ) of a pretrained pipeline
hosted on the Hub.
* **custom\_pipeline** 
 (
 `str` 
 ,
 *optional* 
 ) —
Can be either:
 
	+ A string, the
	 *repository id* 
	 (for example
	 `CompVis/ldm-text2im-large-256` 
	 ) of a pretrained
	pipeline hosted on the Hub. The repository must contain a file called
	 `pipeline.py` 
	 that defines
	the custom pipeline.
	+ A string, the
	 *file name* 
	 of a community pipeline hosted on GitHub under
	 [Community](https://github.com/huggingface/diffusers/tree/main/examples/community) 
	. Valid file
	names must match the file name and not the pipeline script (
	 `clip_guided_stable_diffusion` 
	 instead of
	 `clip_guided_stable_diffusion.py` 
	 ). Community pipelines are always loaded from the
	current
	 `main` 
	 branch of GitHub.
	+ A path to a
	 *directory* 
	 (
	 `./my_pipeline_directory/` 
	 ) containing a custom pipeline. The directory
	must contain a file called
	 `pipeline.py` 
	 that defines the custom pipeline.


 🧪 This is an experimental feature and may change in the future.
 




 For more information on how to load and create custom pipelines, take a look at
 [How to contribute a
community pipeline](https://huggingface.co/docs/diffusers/main/en/using-diffusers/contribute_pipeline) 
.
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
* **output\_loading\_info(
 `bool` 
 ,** 
*optional* 
 , defaults to
 `False` 
 ) —
Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
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
* **custom\_revision** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"main"` 
 ) —
The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
 `revision` 
 when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
custom pipeline from GitHub, otherwise it defaults to
 `"main"` 
 when loading from the Hub.
* **mirror** 
 (
 `str` 
 ,
 *optional* 
 ) —
Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
information.
* **variant** 
 (
 `str` 
 ,
 *optional* 
 ) —
Load weights from a specified variant filename such as
 `"fp16"` 
 or
 `"ema"` 
. This is ignored when
loading
 `from_flax` 
.
* **use\_safetensors** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
If set to
 `None` 
 , the safetensors weights are downloaded if they’re available
 **and** 
 if the
safetensors library is installed. If set to
 `True` 
 , the model is forcibly loaded from safetensors
weights. If set to
 `False` 
 , safetensors weights are not loaded.
* **use\_onnx** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
If set to
 `True` 
 , ONNX weights will always be downloaded if present. If set to
 `False` 
 , ONNX weights
will never be downloaded. By default
 `use_onnx` 
 defaults to the
 `_is_onnx` 
 class attribute which is
 `False` 
 for non-ONNX pipelines and
 `True` 
 for ONNX pipelines. ONNX weights include both files ending
with
 `.onnx` 
 and
 `.pb` 
.
* **trust\_remote\_code** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether or not to allow for custom pipelines and components defined on the Hub in their own files. This
option should only be set to
 `True` 
 for repositories you trust and in which you have read the code, as
it will execute code present on the Hub on your local machine.




 Returns
 




 export const metadata = 'undefined';
 

`os.PathLike` 




 export const metadata = 'undefined';
 




 A path to the downloaded pipeline.
 



 Download and cache a PyTorch diffusion pipeline from pretrained pipeline weights.
 




 To use private or
 [gated models](https://huggingface.co/docs/hub/models-gated#gated-models) 
 , log-in with
 `huggingface-cli login` 
.
 


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




 enable\_sequential\_cpu\_offload




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L1466)



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


 Offloads all models to CPU using 🤗 Accelerate, significantly reducing memory usage. When called, the state
dicts of all
 `torch.nn.Module` 
 components (except those in
 `self._exclude_from_cpu_offload` 
 ) are saved to CPU
and then moved to
 `torch.device('meta')` 
 and loaded to GPU only when their specific submodule has its
 `forward` 
 method called. Offloading happens on a submodule basis. Memory savings are higher than with
 `enable_model_cpu_offload` 
 , but performance is lower.
 




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




 from\_pretrained




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L899)



 (
 


 pretrained\_model\_name\_or\_path
 
 : typing.Union[str, os.PathLike, NoneType]
 




 \*\*kwargs
 




 )
 


 Parameters
 




* **pretrained\_model\_name\_or\_path** 
 (
 `str` 
 or
 `os.PathLike` 
 ,
 *optional* 
 ) —
Can be either:
 
	+ A string, the
	 *repo id* 
	 (for example
	 `CompVis/ldm-text2im-large-256` 
	 ) of a pretrained pipeline
	hosted on the Hub.
	+ A path to a
	 *directory* 
	 (for example
	 `./my_pipeline_directory/` 
	 ) containing pipeline weights
	saved using
	 [save\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.save_pretrained) 
	.
* **torch\_dtype** 
 (
 `str` 
 or
 `torch.dtype` 
 ,
 *optional* 
 ) —
Override the default
 `torch.dtype` 
 and load the model with another dtype. If “auto” is passed, the
dtype is automatically derived from the model’s weights.
* **custom\_pipeline** 
 (
 `str` 
 ,
 *optional* 
 ) —
 

 🧪 This is an experimental feature and may change in the future.
 




 Can be either:
 



	+ A string, the
	 *repo id* 
	 (for example
	 `hf-internal-testing/diffusers-dummy-pipeline` 
	 ) of a custom
	pipeline hosted on the Hub. The repository must contain a file called pipeline.py that defines
	the custom pipeline.
	+ A string, the
	 *file name* 
	 of a community pipeline hosted on GitHub under
	 [Community](https://github.com/huggingface/diffusers/tree/main/examples/community) 
	. Valid file
	names must match the file name and not the pipeline script (
	 `clip_guided_stable_diffusion` 
	 instead of
	 `clip_guided_stable_diffusion.py` 
	 ). Community pipelines are always loaded from the
	current main branch of GitHub.
	+ A path to a directory (
	 `./my_pipeline_directory/` 
	 ) containing a custom pipeline. The directory
	must contain a file called
	 `pipeline.py` 
	 that defines the custom pipeline.

 For more information on how to load and create custom pipelines, please have a look at
 [Loading and
Adding Custom
Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)
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
* **cache\_dir** 
 (
 `Union[str, os.PathLike]` 
 ,
 *optional* 
 ) —
Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
is not used.
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
* **output\_loading\_info(
 `bool` 
 ,** 
*optional* 
 , defaults to
 `False` 
 ) —
Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
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
* **custom\_revision** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"main"` 
 ) —
The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
 `revision` 
 when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
custom pipeline from GitHub, otherwise it defaults to
 `"main"` 
 when loading from the Hub.
* **mirror** 
 (
 `str` 
 ,
 *optional* 
 ) —
Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
information.
* **device\_map** 
 (
 `str` 
 or
 `Dict[str, Union[int, str, torch.device]]` 
 ,
 *optional* 
 ) —
A map that specifies where each submodule should go. It doesn’t need to be defined for each
parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
same device.
 
 Set
 `device_map="auto"` 
 to have 🤗 Accelerate automatically compute the most optimized
 `device_map` 
. For
more information about each option see
 [designing a device
map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map) 
.
* **max\_memory** 
 (
 `Dict` 
 ,
 *optional* 
 ) —
A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
each GPU and the available CPU RAM if unset.
* **offload\_folder** 
 (
 `str` 
 or
 `os.PathLike` 
 ,
 *optional* 
 ) —
The path to offload weights if device\_map contains the value
 `"disk"` 
.
* **offload\_state\_dict** 
 (
 `bool` 
 ,
 *optional* 
 ) —
If
 `True` 
 , temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to
 `True` 
 when there is some disk offload.
* **low\_cpu\_mem\_usage** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 if torch version >= 1.9.0 else
 `False` 
 ) —
Speed up model loading only loading the pretrained weights and not initializing the weights. This also
tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
argument to
 `True` 
 will raise an error.
* **use\_safetensors** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
If set to
 `None` 
 , the safetensors weights are downloaded if they’re available
 **and** 
 if the
safetensors library is installed. If set to
 `True` 
 , the model is forcibly loaded from safetensors
weights. If set to
 `False` 
 , safetensors weights are not loaded.
* **use\_onnx** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
If set to
 `True` 
 , ONNX weights will always be downloaded if present. If set to
 `False` 
 , ONNX weights
will never be downloaded. By default
 `use_onnx` 
 defaults to the
 `_is_onnx` 
 class attribute which is
 `False` 
 for non-ONNX pipelines and
 `True` 
 for ONNX pipelines. ONNX weights include both files ending
with
 `.onnx` 
 and
 `.pb` 
.
* **kwargs** 
 (remaining dictionary of keyword arguments,
 *optional* 
 ) —
Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
class). The overwritten components are passed directly to the pipelines
 `__init__` 
 method. See example
below for more information.
* **variant** 
 (
 `str` 
 ,
 *optional* 
 ) —
Load weights from a specified variant filename such as
 `"fp16"` 
 or
 `"ema"` 
. This is ignored when
loading
 `from_flax` 
.


 Instantiate a PyTorch diffusion pipeline from pretrained pipeline weights.
 



 The pipeline is set in evaluation mode (
 `model.eval()` 
 ) by default.
 


 If you get the error message below, you need to finetune the weights for your downstream task:
 



```
Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
- conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```


 To use private or
 [gated](https://huggingface.co/docs/hub/models-gated#gated-models) 
 models, log-in with
 `huggingface-cli login` 
.
 



 Examples:
 



```
>>> from diffusers import DiffusionPipeline

>>> # Download pipeline from huggingface.co and cache.
>>> pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

>>> # Download pipeline that requires an authorization token
>>> # For more information on access tokens, please refer to this section
>>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
>>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

>>> # Use a different scheduler
>>> from diffusers import LMSDiscreteScheduler

>>> scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
>>> pipeline.scheduler = scheduler
```


#### 




 maybe\_free\_model\_hooks




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L1447)



 (
 

 )
 




 Function that offloads all components, removes all model hooks that were added when using
 `enable_model_cpu_offload` 
 and then applies them again. In case the model has not been offloaded this function
is a no-op. Make sure to add this function to the end of the
 `__call__` 
 function of your pipeline so that it
functions correctly when applying enable\_model\_cpu\_offload.
 




#### 




 numpy\_to\_pil




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L1941)



 (
 


 images
 




 )
 




 Convert a NumPy image or a batch of images to a PIL image.
 




#### 




 save\_pretrained




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L597)



 (
 


 save\_directory
 
 : typing.Union[str, os.PathLike]
 




 safe\_serialization
 
 : bool = True
 




 variant
 
 : typing.Optional[str] = None
 




 push\_to\_hub
 
 : bool = False
 




 \*\*kwargs
 




 )
 


 Parameters
 




* **save\_directory** 
 (
 `str` 
 or
 `os.PathLike` 
 ) —
Directory to save a pipeline to. Will be created if it doesn’t exist.
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
* **variant** 
 (
 `str` 
 ,
 *optional* 
 ) —
If specified, weights are saved in the format
 `pytorch_model.<variant>.bin` 
.
* **push\_to\_hub** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
repository you want to push to with
 `repo_id` 
 (will default to the name of
 `save_directory` 
 in your
namespace).
* **kwargs** 
 (
 `Dict[str, Any]` 
 ,
 *optional* 
 ) —
Additional keyword arguments passed along to the
 [push\_to\_hub()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub) 
 method.


 Save all saveable variables of the pipeline to a directory. A pipeline variable can be saved and loaded if its
class implements both a save and loading method. The pipeline is easily reloaded using the
 [from\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained) 
 class method.
 


## FlaxDiffusionPipeline




### 




 class
 

 diffusers.
 

 FlaxDiffusionPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_flax_utils.py#L101)



 (
 

 )
 




 Base class for Flax-based pipelines.
 



[FlaxDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.FlaxDiffusionPipeline) 
 stores all components (models, schedulers, and processors) for diffusion pipelines and
provides methods for loading, downloading and saving models. It also includes methods to:
 


* enable/disable the progress bar for the denoising iteration



 Class attributes:
 


* **config\_name** 
 (
 `str` 
 ) — The configuration filename that stores the class and module names of all the
diffusion pipeline’s components.



#### 




 from\_pretrained




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_flax_utils.py#L228)



 (
 


 pretrained\_model\_name\_or\_path
 
 : typing.Union[str, os.PathLike, NoneType]
 




 \*\*kwargs
 




 )
 


 Parameters
 




* **pretrained\_model\_name\_or\_path** 
 (
 `str` 
 or
 `os.PathLike` 
 ,
 *optional* 
 ) —
Can be either:
 
	+ A string, the
	 *repo id* 
	 (for example
	 `runwayml/stable-diffusion-v1-5` 
	 ) of a pretrained pipeline
	hosted on the Hub.
	+ A path to a
	 *directory* 
	 (for example
	 `./my_model_directory` 
	 ) containing the model weights saved
	using
	 [save\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.FlaxDiffusionPipeline.save_pretrained) 
	.
* **dtype** 
 (
 `str` 
 or
 `jnp.dtype` 
 ,
 *optional* 
 ) —
Override the default
 `jnp.dtype` 
 and load the model under this dtype. If
 `"auto"` 
 , the dtype is
automatically derived from the model’s weights.
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
* **output\_loading\_info(
 `bool` 
 ,** 
*optional* 
 , defaults to
 `False` 
 ) —
Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
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
* **mirror** 
 (
 `str` 
 ,
 *optional* 
 ) —
Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
information.
* **kwargs** 
 (remaining dictionary of keyword arguments,
 *optional* 
 ) —
Can be used to overwrite load and saveable variables (the pipeline components) of the specific pipeline
class. The overwritten components are passed directly to the pipelines
 `__init__` 
 method.


 Instantiate a Flax-based diffusion pipeline from pretrained pipeline weights.
 



 The pipeline is set in evaluation mode (`model.eval()) by default and dropout modules are deactivated.
 


 If you get the error message below, you need to finetune the weights for your downstream task:
 



```
Some weights of FlaxUNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
```


 To use private or
 [gated models](https://huggingface.co/docs/hub/models-gated#gated-models) 
 , log-in with
 `huggingface-cli login` 
.
 



 Examples:
 



```
>>> from diffusers import FlaxDiffusionPipeline

>>> # Download pipeline from huggingface.co and cache.
>>> # Requires to be logged in to Hugging Face hub,
>>> # see more in the documentation
>>> pipeline, params = FlaxDiffusionPipeline.from_pretrained(
...     "runwayml/stable-diffusion-v1-5",
...     revision="bf16",
...     dtype=jnp.bfloat16,
... )

>>> # Download pipeline, but use a different scheduler
>>> from diffusers import FlaxDPMSolverMultistepScheduler

>>> model_id = "runwayml/stable-diffusion-v1-5"
>>> dpmpp, dpmpp_state = FlaxDPMSolverMultistepScheduler.from_pretrained(
...     model_id,
...     subfolder="scheduler",
... )

>>> dpm_pipe, dpm_params = FlaxStableDiffusionPipeline.from_pretrained(
...     model_id, revision="bf16", dtype=jnp.bfloat16, scheduler=dpmpp
... )
>>> dpm_params["scheduler"] = dpmpp_state
```


#### 




 numpy\_to\_pil




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_flax_utils.py#L585)



 (
 


 images
 




 )
 




 Convert a NumPy image or a batch of images to a PIL image.
 




#### 




 save\_pretrained




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_flax_utils.py#L150)



 (
 


 save\_directory
 
 : typing.Union[str, os.PathLike]
 




 params
 
 : typing.Union[typing.Dict, flax.core.frozen\_dict.FrozenDict]
 




 push\_to\_hub
 
 : bool = False
 




 \*\*kwargs
 




 )
 


 Parameters
 




* **save\_directory** 
 (
 `str` 
 or
 `os.PathLike` 
 ) —
Directory to which to save. Will be created if it doesn’t exist.
* **push\_to\_hub** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
repository you want to push to with
 `repo_id` 
 (will default to the name of
 `save_directory` 
 in your
namespace).
* **kwargs** 
 (
 `Dict[str, Any]` 
 ,
 *optional* 
 ) —
Additional keyword arguments passed along to the
 [push\_to\_hub()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub) 
 method.


 Save all saveable variables of the pipeline to a directory. A pipeline variable can be saved and loaded if its
class implements both a save and loading method. The pipeline is easily reloaded using the
 [from\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.FlaxDiffusionPipeline.from_pretrained) 
 class method.
 


## PushToHubMixin




### 




 class
 

 diffusers.utils.
 

 PushToHubMixin




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/utils/hub_utils.py#L373)



 (
 

 )
 




 A Mixin to push a model, scheduler, or pipeline to the Hugging Face Hub.
 



#### 




 push\_to\_hub




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/utils/hub_utils.py#L402)



 (
 


 repo\_id
 
 : str
 




 commit\_message
 
 : typing.Optional[str] = None
 




 private
 
 : typing.Optional[bool] = None
 




 token
 
 : typing.Optional[str] = None
 




 create\_pr
 
 : bool = False
 




 safe\_serialization
 
 : bool = True
 




 variant
 
 : typing.Optional[str] = None
 



 )
 


 Parameters
 




* **repo\_id** 
 (
 `str` 
 ) —
The name of the repository you want to push your model, scheduler, or pipeline files to. It should
contain your organization name when pushing to an organization.
 `repo_id` 
 can also be a path to a local
directory.
* **commit\_message** 
 (
 `str` 
 ,
 *optional* 
 ) —
Message to commit while pushing. Default to
 `"Upload {object}"` 
.
* **private** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Whether or not the repository created should be private.
* **token** 
 (
 `str` 
 ,
 *optional* 
 ) —
The token to use as HTTP bearer authorization for remote files. The token generated when running
 `huggingface-cli login` 
 (stored in
 `~/.huggingface` 
 ).
* **create\_pr** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether or not to create a PR with the uploaded files or directly commit.
* **safe\_serialization** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to convert the model weights to the
 `safetensors` 
 format.
* **variant** 
 (
 `str` 
 ,
 *optional* 
 ) —
If specified, weights are saved in the format
 `pytorch_model.<variant>.bin` 
.


 Upload model, scheduler, or pipeline files to the 🤗 Hugging Face Hub.
 


 Examples:
 



```
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="unet")

# Push the `unet` to your namespace with the name "my-finetuned-unet".
unet.push_to_hub("my-finetuned-unet")

# Push the `unet` to an organization with the name "my-finetuned-unet".
unet.push_to_hub("your-org/my-finetuned-unet")
```