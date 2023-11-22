# DiT

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/dit>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/dit>



[Scalable Diffusion Models with Transformers](https://huggingface.co/papers/2212.09748) 
 (DiT) is by William Peebles and Saining Xie.
 



 The abstract from the paper is:
 



*We explore a new class of diffusion models based on the transformer architecture. We train latent diffusion models of images, replacing the commonly-used U-Net backbone with a transformer that operates on latent patches. We analyze the scalability of our Diffusion Transformers (DiTs) through the lens of forward pass complexity as measured by Gflops. We find that DiTs with higher Gflops — through increased transformer depth/width or increased number of input tokens — consistently have lower FID. In addition to possessing good scalability properties, our largest DiT-XL/2 models outperform all prior diffusion models on the class-conditional ImageNet 512x512 and 256x256 benchmarks, achieving a state-of-the-art FID of 2.27 on the latter.* 




 The original codebase can be found at
 [facebookresearch/dit](https://github.com/facebookresearch/dit) 
.
 




 Make sure to check out the Schedulers
 [guide](../../using-diffusers/schedulers) 
 to learn how to explore the tradeoff between scheduler speed and quality, and see the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## DiTPipeline




### 




 class
 

 diffusers.
 

 DiTPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/dit/pipeline_dit.py#L31)



 (
 


 transformer
 
 : Transformer2DModel
 




 vae
 
 : AutoencoderKL
 




 scheduler
 
 : KarrasDiffusionSchedulers
 




 id2label
 
 : typing.Union[typing.Dict[int, str], NoneType] = None
 



 )
 


 Parameters
 




* **transformer** 
 (
 [Transformer2DModel](/docs/diffusers/v0.23.0/en/api/models/transformer2d#diffusers.Transformer2DModel) 
 ) —
A class conditioned
 `Transformer2DModel` 
 to denoise the encoded image latents.
* **vae** 
 (
 [AutoencoderKL](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL) 
 ) —
Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
* **scheduler** 
 (
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 ) —
A scheduler to be used in combination with
 `transformer` 
 to denoise the encoded image latents.


 Pipeline for image generation based on a Transformer backbone instead of a UNet.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/dit/pipeline_dit.py#L91)



 (
 


 class\_labels
 
 : typing.List[int]
 




 guidance\_scale
 
 : float = 4.0
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 num\_inference\_steps
 
 : int = 50
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 or
 `tuple` 


 Parameters
 




* **class\_labels** 
 (List[int]) —
List of ImageNet class labels for the images to be generated.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
A higher guidance scale value encourages the model to generate images closely linked to the text
 `prompt` 
 at the expense of lower image quality. Guidance scale is enabled when
 `guidance_scale > 1` 
.
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
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 250) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
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
 [ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 instead of a plain tuple.




 Returns
 




 export const metadata = 'undefined';
 

[ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is
 `True` 
 ,
 [ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 is returned, otherwise a
 `tuple` 
 is
returned where the first element is a list with the generated images
 



 The call function to the pipeline for generation.
 


 Examples:
 



```
>>> from diffusers import DiTPipeline, DPMSolverMultistepScheduler
>>> import torch

>>> pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
>>> pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
>>> pipe = pipe.to("cuda")

>>> # pick words from Imagenet class labels
>>> pipe.labels  # to print all available words

>>> # pick words that exist in ImageNet
>>> words = ["white shark", "umbrella"]

>>> class_ids = pipe.get_label_ids(words)

>>> generator = torch.manual_seed(33)
>>> output = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator)

>>> image = output.images[0]  # label 'white shark'
```


#### 




 get\_label\_ids




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/dit/pipeline_dit.py#L66)



 (
 


 label
 
 : typing.Union[str, typing.List[str]]
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`list` 
 of
 `int` 


 Parameters
 




* **label** 
 (
 `str` 
 or
 `dict` 
 of
 `str` 
 ) —
Label strings to be mapped to class ids.




 Returns
 




 export const metadata = 'undefined';
 

`list` 
 of
 `int` 




 export const metadata = 'undefined';
 




 Class ids to be processed by pipeline.
 



 Map label strings from ImageNet to corresponding class ids.
 


## ImagePipelineOutput




### 




 class
 

 diffusers.
 

 ImagePipelineOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L110)



 (
 


 images
 
 : typing.Union[typing.List[PIL.Image.Image], numpy.ndarray]
 



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


 Output class for image pipelines.