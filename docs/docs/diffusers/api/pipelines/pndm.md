# PNDM

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/pndm>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/pndm>



[Pseudo Numerical methods for Diffusion Models on manifolds](https://huggingface.co/papers/2202.09778) 
 (PNDM) is by Luping Liu, Yi Ren, Zhijie Lin and Zhou Zhao.
 



 The abstract from the paper is:
 



*Denoising Diffusion Probabilistic Models (DDPMs) can generate high-quality samples such as image and audio samples. However, DDPMs require hundreds to thousands of iterations to produce final samples. Several prior works have successfully accelerated DDPMs through adjusting the variance schedule (e.g., Improved Denoising Diffusion Probabilistic Models) or the denoising equation (e.g., Denoising Diffusion Implicit Models (DDIMs)). However, these acceleration methods cannot maintain the quality of samples and even introduce new noise at a high speedup rate, which limit their practicability. To accelerate the inference process while keeping the sample quality, we provide a fresh perspective that DDPMs should be treated as solving differential equations on manifolds. Under such a perspective, we propose pseudo numerical methods for diffusion models (PNDMs). Specifically, we figure out how to solve differential equations on manifolds and show that DDIMs are simple cases of pseudo numerical methods. We change several classical numerical methods to corresponding pseudo numerical methods and find that the pseudo linear multi-step method is the best in most situations. According to our experiments, by directly using pre-trained models on Cifar10, CelebA and LSUN, PNDMs can generate higher quality synthetic images with only 50 steps compared with 1000-step DDIMs (20x speedup), significantly outperform DDIMs with 250 steps (by around 0.4 in FID) and have good generalization on different variance schedules.* 




 The original codebase can be found at
 [luping-liu/PNDM](https://github.com/luping-liu/PNDM) 
.
 




 Make sure to check out the Schedulers
 [guide](../../using-diffusers/schedulers) 
 to learn how to explore the tradeoff between scheduler speed and quality, and see the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## PNDMPipeline




### 




 class
 

 diffusers.
 

 PNDMPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pndm/pipeline_pndm.py#L26)



 (
 


 unet
 
 : UNet2DModel
 




 scheduler
 
 : PNDMScheduler
 



 )
 


 Parameters
 




* **unet** 
 (
 [UNet2DModel](/docs/diffusers/v0.23.0/en/api/models/unet2d#diffusers.UNet2DModel) 
 ) —
A
 `UNet2DModel` 
 to denoise the encoded image latents.
* **scheduler** 
 (
 [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler) 
 ) —
A
 `PNDMScheduler` 
 to be used in combination with
 `unet` 
 to denoise the encoded image.


 Pipeline for unconditional image generation.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pndm/pipeline_pndm.py#L50)



 (
 


 batch\_size
 
 : int = 1
 




 num\_inference\_steps
 
 : int = 50
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 




 \*\*kwargs
 




 )
 

 →
 



 export const metadata = 'undefined';
 

[ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 or
 `tuple` 


 Parameters
 




* **batch\_size** 
 (
 `int` 
 ,
 `optional` 
 , defaults to 1) —
The number of images to generate.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 `optional` 
 , defaults to 50) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **generator** 
 (
 `torch.Generator` 
 ,
 `optional` 
 ) —
A
 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)
 to make
generation deterministic.
* **output\_type** 
 (
 `str` 
 ,
 `optional` 
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
returned where the first element is a list with the generated images.
 



 The call function to the pipeline for generation.
 


 Example:
 



```
>>> from diffusers import PNDMPipeline

>>> # load model and scheduler
>>> pndm = PNDMPipeline.from_pretrained("google/ddpm-cifar10-32")

>>> # run pipeline in inference (sample random noise and denoise)
>>> image = pndm().images[0]

>>> # save image
>>> image.save("pndm\_generated\_image.png")
```


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