# RePaint

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/repaint>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/repaint>



[RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://huggingface.co/papers/2201.09865) 
 is by Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, Luc Van Gool.
 



 The abstract from the paper is:
 



*Free-form inpainting is the task of adding new content to an image in the regions specified by an arbitrary binary mask. Most existing approaches train for a certain distribution of masks, which limits their generalization capabilities to unseen mask types. Furthermore, training with pixel-wise and perceptual losses often leads to simple textural extensions towards the missing areas instead of semantically meaningful generation. In this work, we propose RePaint: A Denoising Diffusion Probabilistic Model (DDPM) based inpainting approach that is applicable to even extreme masks. We employ a pretrained unconditional DDPM as the generative prior. To condition the generation process, we only alter the reverse diffusion iterations by sampling the unmasked regions using the given image information. Since this technique does not modify or condition the original DDPM network itself, the model produces high-quality and diverse output images for any inpainting form. We validate our method for both faces and general-purpose image inpainting using standard and extreme masks.
RePaint outperforms state-of-the-art Autoregressive, and GAN approaches for at least five out of six mask distributions.* 




 The original codebase can be found at
 [andreas128/RePaint](https://github.com/andreas128/RePaint) 
.
 




 Make sure to check out the Schedulers
 [guide](../../using-diffusers/schedulers) 
 to learn how to explore the tradeoff between scheduler speed and quality, and see the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## RePaintPipeline




### 




 class
 

 diffusers.
 

 RePaintPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/repaint/pipeline_repaint.py#L76)



 (
 


 unet
 


 scheduler
 




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
 [RePaintScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/repaint#diffusers.RePaintScheduler) 
 ) —
A
 `RePaintScheduler` 
 to be used in combination with
 `unet` 
 to denoise the encoded image.


 Pipeline for image inpainting using RePaint.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/repaint/pipeline_repaint.py#L98)



 (
 


 image
 
 : typing.Union[torch.Tensor, PIL.Image.Image]
 




 mask\_image
 
 : typing.Union[torch.Tensor, PIL.Image.Image]
 




 num\_inference\_steps
 
 : int = 250
 




 eta
 
 : float = 0.0
 




 jump\_length
 
 : int = 10
 




 jump\_n\_sample
 
 : int = 10
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




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
 




* **image** 
 (
 `torch.FloatTensor` 
 or
 `PIL.Image.Image` 
 ) —
The original image to inpaint on.
* **mask\_image** 
 (
 `torch.FloatTensor` 
 or
 `PIL.Image.Image` 
 ) —
The mask\_image where 0.0 define which part of the original image to inpaint.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1000) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **eta** 
 (
 `float` 
 ) —
The weight of the added noise in a diffusion step. Its value is between 0.0 and 1.0; 0.0 corresponds to
DDIM and 1.0 is the DDPM scheduler.
* **jump\_length** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 10) —
The number of steps taken forward in time before going backward in time for a single jump (“j” in
RePaint paper). Take a look at Figure 9 and 10 in the
 [paper](https://arxiv.org/pdf/2201.09865.pdf) 
.
* **jump\_n\_sample** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 10) —
The number of times to make a forward time jump for a given chosen time sample. Take a look at Figure 9
and 10 in the
 [paper](https://arxiv.org/pdf/2201.09865.pdf) 
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
>>> from io import BytesIO
>>> import torch
>>> import PIL
>>> import requests
>>> from diffusers import RePaintPipeline, RePaintScheduler


>>> def download\_image(url):
...     response = requests.get(url)
...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


>>> img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/celeba\_hq\_256.png"
>>> mask_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/mask\_256.png"

>>> # Load the original image and the mask as PIL images
>>> original_image = download_image(img_url).resize((256, 256))
>>> mask_image = download_image(mask_url).resize((256, 256))

>>> # Load the RePaint scheduler and pipeline based on a pretrained DDPM model
>>> scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
>>> pipe = RePaintPipeline.from_pretrained("google/ddpm-ema-celebahq-256", scheduler=scheduler)
>>> pipe = pipe.to("cuda")

>>> generator = torch.Generator(device="cuda").manual_seed(0)
>>> output = pipe(
...     image=original_image,
...     mask_image=mask_image,
...     num_inference_steps=250,
...     eta=0.0,
...     jump_length=10,
...     jump_n_sample=10,
...     generator=generator,
... )
>>> inpainted_image = output.images[0]
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