# Latent Diffusion

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/latent_diffusion>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/latent_diffusion>



 Latent Diffusion was proposed in
 [High-Resolution Image Synthesis with Latent Diffusion Models](https://huggingface.co/papers/2112.10752) 
 by Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer.
 



 The abstract from the paper is:
 



*By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. Our latent diffusion models (LDMs) achieve a new state of the art for image inpainting and highly competitive performance on various tasks, including unconditional image generation, semantic scene synthesis, and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs.* 




 The original codebase can be found at
 [Compvis/latent-diffusion](https://github.com/CompVis/latent-diffusion) 
.
 




 Make sure to check out the Schedulers
 [guide](../../using-diffusers/schedulers) 
 to learn how to explore the tradeoff between scheduler speed and quality, and see the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## LDMTextToImagePipeline




### 




 class
 

 diffusers.
 

 LDMTextToImagePipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py#L32)



 (
 


 vqvae
 
 : typing.Union[diffusers.models.vq\_model.VQModel, diffusers.models.autoencoder\_kl.AutoencoderKL]
 




 bert
 
 : PreTrainedModel
 




 tokenizer
 
 : PreTrainedTokenizer
 




 unet
 
 : typing.Union[diffusers.models.unet\_2d.UNet2DModel, diffusers.models.unet\_2d\_condition.UNet2DConditionModel]
 




 scheduler
 
 : typing.Union[diffusers.schedulers.scheduling\_ddim.DDIMScheduler, diffusers.schedulers.scheduling\_pndm.PNDMScheduler, diffusers.schedulers.scheduling\_lms\_discrete.LMSDiscreteScheduler]
 



 )
 


 Parameters
 




* **vqvae** 
 (
 [VQModel](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.VQModel) 
 ) —
Vector-quantized (VQ) model to encode and decode images to and from latent representations.
* **bert** 
 (
 `LDMBertModel` 
 ) —
Text-encoder model based on
 `BERT` 
.
* **tokenizer** 
 (
 [BertTokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/bert#transformers.BertTokenizer) 
 ) —
A
 `BertTokenizer` 
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


 Pipeline for text-to-image generation using latent diffusion.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py#L66)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]]
 




 height
 
 : typing.Optional[int] = None
 




 width
 
 : typing.Optional[int] = None
 




 num\_inference\_steps
 
 : typing.Optional[int] = 50
 




 guidance\_scale
 
 : typing.Optional[float] = 1.0
 




 eta
 
 : typing.Optional[float] = 0.0
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




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
 , defaults to 1.0) —
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
>>> from diffusers import DiffusionPipeline

>>> # load model and scheduler
>>> ldm = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

>>> # run pipeline in inference (sample random noise and denoise)
>>> prompt = "A painting of a squirrel eating a burger"
>>> images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6).images

>>> # save images
>>> for idx, image in enumerate(images):
...     image.save(f"squirrel-{idx}.png")
```


## LDMSuperResolutionPipeline




### 




 class
 

 diffusers.
 

 LDMSuperResolutionPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion_superresolution.py#L33)



 (
 


 vqvae
 
 : VQModel
 




 unet
 
 : UNet2DModel
 




 scheduler
 
 : typing.Union[diffusers.schedulers.scheduling\_ddim.DDIMScheduler, diffusers.schedulers.scheduling\_pndm.PNDMScheduler, diffusers.schedulers.scheduling\_lms\_discrete.LMSDiscreteScheduler, diffusers.schedulers.scheduling\_euler\_discrete.EulerDiscreteScheduler, diffusers.schedulers.scheduling\_euler\_ancestral\_discrete.EulerAncestralDiscreteScheduler, diffusers.schedulers.scheduling\_dpmsolver\_multistep.DPMSolverMultistepScheduler]
 



 )
 


 Parameters
 




* **vqvae** 
 (
 [VQModel](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.VQModel) 
 ) —
Vector-quantized (VQ) model to encode and decode images to and from latent representations.
* **unet** 
 (
 [UNet2DModel](/docs/diffusers/v0.23.0/en/api/models/unet2d#diffusers.UNet2DModel) 
 ) —
A
 `UNet2DModel` 
 to denoise the encoded image.
* **scheduler** 
 (
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 ) —
A scheduler to be used in combination with
 `unet` 
 to denoise the encoded image latens. Can be one of
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 ,
 [LMSDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler) 
 ,
 [EulerDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/euler#diffusers.EulerDiscreteScheduler) 
 ,
 [EulerAncestralDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/euler_ancestral#diffusers.EulerAncestralDiscreteScheduler) 
 ,
 [DPMSolverMultistepScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler) 
 , or
 [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler) 
.


 A pipeline for image super-resolution using latent diffusion.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion_superresolution.py#L67)



 (
 


 image
 
 : typing.Union[torch.Tensor, PIL.Image.Image] = None
 




 batch\_size
 
 : typing.Optional[int] = 1
 




 num\_inference\_steps
 
 : typing.Optional[int] = 100
 




 eta
 
 : typing.Optional[float] = 0.0
 




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
 `torch.Tensor` 
 or
 `PIL.Image.Image` 
 ) —
 `Image` 
 or tensor representing an image batch to be used as the starting point for the process.
* **batch\_size** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
Number of images to generate.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
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
 


 Example:
 



```
>>> import requests
>>> from PIL import Image
>>> from io import BytesIO
>>> from diffusers import LDMSuperResolutionPipeline
>>> import torch

>>> # load model and scheduler
>>> pipeline = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")
>>> pipeline = pipeline.to("cuda")

>>> # let's download an image
>>> url = (
...     "https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"
... )
>>> response = requests.get(url)
>>> low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
>>> low_res_img = low_res_img.resize((128, 128))

>>> # run pipeline in inference (sample random noise and denoise)
>>> upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
>>> # save image
>>> upscaled_image.save("ldm\_generated\_image.png")
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