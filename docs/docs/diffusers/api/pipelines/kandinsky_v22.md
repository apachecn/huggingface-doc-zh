# Kandinsky 2.2

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/kandinsky_v22>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/kandinsky_v22>



 Kandinsky 2.1 is created by
 [Arseniy Shakhmatov](https://github.com/cene555) 
 ,
 [Anton Razzhigaev](https://github.com/razzant) 
 ,
 [Aleksandr Nikolich](https://github.com/AlexWortega) 
 ,
 [Igor Pavlov](https://github.com/boomb0om) 
 ,
 [Andrey Kuznetsov](https://github.com/kuznetsoffandrey) 
 and
 [Denis Dimitrov](https://github.com/denndimitrov) 
.
 



 The description from it’s GitHub page is:
 



*Kandinsky 2.2 brings substantial improvements upon its predecessor, Kandinsky 2.1, by introducing a new, more powerful image encoder - CLIP-ViT-G and the ControlNet support. The switch to CLIP-ViT-G as the image encoder significantly increases the model’s capability to generate more aesthetic pictures and better understand text, thus enhancing the model’s overall performance. The addition of the ControlNet mechanism allows the model to effectively control the process of generating images. This leads to more accurate and visually appealing outputs and opens new possibilities for text-guided image manipulation.* 




 The original codebase can be found at
 [ai-forever/Kandinsky-2](https://github.com/ai-forever/Kandinsky-2) 
.
 




 Check out the
 [Kandinsky Community](https://huggingface.co/kandinsky-community) 
 organization on the Hub for the official model checkpoints for tasks like text-to-image, image-to-image, and inpainting.
 


## KandinskyV22PriorPipeline




### 




 class
 

 diffusers.
 

 KandinskyV22PriorPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior.py#L84)



 (
 


 prior
 
 : PriorTransformer
 




 image\_encoder
 
 : CLIPVisionModelWithProjection
 




 text\_encoder
 
 : CLIPTextModelWithProjection
 




 tokenizer
 
 : CLIPTokenizer
 




 scheduler
 
 : UnCLIPScheduler
 




 image\_processor
 
 : CLIPImageProcessor
 



 )
 


 Parameters
 




* **prior** 
 (
 [PriorTransformer](/docs/diffusers/v0.23.0/en/api/models/prior_transformer#diffusers.PriorTransformer) 
 ) —
The canonincal unCLIP prior to approximate the image embedding from the text embedding.
* **image\_encoder** 
 (
 `CLIPVisionModelWithProjection` 
 ) —
Frozen image-encoder.
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
 `UnCLIPScheduler` 
 ) —
A scheduler to be used in combination with
 `prior` 
 to generate image embedding.
* **image\_processor** 
 (
 `CLIPImageProcessor` 
 ) —
A image\_processor to be used to preprocess image from clip.


 Pipeline for generating image prior for Kandinsky
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior.py#L370)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]]
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_images\_per\_prompt
 
 : int = 1
 




 num\_inference\_steps
 
 : int = 25
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 guidance\_scale
 
 : float = 4.0
 




 output\_type
 
 : typing.Optional[str] = 'pt'
 




 return\_dict
 
 : bool = True
 




 callback\_on\_step\_end
 
 : typing.Union[typing.Callable[[int, int, typing.Dict], NoneType], NoneType] = None
 




 callback\_on\_step\_end\_tensor\_inputs
 
 : typing.List[str] = ['latents']
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`KandinskyPriorPipelineOutput` 
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
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
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
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
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
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pt"` 
 ) —
The output format of the generate image. Choose between:
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




 Returns
 




 export const metadata = 'undefined';
 

`KandinskyPriorPipelineOutput` 
 or
 `tuple` 




 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
>>> import torch

>>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")
>>> pipe_prior.to("cuda")
>>> prompt = "red cat, 4k photo"
>>> image_emb, negative_image_emb = pipe_prior(prompt).to_tuple()

>>> pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
>>> pipe.to("cuda")
>>> image = pipe(
...     image_embeds=image_emb,
...     negative_image_embeds=negative_image_emb,
...     height=768,
...     width=768,
...     num_inference_steps=50,
... ).images
>>> image[0].save("cat.png")
```


#### 




 interpolate




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior.py#L131)



 (
 


 images\_and\_prompts
 
 : typing.List[typing.Union[str, PIL.Image.Image, torch.FloatTensor]]
 




 weights
 
 : typing.List[float]
 




 num\_images\_per\_prompt
 
 : int = 1
 




 num\_inference\_steps
 
 : int = 25
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prior\_prompt
 
 : typing.Optional[str] = None
 




 negative\_prompt
 
 : str = ''
 




 guidance\_scale
 
 : float = 4.0
 




 device
 
 = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`KandinskyPriorPipelineOutput` 
 or
 `tuple` 


 Parameters
 




* **images\_and\_prompts** 
 (
 `List[Union[str, PIL.Image.Image, torch.FloatTensor]]` 
 ) —
list of prompts and images to guide the image generation.
weights — (
 `List[float]` 
 ):
list of weights for each condition in
 `images_and_prompts`
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
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
* **negative\_prior\_prompt** 
 (
 `str` 
 ,
 *optional* 
 ) —
The prompt not to guide the prior diffusion process. Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is less than
 `1` 
 ).
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is less than
 `1` 
 ).
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
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




 Returns
 




 export const metadata = 'undefined';
 

`KandinskyPriorPipelineOutput` 
 or
 `tuple` 




 Function invoked when using the prior pipeline for interpolation.
 


 Examples:
 



```
>>> from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
>>> from diffusers.utils import load_image
>>> import PIL
>>> import torch
>>> from torchvision import transforms

>>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
... )
>>> pipe_prior.to("cuda")
>>> img1 = load_image(
...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
...     "/kandinsky/cat.png"
... )
>>> img2 = load_image(
...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
...     "/kandinsky/starry\_night.jpeg"
... )
>>> images_texts = ["a cat", img1, img2]
>>> weights = [0.3, 0.3, 0.4]
>>> out = pipe_prior.interpolate(images_texts, weights)
>>> pipe = KandinskyV22Pipeline.from_pretrained(
...     "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
... )
>>> pipe.to("cuda")
>>> image = pipe(
...     image_embeds=out.image_embeds,
...     negative_image_embeds=out.negative_image_embeds,
...     height=768,
...     width=768,
...     num_inference_steps=50,
... ).images[0]
>>> image.save("starry\_cat.png")
```


## KandinskyV22Pipeline




### 




 class
 

 diffusers.
 

 KandinskyV22Pipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2.py#L64)



 (
 


 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : DDPMScheduler
 




 movq
 
 : VQModel
 



 )
 


 Parameters
 




* **scheduler** 
 (Union[
 `DDIMScheduler` 
 ,
 `DDPMScheduler` 
 ]) —
A scheduler to be used in combination with
 `unet` 
 to generate image latents.
* **unet** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) —
Conditional U-Net architecture to denoise the image embedding.
* **movq** 
 (
 [VQModel](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.VQModel) 
 ) —
MoVQ Decoder to generate the image from the latents.


 Pipeline for text-to-image generation using Kandinsky
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2.py#L122)



 (
 


 image\_embeds
 
 : typing.Union[torch.FloatTensor, typing.List[torch.FloatTensor]]
 




 negative\_image\_embeds
 
 : typing.Union[torch.FloatTensor, typing.List[torch.FloatTensor]]
 




 height
 
 : int = 512
 




 width
 
 : int = 512
 




 num\_inference\_steps
 
 : int = 100
 




 guidance\_scale
 
 : float = 4.0
 




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
 

 →
 



 export const metadata = 'undefined';
 

[ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 or
 `tuple` 


 Parameters
 




* **image\_embeds** 
 (
 `torch.FloatTensor` 
 or
 `List[torch.FloatTensor]` 
 ) —
The clip image embeddings for text prompt, that will be used to condition the image generation.
* **negative\_image\_embeds** 
 (
 `torch.FloatTensor` 
 or
 `List[torch.FloatTensor]` 
 ) —
The clip image embeddings for negative text prompt, will be used to condition the image generation.
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
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
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




 Returns
 




 export const metadata = 'undefined';
 

[ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 or
 `tuple` 




 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
>>> import torch

>>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")
>>> pipe_prior.to("cuda")
>>> prompt = "red cat, 4k photo"
>>> out = pipe_prior(prompt)
>>> image_emb = out.image_embeds
>>> zero_image_emb = out.negative_image_embeds
>>> pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
>>> pipe.to("cuda")
>>> image = pipe(
...     image_embeds=image_emb,
...     negative_image_embeds=zero_image_emb,
...     height=768,
...     width=768,
...     num_inference_steps=50,
... ).images
>>> image[0].save("cat.png")
```


## KandinskyV22CombinedPipeline




### 




 class
 

 diffusers.
 

 KandinskyV22CombinedPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L107)



 (
 


 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : DDPMScheduler
 




 movq
 
 : VQModel
 




 prior\_prior
 
 : PriorTransformer
 




 prior\_image\_encoder
 
 : CLIPVisionModelWithProjection
 




 prior\_text\_encoder
 
 : CLIPTextModelWithProjection
 




 prior\_tokenizer
 
 : CLIPTokenizer
 




 prior\_scheduler
 
 : UnCLIPScheduler
 




 prior\_image\_processor
 
 : CLIPImageProcessor
 



 )
 


 Parameters
 




* **scheduler** 
 (Union[
 `DDIMScheduler` 
 ,
 `DDPMScheduler` 
 ]) —
A scheduler to be used in combination with
 `unet` 
 to generate image latents.
* **unet** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) —
Conditional U-Net architecture to denoise the image embedding.
* **movq** 
 (
 [VQModel](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.VQModel) 
 ) —
MoVQ Decoder to generate the image from the latents.
* **prior\_prior** 
 (
 [PriorTransformer](/docs/diffusers/v0.23.0/en/api/models/prior_transformer#diffusers.PriorTransformer) 
 ) —
The canonincal unCLIP prior to approximate the image embedding from the text embedding.
* **prior\_image\_encoder** 
 (
 `CLIPVisionModelWithProjection` 
 ) —
Frozen image-encoder.
* **prior\_text\_encoder** 
 (
 `CLIPTextModelWithProjection` 
 ) —
Frozen text-encoder.
* **prior\_tokenizer** 
 (
 `CLIPTokenizer` 
 ) —
Tokenizer of class
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 
.
* **prior\_scheduler** 
 (
 `UnCLIPScheduler` 
 ) —
A scheduler to be used in combination with
 `prior` 
 to generate image embedding.
* **prior\_image\_processor** 
 (
 `CLIPImageProcessor` 
 ) —
A image\_processor to be used to preprocess image from clip.


 Combined Pipeline for text-to-image generation using Kandinsky
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L201)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]]
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_inference\_steps
 
 : int = 100
 




 guidance\_scale
 
 : float = 4.0
 




 num\_images\_per\_prompt
 
 : int = 1
 




 height
 
 : int = 512
 




 width
 
 : int = 512
 




 prior\_guidance\_scale
 
 : float = 4.0
 




 prior\_num\_inference\_steps
 
 : int = 25
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : int = 1
 




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
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
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
* **prior\_num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
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
A function that calls at the end of each denoising steps during the inference of the prior pipeline.
The function is called with the following arguments:
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
 attribute of your prior pipeline class.
* **callback\_on\_step\_end** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
A function that calls at the end of each denoising steps during the inference of the decoder pipeline.
The function is called with the following arguments:
 `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)` 
.
 `callback_kwargs` 
 will include a list of all tensors
as specified by
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
 

[ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 or
 `tuple` 




 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

prompt = "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k"

image = pipe(prompt=prompt, num_inference_steps=25).images[0]
```


#### 




 enable\_sequential\_cpu\_offload




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L181)



 (
 


 gpu\_id
 
 = 0
 



 )
 




 Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
text\_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
 `torch.device('meta') and loaded to GPU only when their specific submodule has its` 
 forward
 `method called. Note that offloading happens on a submodule basis. Memory savings are higher than with` 
 enable\_model\_cpu\_offload`, but performance is lower.
 


## KandinskyV22ControlnetPipeline




### 




 class
 

 diffusers.
 

 KandinskyV22ControlnetPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_controlnet.py#L106)



 (
 


 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : DDPMScheduler
 




 movq
 
 : VQModel
 



 )
 


 Parameters
 




* **scheduler** 
 (
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 ) —
A scheduler to be used in combination with
 `unet` 
 to generate image latents.
* **unet** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) —
Conditional U-Net architecture to denoise the image embedding.
* **movq** 
 (
 [VQModel](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.VQModel) 
 ) —
MoVQ Decoder to generate the image from the latents.


 Pipeline for text-to-image generation using Kandinsky
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_controlnet.py#L151)



 (
 


 image\_embeds
 
 : typing.Union[torch.FloatTensor, typing.List[torch.FloatTensor]]
 




 negative\_image\_embeds
 
 : typing.Union[torch.FloatTensor, typing.List[torch.FloatTensor]]
 




 hint
 
 : FloatTensor
 




 height
 
 : int = 512
 




 width
 
 : int = 512
 




 num\_inference\_steps
 
 : int = 100
 




 guidance\_scale
 
 : float = 4.0
 




 num\_images\_per\_prompt
 
 : int = 1
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : int = 1
 




 return\_dict
 
 : bool = True
 



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
* **hint** 
 (
 `torch.FloatTensor` 
 ) —
The controlnet condition.
* **image\_embeds** 
 (
 `torch.FloatTensor` 
 or
 `List[torch.FloatTensor]` 
 ) —
The clip image embeddings for text prompt, that will be used to condition the image generation.
* **negative\_image\_embeds** 
 (
 `torch.FloatTensor` 
 or
 `List[torch.FloatTensor]` 
 ) —
The clip image embeddings for negative text prompt, will be used to condition the image generation.
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
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
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




 Function invoked when calling the pipeline for generation.
 



 Examples:
 


## KandinskyV22PriorEmb2EmbPipeline




### 




 class
 

 diffusers.
 

 KandinskyV22PriorEmb2EmbPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior_emb2emb.py#L102)



 (
 


 prior
 
 : PriorTransformer
 




 image\_encoder
 
 : CLIPVisionModelWithProjection
 




 text\_encoder
 
 : CLIPTextModelWithProjection
 




 tokenizer
 
 : CLIPTokenizer
 




 scheduler
 
 : UnCLIPScheduler
 




 image\_processor
 
 : CLIPImageProcessor
 



 )
 


 Parameters
 




* **prior** 
 (
 [PriorTransformer](/docs/diffusers/v0.23.0/en/api/models/prior_transformer#diffusers.PriorTransformer) 
 ) —
The canonincal unCLIP prior to approximate the image embedding from the text embedding.
* **image\_encoder** 
 (
 `CLIPVisionModelWithProjection` 
 ) —
Frozen image-encoder.
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
 `UnCLIPScheduler` 
 ) —
A scheduler to be used in combination with
 `prior` 
 to generate image embedding.


 Pipeline for generating image prior for Kandinsky
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior_emb2emb.py#L396)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]]
 




 image
 
 : typing.Union[torch.Tensor, typing.List[torch.Tensor], PIL.Image.Image, typing.List[PIL.Image.Image]]
 




 strength
 
 : float = 0.3
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_images\_per\_prompt
 
 : int = 1
 




 num\_inference\_steps
 
 : int = 25
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 guidance\_scale
 
 : float = 4.0
 




 output\_type
 
 : typing.Optional[str] = 'pt'
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`KandinskyPriorPipelineOutput` 
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
* **strength** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.8) —
Conceptually, indicates how much to transform the reference
 `emb` 
. Must be between 0 and 1.
 `image` 
 will be used as a starting point, adding more noise to it the larger the
 `strength` 
. The number of
denoising steps depends on the amount of noise initially added.
* **emb** 
 (
 `torch.FloatTensor` 
 ) —
The image embedding.
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
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
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
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
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
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pt"` 
 ) —
The output format of the generate image. Choose between:
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




 Returns
 




 export const metadata = 'undefined';
 

`KandinskyPriorPipelineOutput` 
 or
 `tuple` 




 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> from diffusers import KandinskyV22Pipeline, KandinskyV22PriorEmb2EmbPipeline
>>> import torch

>>> pipe_prior = KandinskyPriorPipeline.from_pretrained(
...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
... )
>>> pipe_prior.to("cuda")

>>> prompt = "red cat, 4k photo"
>>> img = load_image(
...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
...     "/kandinsky/cat.png"
... )
>>> image_emb, nagative_image_emb = pipe_prior(prompt, image=img, strength=0.2).to_tuple()

>>> pipe = KandinskyPipeline.from_pretrained(
...     "kandinsky-community/kandinsky-2-2-decoder, torch\_dtype=torch.float16"
... )
>>> pipe.to("cuda")

>>> image = pipe(
...     image_embeds=image_emb,
...     negative_image_embeds=negative_image_emb,
...     height=768,
...     width=768,
...     num_inference_steps=100,
... ).images

>>> image[0].save("cat.png")
```


#### 




 interpolate




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior_emb2emb.py#L155)



 (
 


 images\_and\_prompts
 
 : typing.List[typing.Union[str, PIL.Image.Image, torch.FloatTensor]]
 




 weights
 
 : typing.List[float]
 




 num\_images\_per\_prompt
 
 : int = 1
 




 num\_inference\_steps
 
 : int = 25
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prior\_prompt
 
 : typing.Optional[str] = None
 




 negative\_prompt
 
 : str = ''
 




 guidance\_scale
 
 : float = 4.0
 




 device
 
 = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`KandinskyPriorPipelineOutput` 
 or
 `tuple` 


 Parameters
 




* **images\_and\_prompts** 
 (
 `List[Union[str, PIL.Image.Image, torch.FloatTensor]]` 
 ) —
list of prompts and images to guide the image generation.
weights — (
 `List[float]` 
 ):
list of weights for each condition in
 `images_and_prompts`
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
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
* **negative\_prior\_prompt** 
 (
 `str` 
 ,
 *optional* 
 ) —
The prompt not to guide the prior diffusion process. Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is less than
 `1` 
 ).
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is less than
 `1` 
 ).
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
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




 Returns
 




 export const metadata = 'undefined';
 

`KandinskyPriorPipelineOutput` 
 or
 `tuple` 




 Function invoked when using the prior pipeline for interpolation.
 


 Examples:
 



```
>>> from diffusers import KandinskyV22PriorEmb2EmbPipeline, KandinskyV22Pipeline
>>> from diffusers.utils import load_image
>>> import PIL

>>> import torch
>>> from torchvision import transforms

>>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
... )
>>> pipe_prior.to("cuda")

>>> img1 = load_image(
...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
...     "/kandinsky/cat.png"
... )

>>> img2 = load_image(
...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
...     "/kandinsky/starry\_night.jpeg"
... )

>>> images_texts = ["a cat", img1, img2]
>>> weights = [0.3, 0.3, 0.4]
>>> image_emb, zero_image_emb = pipe_prior.interpolate(images_texts, weights)

>>> pipe = KandinskyV22Pipeline.from_pretrained(
...     "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
... )
>>> pipe.to("cuda")

>>> image = pipe(
...     image_embeds=image_emb,
...     negative_image_embeds=zero_image_emb,
...     height=768,
...     width=768,
...     num_inference_steps=150,
... ).images[0]

>>> image.save("starry\_cat.png")
```


## KandinskyV22Img2ImgPipeline




### 




 class
 

 diffusers.
 

 KandinskyV22Img2ImgPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_img2img.py#L92)



 (
 


 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : DDPMScheduler
 




 movq
 
 : VQModel
 



 )
 


 Parameters
 




* **scheduler** 
 (
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 ) —
A scheduler to be used in combination with
 `unet` 
 to generate image latents.
* **unet** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) —
Conditional U-Net architecture to denoise the image embedding.
* **movq** 
 (
 [VQModel](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.VQModel) 
 ) —
MoVQ Decoder to generate the image from the latents.


 Pipeline for image-to-image generation using Kandinsky
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_img2img.py#L190)



 (
 


 image\_embeds
 
 : typing.Union[torch.FloatTensor, typing.List[torch.FloatTensor]]
 




 image
 
 : typing.Union[torch.FloatTensor, PIL.Image.Image, typing.List[torch.FloatTensor], typing.List[PIL.Image.Image]]
 




 negative\_image\_embeds
 
 : typing.Union[torch.FloatTensor, typing.List[torch.FloatTensor]]
 




 height
 
 : int = 512
 




 width
 
 : int = 512
 




 num\_inference\_steps
 
 : int = 100
 




 guidance\_scale
 
 : float = 4.0
 




 strength
 
 : float = 0.3
 




 num\_images\_per\_prompt
 
 : int = 1
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




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
 

 →
 



 export const metadata = 'undefined';
 

[ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 or
 `tuple` 


 Parameters
 




* **image\_embeds** 
 (
 `torch.FloatTensor` 
 or
 `List[torch.FloatTensor]` 
 ) —
The clip image embeddings for text prompt, that will be used to condition the image generation.
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
 , or tensor representing an image batch, that will be used as the starting point for the
process. Can also accept image latents as
 `image` 
 , if passing latents directly, it will not be encoded
again.
* **strength** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.8) —
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
. A value of 1, therefore, essentially ignores
 `image` 
.
* **negative\_image\_embeds** 
 (
 `torch.FloatTensor` 
 or
 `List[torch.FloatTensor]` 
 ) —
The clip image embeddings for negative text prompt, will be used to condition the image generation.
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
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
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




 Returns
 




 export const metadata = 'undefined';
 

[ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 or
 `tuple` 




 Function invoked when calling the pipeline for generation.
 



 Examples:
 


## KandinskyV22Img2ImgCombinedPipeline




### 




 class
 

 diffusers.
 

 KandinskyV22Img2ImgCombinedPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L334)



 (
 


 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : DDPMScheduler
 




 movq
 
 : VQModel
 




 prior\_prior
 
 : PriorTransformer
 




 prior\_image\_encoder
 
 : CLIPVisionModelWithProjection
 




 prior\_text\_encoder
 
 : CLIPTextModelWithProjection
 




 prior\_tokenizer
 
 : CLIPTokenizer
 




 prior\_scheduler
 
 : UnCLIPScheduler
 




 prior\_image\_processor
 
 : CLIPImageProcessor
 



 )
 


 Parameters
 




* **scheduler** 
 (Union[
 `DDIMScheduler` 
 ,
 `DDPMScheduler` 
 ]) —
A scheduler to be used in combination with
 `unet` 
 to generate image latents.
* **unet** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) —
Conditional U-Net architecture to denoise the image embedding.
* **movq** 
 (
 [VQModel](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.VQModel) 
 ) —
MoVQ Decoder to generate the image from the latents.
* **prior\_prior** 
 (
 [PriorTransformer](/docs/diffusers/v0.23.0/en/api/models/prior_transformer#diffusers.PriorTransformer) 
 ) —
The canonincal unCLIP prior to approximate the image embedding from the text embedding.
* **prior\_image\_encoder** 
 (
 `CLIPVisionModelWithProjection` 
 ) —
Frozen image-encoder.
* **prior\_text\_encoder** 
 (
 `CLIPTextModelWithProjection` 
 ) —
Frozen text-encoder.
* **prior\_tokenizer** 
 (
 `CLIPTokenizer` 
 ) —
Tokenizer of class
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 
.
* **prior\_scheduler** 
 (
 `UnCLIPScheduler` 
 ) —
A scheduler to be used in combination with
 `prior` 
 to generate image embedding.
* **prior\_image\_processor** 
 (
 `CLIPImageProcessor` 
 ) —
A image\_processor to be used to preprocess image from clip.


 Combined Pipeline for image-to-image generation using Kandinsky
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L438)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]]
 




 image
 
 : typing.Union[torch.FloatTensor, PIL.Image.Image, typing.List[torch.FloatTensor], typing.List[PIL.Image.Image]]
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_inference\_steps
 
 : int = 100
 




 guidance\_scale
 
 : float = 4.0
 




 strength
 
 : float = 0.3
 




 num\_images\_per\_prompt
 
 : int = 1
 




 height
 
 : int = 512
 




 width
 
 : int = 512
 




 prior\_guidance\_scale
 
 : float = 4.0
 




 prior\_num\_inference\_steps
 
 : int = 25
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : int = 1
 




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
 , or tensor representing an image batch, that will be used as the starting point for the
process. Can also accept image latents as
 `image` 
 , if passing latents directly, it will not be encoded
again.
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
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
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
. A value of 1, therefore, essentially ignores
 `image` 
.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
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
* **prior\_num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
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




 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
from diffusers import AutoPipelineForImage2Image
import torch
import requests
from io import BytesIO
from PIL import Image
import os

pipe = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

prompt = "A fantasy landscape, Cinematic lighting"
negative_prompt = "low quality, bad quality"

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")
image.thumbnail((768, 768))

image = pipe(prompt=prompt, image=original_image, num_inference_steps=25).images[0]
```


#### 




 enable\_model\_cpu\_offload




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L408)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L418)



 (
 


 gpu\_id
 
 = 0
 



 )
 




 Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
text\_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
 `torch.device('meta') and loaded to GPU only when their specific submodule has its` 
 forward
 `method called. Note that offloading happens on a submodule basis. Memory savings are higher than with` 
 enable\_model\_cpu\_offload`, but performance is lower.
 


## KandinskyV22ControlnetImg2ImgPipeline




### 




 class
 

 diffusers.
 

 KandinskyV22ControlnetImg2ImgPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_controlnet_img2img.py#L120)



 (
 


 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : DDPMScheduler
 




 movq
 
 : VQModel
 



 )
 


 Parameters
 




* **scheduler** 
 (
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 ) —
A scheduler to be used in combination with
 `unet` 
 to generate image latents.
* **unet** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) —
Conditional U-Net architecture to denoise the image embedding.
* **movq** 
 (
 [VQModel](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.VQModel) 
 ) —
MoVQ Decoder to generate the image from the latents.


 Pipeline for image-to-image generation using Kandinsky
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_controlnet_img2img.py#L206)



 (
 


 image\_embeds
 
 : typing.Union[torch.FloatTensor, typing.List[torch.FloatTensor]]
 




 image
 
 : typing.Union[torch.FloatTensor, PIL.Image.Image, typing.List[torch.FloatTensor], typing.List[PIL.Image.Image]]
 




 negative\_image\_embeds
 
 : typing.Union[torch.FloatTensor, typing.List[torch.FloatTensor]]
 




 hint
 
 : FloatTensor
 




 height
 
 : int = 512
 




 width
 
 : int = 512
 




 num\_inference\_steps
 
 : int = 100
 




 guidance\_scale
 
 : float = 4.0
 




 strength
 
 : float = 0.3
 




 num\_images\_per\_prompt
 
 : int = 1
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : int = 1
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 or
 `tuple` 


 Parameters
 




* **image\_embeds** 
 (
 `torch.FloatTensor` 
 or
 `List[torch.FloatTensor]` 
 ) —
The clip image embeddings for text prompt, that will be used to condition the image generation.
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
 , or tensor representing an image batch, that will be used as the starting point for the
process. Can also accept image latents as
 `image` 
 , if passing latents directly, it will not be encoded
again.
* **strength** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.8) —
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
. A value of 1, therefore, essentially ignores
 `image` 
.
* **hint** 
 (
 `torch.FloatTensor` 
 ) —
The controlnet condition.
* **negative\_image\_embeds** 
 (
 `torch.FloatTensor` 
 or
 `List[torch.FloatTensor]` 
 ) —
The clip image embeddings for negative text prompt, will be used to condition the image generation.
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
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
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




 Function invoked when calling the pipeline for generation.
 



 Examples:
 


## KandinskyV22InpaintPipeline




### 




 class
 

 diffusers.
 

 KandinskyV22InpaintPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_inpainting.py#L235)



 (
 


 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : DDPMScheduler
 




 movq
 
 : VQModel
 



 )
 


 Parameters
 




* **scheduler** 
 (
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 ) —
A scheduler to be used in combination with
 `unet` 
 to generate image latents.
* **unet** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) —
Conditional U-Net architecture to denoise the image embedding.
* **movq** 
 (
 [VQModel](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.VQModel) 
 ) —
MoVQ Decoder to generate the image from the latents.


 Pipeline for text-guided image inpainting using Kandinsky2.1
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_inpainting.py#L294)



 (
 


 image\_embeds
 
 : typing.Union[torch.FloatTensor, typing.List[torch.FloatTensor]]
 




 image
 
 : typing.Union[torch.FloatTensor, PIL.Image.Image]
 




 mask\_image
 
 : typing.Union[torch.FloatTensor, PIL.Image.Image, numpy.ndarray]
 




 negative\_image\_embeds
 
 : typing.Union[torch.FloatTensor, typing.List[torch.FloatTensor]]
 




 height
 
 : int = 512
 




 width
 
 : int = 512
 




 num\_inference\_steps
 
 : int = 100
 




 guidance\_scale
 
 : float = 4.0
 




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
 

 →
 



 export const metadata = 'undefined';
 

[ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 or
 `tuple` 


 Parameters
 




* **image\_embeds** 
 (
 `torch.FloatTensor` 
 or
 `List[torch.FloatTensor]` 
 ) —
The clip image embeddings for text prompt, that will be used to condition the image generation.
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
 `np.array` 
 ) —
Tensor representing an image batch, to mask
 `image` 
. White pixels in the mask will be repainted, while
black pixels will be preserved. If
 `mask_image` 
 is a PIL image, it will be converted to a single
channel (luminance) before use. If it’s a tensor, it should contain one color channel (L) instead of 3,
so the expected shape would be
 `(B, H, W, 1)` 
.
* **negative\_image\_embeds** 
 (
 `torch.FloatTensor` 
 or
 `List[torch.FloatTensor]` 
 ) —
The clip image embeddings for negative text prompt, will be used to condition the image generation.
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
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
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




 Returns
 




 export const metadata = 'undefined';
 

[ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 or
 `tuple` 




 Function invoked when calling the pipeline for generation.
 



 Examples:
 


## KandinskyV22InpaintCombinedPipeline




### 




 class
 

 diffusers.
 

 KandinskyV22InpaintCombinedPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L582)



 (
 


 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : DDPMScheduler
 




 movq
 
 : VQModel
 




 prior\_prior
 
 : PriorTransformer
 




 prior\_image\_encoder
 
 : CLIPVisionModelWithProjection
 




 prior\_text\_encoder
 
 : CLIPTextModelWithProjection
 




 prior\_tokenizer
 
 : CLIPTokenizer
 




 prior\_scheduler
 
 : UnCLIPScheduler
 




 prior\_image\_processor
 
 : CLIPImageProcessor
 



 )
 


 Parameters
 




* **scheduler** 
 (Union[
 `DDIMScheduler` 
 ,
 `DDPMScheduler` 
 ]) —
A scheduler to be used in combination with
 `unet` 
 to generate image latents.
* **unet** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) —
Conditional U-Net architecture to denoise the image embedding.
* **movq** 
 (
 [VQModel](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.VQModel) 
 ) —
MoVQ Decoder to generate the image from the latents.
* **prior\_prior** 
 (
 [PriorTransformer](/docs/diffusers/v0.23.0/en/api/models/prior_transformer#diffusers.PriorTransformer) 
 ) —
The canonincal unCLIP prior to approximate the image embedding from the text embedding.
* **prior\_image\_encoder** 
 (
 `CLIPVisionModelWithProjection` 
 ) —
Frozen image-encoder.
* **prior\_text\_encoder** 
 (
 `CLIPTextModelWithProjection` 
 ) —
Frozen text-encoder.
* **prior\_tokenizer** 
 (
 `CLIPTokenizer` 
 ) —
Tokenizer of class
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 
.
* **prior\_scheduler** 
 (
 `UnCLIPScheduler` 
 ) —
A scheduler to be used in combination with
 `prior` 
 to generate image embedding.
* **prior\_image\_processor** 
 (
 `CLIPImageProcessor` 
 ) —
A image\_processor to be used to preprocess image from clip.


 Combined Pipeline for inpainting generation using Kandinsky
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L676)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]]
 




 image
 
 : typing.Union[torch.FloatTensor, PIL.Image.Image, typing.List[torch.FloatTensor], typing.List[PIL.Image.Image]]
 




 mask\_image
 
 : typing.Union[torch.FloatTensor, PIL.Image.Image, typing.List[torch.FloatTensor], typing.List[PIL.Image.Image]]
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_inference\_steps
 
 : int = 100
 




 guidance\_scale
 
 : float = 4.0
 




 num\_images\_per\_prompt
 
 : int = 1
 




 height
 
 : int = 512
 




 width
 
 : int = 512
 




 prior\_guidance\_scale
 
 : float = 4.0
 




 prior\_num\_inference\_steps
 
 : int = 25
 




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
 , or tensor representing an image batch, that will be used as the starting point for the
process. Can also accept image latents as
 `image` 
 , if passing latents directly, it will not be encoded
again.
* **mask\_image** 
 (
 `np.array` 
 ) —
Tensor representing an image batch, to mask
 `image` 
. White pixels in the mask will be repainted, while
black pixels will be preserved. If
 `mask_image` 
 is a PIL image, it will be converted to a single
channel (luminance) before use. If it’s a tensor, it should contain one color channel (L) instead of 3,
so the expected shape would be
 `(B, H, W, 1)` 
.
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
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 4.0) —
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
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
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
* **prior\_num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 100) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
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




 Returns
 




 export const metadata = 'undefined';
 

[ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 or
 `tuple` 




 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import numpy as np

pipe = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

prompt = "A fantasy landscape, Cinematic lighting"
negative_prompt = "low quality, bad quality"

original_image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main" "/kandinsky/cat.png"
)

mask = np.zeros((768, 768), dtype=np.float32)
# Let's mask out an area above the cat's head
mask[:250, 250:-250] = 1

image = pipe(prompt=prompt, image=original_image, mask_image=mask, num_inference_steps=25).images[0]
```


#### 




 enable\_sequential\_cpu\_offload




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L656)



 (
 


 gpu\_id
 
 = 0
 



 )
 




 Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
text\_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
 `torch.device('meta') and loaded to GPU only when their specific submodule has its` 
 forward
 `method called. Note that offloading happens on a submodule basis. Memory savings are higher than with` 
 enable\_model\_cpu\_offload`, but performance is lower.