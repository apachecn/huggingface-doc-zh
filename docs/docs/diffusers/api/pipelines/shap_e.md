# Shap-E

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/shap_e>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/shap_e>



 The Shap-E model was proposed in
 [Shap-E: Generating Conditional 3D Implicit Functions](https://huggingface.co/papers/2305.02463) 
 by Alex Nichol and Heewon Jun from
 [OpenAI](https://github.com/openai) 
.
 



 The abstract from the paper is:
 



*We present Shap-E, a conditional generative model for 3D assets. Unlike recent work on 3D generative models which produce a single output representation, Shap-E directly generates the parameters of implicit functions that can be rendered as both textured meshes and neural radiance fields. We train Shap-E in two stages: first, we train an encoder that deterministically maps 3D assets into the parameters of an implicit function; second, we train a conditional diffusion model on outputs of the encoder. When trained on a large dataset of paired 3D and text data, our resulting models are capable of generating complex and diverse 3D assets in a matter of seconds. When compared to Point-E, an explicit generative model over point clouds, Shap-E converges faster and reaches comparable or better sample quality despite modeling a higher-dimensional, multi-representation output space.* 




 The original codebase can be found at
 [openai/shap-e](https://github.com/openai/shap-e) 
.
 




 See the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## ShapEPipeline




### 




 class
 

 diffusers.
 

 ShapEPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/shap_e/pipeline_shap_e.py#L79)



 (
 


 prior
 
 : PriorTransformer
 




 text\_encoder
 
 : CLIPTextModelWithProjection
 




 tokenizer
 
 : CLIPTokenizer
 




 scheduler
 
 : HeunDiscreteScheduler
 




 shap\_e\_renderer
 
 : ShapERenderer
 



 )
 


 Parameters
 




* **prior** 
 (
 [PriorTransformer](/docs/diffusers/v0.23.0/en/api/models/prior_transformer#diffusers.PriorTransformer) 
 ) —
The canonical unCLIP prior to approximate the image embedding from the text embedding.
* **text\_encoder** 
 (
 [CLIPTextModelWithProjection](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPTextModelWithProjection) 
 ) —
Frozen text-encoder.
* **tokenizer** 
 (
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPTokenizer) 
 ) —
A
 `CLIPTokenizer` 
 to tokenize text.
* **scheduler** 
 (
 [HeunDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/heun#diffusers.HeunDiscreteScheduler) 
 ) —
A scheduler to be used in combination with the
 `prior` 
 model to generate image embedding.
* **shap\_e\_renderer** 
 (
 `ShapERenderer` 
 ) —
Shap-E renderer projects the generated latents into parameters of a MLP to create 3D objects with the NeRF
rendering method.


 Pipeline for generating latent representation of a 3D asset and rendering with the NeRF method.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/shap_e/pipeline_shap_e.py#L182)



 (
 


 prompt
 
 : str
 




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
 




 frame\_size
 
 : int = 64
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[ShapEPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/shap_e#diffusers.pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput) 
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
 , defaults to 25) —
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
* **frame\_size** 
 (
 `int` 
 ,
 *optional* 
 , default to 64) —
The width and height of each image frame of the generated 3D output.
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pil"` 
 ) —
The output format of the generated image. Choose between
 `"pil"` 
 (
 `PIL.Image.Image` 
 ),
 `"np"` 
 (
 `np.array` 
 ),
 `"latent"` 
 (
 `torch.Tensor` 
 ), or mesh (
 `MeshDecoderOutput` 
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
 [ShapEPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/shap_e#diffusers.pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput) 
 instead of a plain
tuple.




 Returns
 




 export const metadata = 'undefined';
 

[ShapEPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/shap_e#diffusers.pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is
 `True` 
 ,
 [ShapEPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/shap_e#diffusers.pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput) 
 is returned,
otherwise a
 `tuple` 
 is returned where the first element is a list with the generated images.
 



 The call function to the pipeline for generation.
 


 Examples:
 



```
>>> import torch
>>> from diffusers import DiffusionPipeline
>>> from diffusers.utils import export_to_gif

>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

>>> repo = "openai/shap-e"
>>> pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16)
>>> pipe = pipe.to(device)

>>> guidance_scale = 15.0
>>> prompt = "a shark"

>>> images = pipe(
...     prompt,
...     guidance_scale=guidance_scale,
...     num_inference_steps=64,
...     frame_size=256,
... ).images

>>> gif_path = export_to_gif(images[0], "shark\_3d.gif")
```


## ShapEImg2ImgPipeline




### 




 class
 

 diffusers.
 

 ShapEImg2ImgPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/shap_e/pipeline_shap_e_img2img.py#L80)



 (
 


 prior
 
 : PriorTransformer
 




 image\_encoder
 
 : CLIPVisionModel
 




 image\_processor
 
 : CLIPImageProcessor
 




 scheduler
 
 : HeunDiscreteScheduler
 




 shap\_e\_renderer
 
 : ShapERenderer
 



 )
 


 Parameters
 




* **prior** 
 (
 [PriorTransformer](/docs/diffusers/v0.23.0/en/api/models/prior_transformer#diffusers.PriorTransformer) 
 ) —
The canonincal unCLIP prior to approximate the image embedding from the text embedding.
* **image\_encoder** 
 (
 [CLIPVisionModel](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPVisionModel) 
 ) —
Frozen image-encoder.
* **image\_processor** 
 (
 [CLIPImageProcessor](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPImageProcessor) 
 ) —
A
 `CLIPImageProcessor` 
 to process images.
* **scheduler** 
 (
 [HeunDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/heun#diffusers.HeunDiscreteScheduler) 
 ) —
A scheduler to be used in combination with the
 `prior` 
 model to generate image embedding.
* **shap\_e\_renderer** 
 (
 `ShapERenderer` 
 ) —
Shap-E renderer projects the generated latents into parameters of a MLP to create 3D objects with the NeRF
rendering method.


 Pipeline for generating latent representation of a 3D asset and rendering with the NeRF method from an image.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/shap_e/pipeline_shap_e_img2img.py#L164)



 (
 


 image
 
 : typing.Union[PIL.Image.Image, typing.List[PIL.Image.Image]]
 




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
 




 frame\_size
 
 : int = 64
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[ShapEPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/shap_e#diffusers.pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput) 
 or
 `tuple` 


 Parameters
 




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
 or tensor representing an image batch to be used as the starting point. Can also accept image
latents as image, but if passing latents directly it is not encoded again.
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
 , defaults to 25) —
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
* **frame\_size** 
 (
 `int` 
 ,
 *optional* 
 , default to 64) —
The width and height of each image frame of the generated 3D output.
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pil"` 
 ) —
The output format of the generated image. Choose between
 `"pil"` 
 (
 `PIL.Image.Image` 
 ),
 `"np"` 
 (
 `np.array` 
 ),
 `"latent"` 
 (
 `torch.Tensor` 
 ), or mesh (
 `MeshDecoderOutput` 
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
 [ShapEPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/shap_e#diffusers.pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput) 
 instead of a plain
tuple.




 Returns
 




 export const metadata = 'undefined';
 

[ShapEPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/shap_e#diffusers.pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is
 `True` 
 ,
 [ShapEPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/shap_e#diffusers.pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput) 
 is returned,
otherwise a
 `tuple` 
 is returned where the first element is a list with the generated images.
 



 The call function to the pipeline for generation.
 


 Examples:
 



```
>>> from PIL import Image
>>> import torch
>>> from diffusers import DiffusionPipeline
>>> from diffusers.utils import export_to_gif, load_image

>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

>>> repo = "openai/shap-e-img2img"
>>> pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16)
>>> pipe = pipe.to(device)

>>> guidance_scale = 3.0
>>> image_url = "https://hf.co/datasets/diffusers/docs-images/resolve/main/shap-e/corgi.png"
>>> image = load_image(image_url).convert("RGB")

>>> images = pipe(
...     image,
...     guidance_scale=guidance_scale,
...     num_inference_steps=64,
...     frame_size=256,
... ).images

>>> gif_path = export_to_gif(images[0], "corgi\_3d.gif")
```


## ShapEPipelineOutput




### 




 class
 

 diffusers.pipelines.shap\_e.pipeline\_shap\_e.
 

 ShapEPipelineOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/shap_e/pipeline_shap_e.py#L67)



 (
 


 images
 
 : typing.Union[typing.List[typing.List[PIL.Image.Image]], typing.List[typing.List[numpy.ndarray]]]
 



 )
 


 Parameters
 




* **images** 
 (
 `torch.FloatTensor` 
 ) —
A list of images for 3D rendering.


 Output class for
 [ShapEPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/shap_e#diffusers.ShapEPipeline) 
 and
 [ShapEImg2ImgPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/shap_e#diffusers.ShapEImg2ImgPipeline) 
.