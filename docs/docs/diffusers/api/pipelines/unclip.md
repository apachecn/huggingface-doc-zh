# unCLIP

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/unclip>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/unclip>



[Hierarchical Text-Conditional Image Generation with CLIP Latents](https://huggingface.co/papers/2204.06125) 
 is by Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen. The unCLIP model in 🤗 Diffusers comes from kakaobrain’s
 [karlo]((https://github.com/kakaobrain/karlo)) 
.
 



 The abstract from the paper is following:
 



*Contrastive models like CLIP have been shown to learn robust representations of images that capture both semantics and style. To leverage these representations for image generation, we propose a two-stage model: a prior that generates a CLIP image embedding given a text caption, and a decoder that generates an image conditioned on the image embedding. We show that explicitly generating image representations improves image diversity with minimal loss in photorealism and caption similarity. Our decoders conditioned on image representations can also produce variations of an image that preserve both its semantics and style, while varying the non-essential details absent from the image representation. Moreover, the joint embedding space of CLIP enables language-guided image manipulations in a zero-shot fashion. We use diffusion models for the decoder and experiment with both autoregressive and diffusion models for the prior, finding that the latter are computationally more efficient and produce higher-quality samples.* 




 You can find lucidrains DALL-E 2 recreation at
 [lucidrains/DALLE2-pytorch](https://github.com/lucidrains/DALLE2-pytorch) 
.
 




 Make sure to check out the Schedulers
 [guide](../../using-diffusers/schedulers) 
 to learn how to explore the tradeoff between scheduler speed and quality, and see the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## UnCLIPPipeline




### 




 class
 

 diffusers.
 

 UnCLIPPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unclip/pipeline_unclip.py#L34)



 (
 


 prior
 
 : PriorTransformer
 




 decoder
 
 : UNet2DConditionModel
 




 text\_encoder
 
 : CLIPTextModelWithProjection
 




 tokenizer
 
 : CLIPTokenizer
 




 text\_proj
 
 : UnCLIPTextProjModel
 




 super\_res\_first
 
 : UNet2DModel
 




 super\_res\_last
 
 : UNet2DModel
 




 prior\_scheduler
 
 : UnCLIPScheduler
 




 decoder\_scheduler
 
 : UnCLIPScheduler
 




 super\_res\_scheduler
 
 : UnCLIPScheduler
 



 )
 


 Parameters
 




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
* **prior** 
 (
 [PriorTransformer](/docs/diffusers/v0.23.0/en/api/models/prior_transformer#diffusers.PriorTransformer) 
 ) —
The canonical unCLIP prior to approximate the image embedding from the text embedding.
* **text\_proj** 
 (
 `UnCLIPTextProjModel` 
 ) —
Utility class to prepare and combine the embeddings before they are passed to the decoder.
* **decoder** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) —
The decoder to invert the image embedding into an image.
* **super\_res\_first** 
 (
 [UNet2DModel](/docs/diffusers/v0.23.0/en/api/models/unet2d#diffusers.UNet2DModel) 
 ) —
Super resolution UNet. Used in all but the last step of the super resolution diffusion process.
* **super\_res\_last** 
 (
 [UNet2DModel](/docs/diffusers/v0.23.0/en/api/models/unet2d#diffusers.UNet2DModel) 
 ) —
Super resolution UNet. Used in the last step of the super resolution diffusion process.
* **prior\_scheduler** 
 (
 `UnCLIPScheduler` 
 ) —
Scheduler used in the prior denoising process (a modified
 [DDPMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.DDPMScheduler) 
 ).
* **decoder\_scheduler** 
 (
 `UnCLIPScheduler` 
 ) —
Scheduler used in the decoder denoising process (a modified
 [DDPMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.DDPMScheduler) 
 ).
* **super\_res\_scheduler** 
 (
 `UnCLIPScheduler` 
 ) —
Scheduler used in the super resolution denoising process (a modified
 [DDPMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.DDPMScheduler) 
 ).


 Pipeline for text-to-image generation using unCLIP.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unclip/pipeline_unclip.py#L211)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_images\_per\_prompt
 
 : int = 1
 




 prior\_num\_inference\_steps
 
 : int = 25
 




 decoder\_num\_inference\_steps
 
 : int = 25
 




 super\_res\_num\_inference\_steps
 
 : int = 7
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 prior\_latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 decoder\_latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 super\_res\_latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 text\_model\_output
 
 : typing.Union[transformers.models.clip.modeling\_clip.CLIPTextModelOutput, typing.Tuple, NoneType] = None
 




 text\_attention\_mask
 
 : typing.Optional[torch.Tensor] = None
 




 prior\_guidance\_scale
 
 : float = 4.0
 




 decoder\_guidance\_scale
 
 : float = 8.0
 




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
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ) —
The prompt or prompts to guide image generation. This can only be left undefined if
 `text_model_output` 
 and
 `text_attention_mask` 
 is passed.
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **prior\_num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 25) —
The number of denoising steps for the prior. More denoising steps usually lead to a higher quality
image at the expense of slower inference.
* **decoder\_num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 25) —
The number of denoising steps for the decoder. More denoising steps usually lead to a higher quality
image at the expense of slower inference.
* **super\_res\_num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 7) —
The number of denoising steps for super resolution. More denoising steps usually lead to a higher
quality image at the expense of slower inference.
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
* **prior\_latents** 
 (
 `torch.FloatTensor` 
 of shape (batch size, embeddings dimension),
 *optional* 
 ) —
Pre-generated noisy latents to be used as inputs for the prior.
* **decoder\_latents** 
 (
 `torch.FloatTensor` 
 of shape (batch size, channels, height, width),
 *optional* 
 ) —
Pre-generated noisy latents to be used as inputs for the decoder.
* **super\_res\_latents** 
 (
 `torch.FloatTensor` 
 of shape (batch size, channels, super res height, super res width),
 *optional* 
 ) —
Pre-generated noisy latents to be used as inputs for the decoder.
* **prior\_guidance\_scale** 
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
* **decoder\_guidance\_scale** 
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
* **text\_model\_output** 
 (
 `CLIPTextModelOutput` 
 ,
 *optional* 
 ) —
Pre-defined
 `CLIPTextModel` 
 outputs that can be derived from the text encoder. Pre-defined text
outputs can be passed for tasks like text embedding interpolations. Make sure to also pass
 `text_attention_mask` 
 in this case.
 `prompt` 
 can the be left
 `None` 
.
* **text\_attention\_mask** 
 (
 `torch.Tensor` 
 ,
 *optional* 
 ) —
Pre-defined CLIP text attention mask that can be derived from the tokenizer. Pre-defined text attention
masks are necessary when passing
 `text_model_output` 
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
 


## UnCLIPImageVariationPipeline




### 




 class
 

 diffusers.
 

 UnCLIPImageVariationPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unclip/pipeline_unclip_image_variation.py#L39)



 (
 


 decoder
 
 : UNet2DConditionModel
 




 text\_encoder
 
 : CLIPTextModelWithProjection
 




 tokenizer
 
 : CLIPTokenizer
 




 text\_proj
 
 : UnCLIPTextProjModel
 




 feature\_extractor
 
 : CLIPImageProcessor
 




 image\_encoder
 
 : CLIPVisionModelWithProjection
 




 super\_res\_first
 
 : UNet2DModel
 




 super\_res\_last
 
 : UNet2DModel
 




 decoder\_scheduler
 
 : UnCLIPScheduler
 




 super\_res\_scheduler
 
 : UnCLIPScheduler
 



 )
 


 Parameters
 




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
* **feature\_extractor** 
 (
 [CLIPImageProcessor](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPImageProcessor) 
 ) —
Model that extracts features from generated images to be used as inputs for the
 `image_encoder` 
.
* **image\_encoder** 
 (
 [CLIPVisionModelWithProjection](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPVisionModelWithProjection) 
 ) —
Frozen CLIP image-encoder (
 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 
 ).
* **text\_proj** 
 (
 `UnCLIPTextProjModel` 
 ) —
Utility class to prepare and combine the embeddings before they are passed to the decoder.
* **decoder** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) —
The decoder to invert the image embedding into an image.
* **super\_res\_first** 
 (
 [UNet2DModel](/docs/diffusers/v0.23.0/en/api/models/unet2d#diffusers.UNet2DModel) 
 ) —
Super resolution UNet. Used in all but the last step of the super resolution diffusion process.
* **super\_res\_last** 
 (
 [UNet2DModel](/docs/diffusers/v0.23.0/en/api/models/unet2d#diffusers.UNet2DModel) 
 ) —
Super resolution UNet. Used in the last step of the super resolution diffusion process.
* **decoder\_scheduler** 
 (
 `UnCLIPScheduler` 
 ) —
Scheduler used in the decoder denoising process (a modified
 [DDPMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.DDPMScheduler) 
 ).
* **super\_res\_scheduler** 
 (
 `UnCLIPScheduler` 
 ) —
Scheduler used in the super resolution denoising process (a modified
 [DDPMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.DDPMScheduler) 
 ).


 Pipeline to generate image variations from an input image using UnCLIP.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unclip/pipeline_unclip_image_variation.py#L199)



 (
 


 image
 
 : typing.Union[PIL.Image.Image, typing.List[PIL.Image.Image], torch.FloatTensor, NoneType] = None
 




 num\_images\_per\_prompt
 
 : int = 1
 




 decoder\_num\_inference\_steps
 
 : int = 25
 




 super\_res\_num\_inference\_steps
 
 : int = 7
 




 generator
 
 : typing.Optional[torch.\_C.Generator] = None
 




 decoder\_latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 super\_res\_latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 image\_embeddings
 
 : typing.Optional[torch.Tensor] = None
 




 decoder\_guidance\_scale
 
 : float = 8.0
 




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
 `PIL.Image.Image` 
 or
 `List[PIL.Image.Image]` 
 or
 `torch.FloatTensor` 
 ) —
 `Image` 
 or tensor representing an image batch to be used as the starting point. If you provide a
tensor, it needs to be compatible with the
 `CLIPImageProcessor` 
[configuration](https://huggingface.co/fusing/karlo-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json) 
.
Can be left as
 `None` 
 only when
 `image_embeddings` 
 are passed.
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **decoder\_num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 25) —
The number of denoising steps for the decoder. More denoising steps usually lead to a higher quality
image at the expense of slower inference.
* **super\_res\_num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 7) —
The number of denoising steps for super resolution. More denoising steps usually lead to a higher
quality image at the expense of slower inference.
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
* **decoder\_latents** 
 (
 `torch.FloatTensor` 
 of shape (batch size, channels, height, width),
 *optional* 
 ) —
Pre-generated noisy latents to be used as inputs for the decoder.
* **super\_res\_latents** 
 (
 `torch.FloatTensor` 
 of shape (batch size, channels, super res height, super res width),
 *optional* 
 ) —
Pre-generated noisy latents to be used as inputs for the decoder.
* **decoder\_guidance\_scale** 
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
* **image\_embeddings** 
 (
 `torch.Tensor` 
 ,
 *optional* 
 ) —
Pre-defined image embeddings that can be derived from the image encoder. Pre-defined image embeddings
can be passed for tasks like image interpolations.
 `image` 
 can be left as
 `None` 
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