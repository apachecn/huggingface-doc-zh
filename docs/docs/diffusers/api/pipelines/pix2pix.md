# InstructPix2Pix

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/pix2pix>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/pix2pix>



[InstructPix2Pix: Learning to Follow Image Editing Instructions](https://huggingface.co/papers/2211.09800) 
 is by Tim Brooks, Aleksander Holynski and Alexei A. Efros.
 



 The abstract from the paper is:
 



*We propose a method for editing images from human instructions: given an input image and a written instruction that tells the model what to do, our model follows these instructions to edit the image. To obtain training data for this problem, we combine the knowledge of two large pretrained models — a language model (GPT-3) and a text-to-image model (Stable Diffusion) — to generate a large dataset of image editing examples. Our conditional diffusion model, InstructPix2Pix, is trained on our generated data, and generalizes to real images and user-written instructions at inference time. Since it performs edits in the forward pass and does not require per example fine-tuning or inversion, our model edits images quickly, in a matter of seconds. We show compelling editing results for a diverse collection of input images and written instructions.* 




 You can find additional information about InstructPix2Pix on the
 [project page](https://www.timothybrooks.com/instruct-pix2pix) 
 ,
 [original codebase](https://github.com/timothybrooks/instruct-pix2pix) 
 , and try it out in a
 [demo](https://huggingface.co/spaces/timbrooks/instruct-pix2pix) 
.
 




 Make sure to check out the Schedulers
 [guide](../../using-diffusers/schedulers) 
 to learn how to explore the tradeoff between scheduler speed and quality, and see the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## StableDiffusionInstructPix2PixPipeline




### 




 class
 

 diffusers.
 

 StableDiffusionInstructPix2PixPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py#L61)



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
 




 safety\_checker
 
 : StableDiffusionSafetyChecker
 




 feature\_extractor
 
 : CLIPImageProcessor
 




 requires\_safety\_checker
 
 : bool = True
 



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
* **safety\_checker** 
 (
 `StableDiffusionSafetyChecker` 
 ) —
Classification module that estimates whether generated images could be considered offensive or harmful.
Please refer to the
 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) 
 for more details
about a model’s potential harms.
* **feature\_extractor** 
 (
 [CLIPImageProcessor](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPImageProcessor) 
 ) —
A
 `CLIPImageProcessor` 
 to extract features from generated images; used as inputs to the
 `safety_checker` 
.


 Pipeline for pixel-level image editing by following text instructions (based on Stable Diffusion).
 



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py#L139)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 image
 
 : typing.Union[PIL.Image.Image, numpy.ndarray, torch.FloatTensor, typing.List[PIL.Image.Image], typing.List[numpy.ndarray], typing.List[torch.FloatTensor]] = None
 




 num\_inference\_steps
 
 : int = 100
 




 guidance\_scale
 
 : float = 7.5
 




 image\_guidance\_scale
 
 : float = 1.5
 




 negative\_prompt
 
 : typing.Union[typing.List[str], str, NoneType] = None
 




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
`np.ndarray` 
 ,
 `PIL.Image.Image` 
 ,
 `List[torch.FloatTensor]` 
 ,
 `List[PIL.Image.Image]` 
 , or
 `List[np.ndarray]` 
 ) —
 `Image` 
 or tensor representing an image batch to be repainted according to
 `prompt` 
. Can also accept
image latents as
 `image` 
 , but if passing latents directly it is not encoded again.
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
 , defaults to 7.5) —
A higher guidance scale value encourages the model to generate images closely linked to the text
 `prompt` 
 at the expense of lower image quality. Guidance scale is enabled when
 `guidance_scale > 1` 
.
* **image\_guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1.5) —
Push the generated image towards the inital
 `image` 
. Image guidance scale is enabled by setting
 `image_guidance_scale > 1` 
. Higher image guidance scale encourages generated images that are closely
linked to the source
 `image` 
 , usually at the expense of lower image quality. This pipeline requires a
value of at least
 `1` 
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
 is returned where the first element is a list with the generated images and the
second element is a list of
 `bool` 
 s indicating whether the corresponding generated image contains
“not-safe-for-work” (nsfw) content.
 



 The call function to the pipeline for generation.
 


 Examples:
 



```
>>> import PIL
>>> import requests
>>> import torch
>>> from io import BytesIO

>>> from diffusers import StableDiffusionInstructPix2PixPipeline


>>> def download\_image(url):
...     response = requests.get(url)
...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


>>> img_url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"

>>> image = download_image(img_url).resize((512, 512))

>>> pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
...     "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
... )
>>> pipe = pipe.to("cuda")

>>> prompt = "make the mountains snowy"
>>> image = pipe(prompt=prompt, image=image).images[0]
```


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




 disable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py#L778)



 (
 

 )
 




 Disables the FreeU mechanism if enabled.
 




#### 




 enable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py#L755)



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
 


## StableDiffusionXLInstructPix2PixPipeline




### 




 class
 

 diffusers.
 

 StableDiffusionXLInstructPix2PixPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py#L105)



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
 requires a aesthetic\_score condition to be passed during inference. Also see the config
of
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


 Pipeline for pixel-level image editing by following text instructions. Based on Stable Diffusion XL.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



 In addition the pipeline inherits the following loading methods:
 


* *LoRA* 
 :
 [loaders.StableDiffusionXLLoraLoaderMixin.load\_lora\_weights()](/docs/diffusers/v0.23.0/en/api/loaders#diffusers.loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights)



 as well as the following saving methods:
 


* *LoRA* 
 :
 `loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights()`



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py#L645)



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
 
 : int = 100
 




 denoising\_end
 
 : typing.Optional[float] = None
 




 guidance\_scale
 
 : float = 5.0
 




 image\_guidance\_scale
 
 : float = 1.5
 




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
 




 guidance\_rescale
 
 : float = 0.0
 




 original\_size
 
 : typing.Tuple[int, int] = None
 




 crops\_coords\_top\_left
 
 : typing.Tuple[int, int] = (0, 0)
 




 target\_size
 
 : typing.Tuple[int, int] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` 
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
 or
 `PIL.Image.Image` 
 or
 `np.ndarray` 
 or
 `List[torch.FloatTensor]` 
 or
 `List[PIL.Image.Image]` 
 or
 `List[np.ndarray]` 
 ) —
The image(s) to modify with the pipeline.
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
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 50) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **denoising\_end** 
 (
 `float` 
 ,
 *optional* 
 ) —
When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
completed before it is intentionally prematurely terminated. As a result, the returned sample will
still retain a substantial amount of noise as determined by the discrete timesteps selected by the
scheduler. The denoising\_end parameter should ideally be utilized when this pipeline forms a part of a
“Mixture of Denoisers” multi-pipeline setup, as elaborated in
 [**Refining the Image
Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 5.0) —
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
* **image\_guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1.5) —
Image guidance scale is to push the generated image towards the inital image
 `image` 
. Image guidance
scale is enabled by setting
 `image_guidance_scale > 1` 
. Higher image guidance scale encourages to
generate images that are closely linked to the source image
 `image` 
 , usually at the expense of lower
image quality. This pipeline requires a value of at least
 `1` 
.
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
 `~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput` 
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
* **guidance\_rescale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
Guidance rescale factor proposed by
 [Common Diffusion Noise Schedules and Sample Steps are
Flawed](https://arxiv.org/pdf/2305.08891.pdf) 
`guidance_scale` 
 is defined as
 `φ` 
 in equation 16. of
 [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) 
.
Guidance rescale factor should fix overexposure when using zero terminal SNR.
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




 Returns
 




 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` 
 if
 `return_dict` 
 is True, otherwise a
 `tuple` 
. When returning a tuple, the first element is a list with the generated images.
 



 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> import torch
>>> from diffusers import StableDiffusionXLInstructPix2PixPipeline
>>> from diffusers.utils import load_image

>>> resolution = 768
>>> image = load_image(
...     "https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
... ).resize((resolution, resolution))
>>> edit_instruction = "Turn sky into a cloudy one"

>>> pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
...     "diffusers/sdxl-instructpix2pix-768", torch_dtype=torch.float16
... ).to("cuda")

>>> edited_image = pipe(
...     prompt=edit_instruction,
...     image=image,
...     height=resolution,
...     width=resolution,
...     guidance_scale=3.0,
...     image_guidance_scale=1.5,
...     num_inference_steps=30,
... ).images[0]
>>> edited_image
```


#### 




 disable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py#L641)



 (
 

 )
 




 Disables the FreeU mechanism if enabled.
 




#### 




 disable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py#L201)



 (
 

 )
 




 Disable sliced VAE decoding. If
 `enable_vae_slicing` 
 was previously invoked, this method will go back to
computing decoding in one step.
 




#### 




 disable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py#L217)



 (
 

 )
 




 Disable tiled VAE decoding. If
 `enable_vae_tiling` 
 was previously invoked, this method will go back to
computing decoding in one step.
 




#### 




 enable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py#L618)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py#L192)



 (
 

 )
 




 Enable sliced VAE decoding.
 



 When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 enable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py#L208)



 (
 

 )
 




 Enable tiled VAE decoding.
 



 When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py#L224)



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


 Encodes the prompt into text encoder hidden states.