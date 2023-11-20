# Loaders

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/loaders>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/loaders>



 Adapters (textual inversion, LoRA, hypernetworks) allow you to modify a diffusion model to generate images in a specific style without training or finetuning the entire model. The adapter weights are typically only a tiny fraction of the pretrained model’s which making them very portable. 🤗 Diffusers provides an easy-to-use
 `LoaderMixin` 
 API to load adapter weights.
 




 🧪 The
 `LoaderMixins` 
 are highly experimental and prone to future changes. To use private or
 [gated](https://huggingface.co/docs/hub/models-gated#gated-models) 
 models, log-in with
 `huggingface-cli login` 
.
 


## UNet2DConditionLoadersMixin




### 




 class
 

 diffusers.loaders.
 

 UNet2DConditionLoadersMixin




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L256)



 (
 

 )
 




#### 




 disable\_lora




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L739)



 (
 

 )
 




 Disables the active LoRA layers for the unet.
 




#### 




 enable\_lora




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L747)



 (
 

 )
 




 Enables the active LoRA layers for the unet.
 




#### 




 load\_attn\_procs




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L260)



 (
 


 pretrained\_model\_name\_or\_path\_or\_dict
 
 : typing.Union[str, typing.Dict[str, torch.Tensor]]
 




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
Can be either:
 
	+ A string, the model id (for example
	 `google/ddpm-celebahq-256` 
	 ) of a pretrained model hosted on
	the Hub.
	+ A path to a directory (for example
	 `./my_model_directory` 
	 ) containing the model weights saved
	with
	 [ModelMixin.save\_pretrained()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin.save_pretrained) 
	.
	+ A
	 [torch state
	dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict) 
	.
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


 Load pretrained attention processor layers into
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
. Attention processor layers have to be
defined in
 [`attention_processor.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
 and be a
 `torch.nn.Module` 
 class.
 




#### 




 save\_attn\_procs




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L589)



 (
 


 save\_directory
 
 : typing.Union[str, os.PathLike]
 




 is\_main\_process
 
 : bool = True
 




 weight\_name
 
 : str = None
 




 save\_function
 
 : typing.Callable = None
 




 safe\_serialization
 
 : bool = True
 




 \*\*kwargs
 




 )
 


 Parameters
 




* **save\_directory** 
 (
 `str` 
 or
 `os.PathLike` 
 ) —
Directory to save an attention processor to. Will be created if it doesn’t exist.
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


 Save an attention processor to a directory so that it can be reloaded using the
 [load\_attn\_procs()](/docs/diffusers/v0.23.0/en/api/loaders#diffusers.loaders.UNet2DConditionLoadersMixin.load_attn_procs) 
 method.
 




#### 




 set\_adapters




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L707)



 (
 


 adapter\_names
 
 : typing.Union[typing.List[str], str]
 




 weights
 
 : typing.Union[typing.List[float], float, NoneType] = None
 



 )
 


 Parameters
 




* **adapter\_names** 
 (
 `List[str]` 
 or
 `str` 
 ) —
The names of the adapters to use.
* **weights** 
 (
 `Union[List[float], float]` 
 ,
 *optional* 
 ) —
The adapter(s) weights to use with the UNet. If
 `None` 
 , the weights are set to
 `1.0` 
 for all the
adapters.


 Sets the adapter layers for the unet.
 


## TextualInversionLoaderMixin




### 




 class
 

 diffusers.loaders.
 

 TextualInversionLoaderMixin




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L831)



 (
 

 )
 




 Load textual inversion tokens and embeddings to the tokenizer and text encoder.
 



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




 maybe\_convert\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L836)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]]
 




 tokenizer
 
 : PreTrainedTokenizer
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`str` 
 or list of
 `str` 


 Parameters
 




* **prompt** 
 (
 `str` 
 or list of
 `str` 
 ) —
The prompt or prompts to guide the image generation.
* **tokenizer** 
 (
 `PreTrainedTokenizer` 
 ) —
The tokenizer responsible for encoding the prompt into input tokens.




 Returns
 




 export const metadata = 'undefined';
 

`str` 
 or list of
 `str` 




 export const metadata = 'undefined';
 




 The converted prompt
 



 Processes prompts that include a special token corresponding to a multi-vector textual inversion embedding to
be replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
inversion token or if the textual inversion token is a single vector, the input prompt is returned.
 


## StableDiffusionXLLoraLoaderMixin




### 




 class
 

 diffusers.loaders.
 

 StableDiffusionXLLoraLoaderMixin




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L3186)



 (
 

 )
 




 This class overrides
 `LoraLoaderMixin` 
 with LoRA loading/saving code that’s specific to SDXL
 



#### 




 load\_lora\_weights




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L3190)



 (
 


 pretrained\_model\_name\_or\_path\_or\_dict
 
 : typing.Union[str, typing.Dict[str, torch.Tensor]]
 




 adapter\_name
 
 : typing.Optional[str] = None
 




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
* **adapter\_name** 
 (
 `str` 
 ,
 *optional* 
 ) —
Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
 `default_{i}` 
 where i is the total number of adapters being loaded.
* **kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
See
 [lora\_state\_dict()](/docs/diffusers/v0.23.0/en/api/loaders#diffusers.loaders.LoraLoaderMixin.lora_state_dict) 
.


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
 


## LoraLoaderMixin




### 




 class
 

 diffusers.loaders.
 

 LoraLoaderMixin




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L1164)



 (
 

 )
 




 Load LoRA layers into
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 and
 [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)
.
 



#### 




 disable\_lora\_for\_text\_encoder




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L2437)



 (
 


 text\_encoder
 
 : typing.Optional[ForwardRef('PreTrainedModel')] = None
 



 )
 


 Parameters
 




* **text\_encoder** 
 (
 `torch.nn.Module` 
 ,
 *optional* 
 ) —
The text encoder module to disable the LoRA layers for. If
 `None` 
 , it will try to get the
 `text_encoder` 
 attribute.


 Disables the LoRA layers for the text encoder.
 




#### 




 enable\_lora\_for\_text\_encoder




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L2454)



 (
 


 text\_encoder
 
 : typing.Optional[ForwardRef('PreTrainedModel')] = None
 



 )
 


 Parameters
 




* **text\_encoder** 
 (
 `torch.nn.Module` 
 ,
 *optional* 
 ) —
The text encoder module to enable the LoRA layers for. If
 `None` 
 , it will try to get the
 `text_encoder` 
 attribute.


 Enables the LoRA layers for the text encoder.
 




#### 




 fuse\_lora




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L2264)



 (
 


 fuse\_unet
 
 : bool = True
 




 fuse\_text\_encoder
 
 : bool = True
 




 lora\_scale
 
 : float = 1.0
 




 safe\_fusing
 
 : bool = False
 



 )
 


 Parameters
 




* **fuse\_unet** 
 (
 `bool` 
 , defaults to
 `True` 
 ) — Whether to fuse the UNet LoRA parameters.
* **fuse\_text\_encoder** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to fuse the text encoder LoRA parameters. If the text encoder wasn’t monkey-patched with the
LoRA parameters then it won’t have any effect.
* **lora\_scale** 
 (
 `float` 
 , defaults to 1.0) —
Controls how much to influence the outputs with the LoRA parameters.
* **safe\_fusing** 
 (
 `bool` 
 , defaults to
 `False` 
 ) —
Whether to check fused weights for NaN values before fusing and if values are NaN not fusing them.


 Fuses the LoRA parameters into the original parameters of the corresponding blocks.
 




 This is an experimental API.
 


#### 




 get\_active\_adapters




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L2510)



 (
 

 )
 




 Gets the list of the current active adapters.
 


 Example:
 



```
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
).to("cuda")
pipeline.load_lora_weights("CiroN2022/toy-face", weight_name="toy\_face\_sdxl.safetensors", adapter_name="toy")
pipeline.get_active_adapters()
```


#### 




 get\_list\_adapters




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L2542)



 (
 

 )
 




 Gets the current list of all available adapters in the pipeline.
 




#### 




 load\_lora\_into\_text\_encoder




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L1664)



 (
 


 state\_dict
 


 network\_alphas
 


 text\_encoder
 


 prefix
 
 = None
 




 lora\_scale
 
 = 1.0
 




 low\_cpu\_mem\_usage
 
 = None
 




 adapter\_name
 
 = None
 




 \_pipeline
 
 = None
 



 )
 


 Parameters
 




* **state\_dict** 
 (
 `dict` 
 ) —
A standard state dict containing the lora layer parameters. The key should be prefixed with an
additional
 `text_encoder` 
 to distinguish between unet lora layers.
* **network\_alphas** 
 (
 `Dict[str, float]` 
 ) —
See
 `LoRALinearLayer` 
 for more details.
* **text\_encoder** 
 (
 `CLIPTextModel` 
 ) —
The text encoder model to load the LoRA layers into.
* **prefix** 
 (
 `str` 
 ) —
Expected prefix of the
 `text_encoder` 
 in the
 `state_dict` 
.
* **lora\_scale** 
 (
 `float` 
 ) —
How much to scale the output of the lora linear layer before it is added with the output of the regular
lora layer.
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
* **adapter\_name** 
 (
 `str` 
 ,
 *optional* 
 ) —
Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
 `default_{i}` 
 where i is the total number of adapters being loaded.


 This will load the LoRA layers specified in
 `state_dict` 
 into
 `text_encoder` 


#### 




 load\_lora\_into\_unet




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L1560)



 (
 


 state\_dict
 


 network\_alphas
 


 unet
 


 low\_cpu\_mem\_usage
 
 = None
 




 adapter\_name
 
 = None
 




 \_pipeline
 
 = None
 



 )
 


 Parameters
 




* **state\_dict** 
 (
 `dict` 
 ) —
A standard state dict containing the lora layer parameters. The keys can either be indexed directly
into the unet or prefixed with an additional
 `unet` 
 which can be used to distinguish between text
encoder lora layers.
* **network\_alphas** 
 (
 `Dict[str, float]` 
 ) —
See
 `LoRALinearLayer` 
 for more details.
* **unet** 
 (
 `UNet2DConditionModel` 
 ) —
The UNet model to load the LoRA layers into.
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
* **adapter\_name** 
 (
 `str` 
 ,
 *optional* 
 ) —
Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
 `default_{i}` 
 where i is the total number of adapters being loaded.


 This will load the LoRA layers specified in
 `state_dict` 
 into
 `unet` 
.
 




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




 lora\_state\_dict




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L1228)



 (
 


 pretrained\_model\_name\_or\_path\_or\_dict
 
 : typing.Union[str, typing.Dict[str, torch.Tensor]]
 




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
Can be either:
 
	+ A string, the
	 *model id* 
	 (for example
	 `google/ddpm-celebahq-256` 
	 ) of a pretrained model hosted on
	the Hub.
	+ A path to a
	 *directory* 
	 (for example
	 `./my_model_directory` 
	 ) containing the model weights saved
	with
	 [ModelMixin.save\_pretrained()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin.save_pretrained) 
	.
	+ A
	 [torch state
	dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict) 
	.
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
* **mirror** 
 (
 `str` 
 ,
 *optional* 
 ) —
Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
information.


 Return state dict for lora weights and the network alphas.
 




 We support loading A1111 formatted LoRA checkpoints in a limited capacity.
 



 This function is experimental and might change in the future.
 


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




 set\_adapters\_for\_text\_encoder




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L2395)



 (
 


 adapter\_names
 
 : typing.Union[typing.List[str], str]
 




 text\_encoder
 
 : typing.Optional[ForwardRef('PreTrainedModel')] = None
 




 text\_encoder\_weights
 
 : typing.List[float] = None
 



 )
 


 Parameters
 




* **adapter\_names** 
 (
 `List[str]` 
 or
 `str` 
 ) —
The names of the adapters to use.
* **text\_encoder** 
 (
 `torch.nn.Module` 
 ,
 *optional* 
 ) —
The text encoder module to set the adapter layers for. If
 `None` 
 , it will try to get the
 `text_encoder` 
 attribute.
* **text\_encoder\_weights** 
 (
 `List[float]` 
 ,
 *optional* 
 ) —
The weights to use for the text encoder. If
 `None` 
 , the weights are set to
 `1.0` 
 for all the adapters.


 Sets the adapter layers for the text encoder.
 




#### 




 set\_lora\_device




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L2564)



 (
 


 adapter\_names
 
 : typing.List[str]
 




 device
 
 : typing.Union[torch.device, str, int]
 



 )
 


 Parameters
 




* **adapter\_names** 
 (
 `List[str]` 
 ) —
List of adapters to send device to.
* **device** 
 (
 `Union[torch.device, str, int]` 
 ) —
Device to send the adapters to. Can be either a torch device, a str or an integer.


 Moves the LoRAs listed in
 `adapter_names` 
 to a target device. Useful for offloading the LoRA to the CPU in case
you want to load multiple adapters and free some GPU memory.
 




#### 




 unfuse\_lora




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L2335)



 (
 


 unfuse\_unet
 
 : bool = True
 




 unfuse\_text\_encoder
 
 : bool = True
 



 )
 


 Parameters
 




* **unfuse\_unet** 
 (
 `bool` 
 , defaults to
 `True` 
 ) — Whether to unfuse the UNet LoRA parameters.
* **unfuse\_text\_encoder** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to unfuse the text encoder LoRA parameters. If the text encoder wasn’t monkey-patched with the
LoRA parameters then it won’t have any effect.


 Reverses the effect of
 [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.fuse_lora)
.
 




 This is an experimental API.
 


#### 




 unload\_lora\_weights




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L2234)



 (
 

 )
 




 Unloads the LoRA parameters.
 


 Examples:
 



```
>>> # Assuming `pipeline` is already loaded with the LoRA parameters.
>>> pipeline.unload_lora_weights()
>>> ...
```


## FromSingleFileMixin




### 




 class
 

 diffusers.loaders.
 

 FromSingleFileMixin




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L2604)



 (
 

 )
 




 Load model weights saved in the
 `.ckpt` 
 format into a
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
.
 



#### 




 from\_single\_file




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L2615)



 (
 


 pretrained\_model\_link\_or\_path
 


 \*\*kwargs
 




 )
 


 Parameters
 




* **pretrained\_model\_link\_or\_path** 
 (
 `str` 
 or
 `os.PathLike` 
 ,
 *optional* 
 ) —
Can be either:
 
	+ A link to the
	 `.ckpt` 
	 file (for example
	 `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"` 
	 ) on the Hub.
	+ A path to a
	 *file* 
	 containing all pipeline weights.
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
 and load the model with another dtype. If
 `"auto"` 
 is passed, the
dtype is automatically derived from the model’s weights.
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
* **extract\_ema** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether to extract the EMA weights or not. Pass
 `True` 
 to extract the EMA weights which usually yield
higher quality images for inference. Non-EMA weights are usually better for continuing finetuning.
* **upcast\_attention** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Whether the attention computation should always be upcasted.
* **image\_size** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 512) —
The image size the model was trained on. Use 512 for all Stable Diffusion v1 models and the Stable
Diffusion v2 base model. Use 768 for Stable Diffusion v2.
* **prediction\_type** 
 (
 `str` 
 ,
 *optional* 
 ) —
The prediction type the model was trained on. Use
 `'epsilon'` 
 for all Stable Diffusion v1 models and
the Stable Diffusion v2 base model. Use
 `'v_prediction'` 
 for Stable Diffusion v2.
* **num\_in\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The number of input channels. If
 `None` 
 , it is automatically inferred.
* **scheduler\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pndm"` 
 ) —
Type of scheduler to use. Should be one of
 `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm", "ddim"]` 
.
* **load\_safety\_checker** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether to load the safety checker or not.
* **text\_encoder** 
 (
 [CLIPTextModel](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPTextModel) 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
An instance of
 `CLIPTextModel` 
 to use, specifically the
 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 
 variant. If this
parameter is
 `None` 
 , the function loads a new instance of
 `CLIPTextModel` 
 by itself if needed.
* **vae** 
 (
 `AutoencoderKL` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations. If
this parameter is
 `None` 
 , the function will load a new instance of [CLIP] by itself, if needed.
* **tokenizer** 
 (
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPTokenizer) 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
An instance of
 `CLIPTokenizer` 
 to use. If this parameter is
 `None` 
 , the function loads a new instance
of
 `CLIPTokenizer` 
 by itself if needed.
* **original\_config\_file** 
 (
 `str` 
 ) —
Path to
 `.yaml` 
 config file corresponding to the original architecture. If
 `None` 
 , will be
automatically inferred by looking for a key that only exists in SD2.0 models.
* **kwargs** 
 (remaining dictionary of keyword arguments,
 *optional* 
 ) —
Can be used to overwrite load and saveable variables (for example the pipeline components of the
specific pipeline class). The overwritten components are directly passed to the pipelines
 `__init__` 
 method. See example below for more information.


 Instantiate a
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
 from pretrained pipeline weights saved in the
 `.ckpt` 
 or
 `.safetensors` 
 format. The pipeline is set in evaluation mode (
 `model.eval()` 
 ) by default.
 


 Examples:
 



```
>>> from diffusers import StableDiffusionPipeline

>>> # Download pipeline from huggingface.co and cache.
>>> pipeline = StableDiffusionPipeline.from_single_file(
...     "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
... )

>>> # Download pipeline from local file
>>> # file is downloaded under ./v1-5-pruned-emaonly.ckpt
>>> pipeline = StableDiffusionPipeline.from_single_file("./v1-5-pruned-emaonly")

>>> # Enable float16 and move to GPU
>>> pipeline = StableDiffusionPipeline.from_single_file(
...     "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt",
...     torch_dtype=torch.float16,
... )
>>> pipeline.to("cuda")
```


## FromOriginalControlnetMixin




### 




 class
 

 diffusers.loaders.
 

 FromOriginalControlnetMixin




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L3043)



 (
 

 )
 




#### 




 from\_single\_file




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L3044)



 (
 


 pretrained\_model\_link\_or\_path
 


 \*\*kwargs
 




 )
 


 Parameters
 




* **pretrained\_model\_link\_or\_path** 
 (
 `str` 
 or
 `os.PathLike` 
 ,
 *optional* 
 ) —
Can be either:
 
	+ A link to the
	 `.ckpt` 
	 file (for example
	 `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"` 
	 ) on the Hub.
	+ A path to a
	 *file* 
	 containing all pipeline weights.
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
 and load the model with another dtype. If
 `"auto"` 
 is passed, the
dtype is automatically derived from the model’s weights.
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
* **local\_files\_only** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether to only load local model weights and configuration files or not. If set to True, the model
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
* **image\_size** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 512) —
The image size the model was trained on. Use 512 for all Stable Diffusion v1 models and the Stable
Diffusion v2 base model. Use 768 for Stable Diffusion v2.
* **upcast\_attention** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Whether the attention computation should always be upcasted.
* **kwargs** 
 (remaining dictionary of keyword arguments,
 *optional* 
 ) —
Can be used to overwrite load and saveable variables (for example the pipeline components of the
specific pipeline class). The overwritten components are directly passed to the pipelines
 `__init__` 
 method. See example below for more information.


 Instantiate a
 [ControlNetModel](/docs/diffusers/v0.23.0/en/api/models/controlnet#diffusers.ControlNetModel) 
 from pretrained controlnet weights saved in the original
 `.ckpt` 
 or
 `.safetensors` 
 format. The pipeline is set in evaluation mode (
 `model.eval()` 
 ) by default.
 


 Examples:
 



```
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

url = "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control\_v11p\_sd15\_canny.pth"  # can also be a local path
model = ControlNetModel.from_single_file(url)

url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.safetensors"  # can also be a local path
pipe = StableDiffusionControlNetPipeline.from_single_file(url, controlnet=controlnet)
```


## FromOriginalVAEMixin




### 




 class
 

 diffusers.loaders.
 

 FromOriginalVAEMixin




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L2851)



 (
 

 )
 




#### 




 from\_single\_file




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/loaders.py#L2852)



 (
 


 pretrained\_model\_link\_or\_path
 


 \*\*kwargs
 




 )
 


 Parameters
 




* **pretrained\_model\_link\_or\_path** 
 (
 `str` 
 or
 `os.PathLike` 
 ,
 *optional* 
 ) —
Can be either:
 
	+ A link to the
	 `.ckpt` 
	 file (for example
	 `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"` 
	 ) on the Hub.
	+ A path to a
	 *file* 
	 containing all pipeline weights.
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
 and load the model with another dtype. If
 `"auto"` 
 is passed, the
dtype is automatically derived from the model’s weights.
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
* **local\_files\_only** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether to only load local model weights and configuration files or not. If set to True, the model
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
* **image\_size** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 512) —
The image size the model was trained on. Use 512 for all Stable Diffusion v1 models and the Stable
Diffusion v2 base model. Use 768 for Stable Diffusion v2.
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
* **upcast\_attention** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Whether the attention computation should always be upcasted.
* **scaling\_factor** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.18215) —
The component-wise standard deviation of the trained latent space computed using the first batch of the
training set. This is used to scale the latent space to have unit variance when training the diffusion
model. The latents are scaled with the formula
 `z = z * scaling_factor` 
 before being passed to the
diffusion model. When decoding, the latents are scaled back to the original scale with the formula:
 `z = 1 /scaling_factor * z` 
. For more details, refer to sections 4.3.2 and D.1 of the
 [High-Resolution
Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) 
 paper.
* **kwargs** 
 (remaining dictionary of keyword arguments,
 *optional* 
 ) —
Can be used to overwrite load and saveable variables (for example the pipeline components of the
specific pipeline class). The overwritten components are directly passed to the pipelines
 `__init__` 
 method. See example below for more information.


 Instantiate a
 [AutoencoderKL](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL) 
 from pretrained controlnet weights saved in the original
 `.ckpt` 
 or
 `.safetensors` 
 format. The pipeline is format. The pipeline is set in evaluation mode (
 `model.eval()` 
 ) by
default.
 




 Make sure to pass both
 `image_size` 
 and
 `scaling_factor` 
 to
 `from_single_file()` 
 if you want to load
a VAE that does accompany a stable diffusion model of v2 or higher or SDXL.
 



 Examples:
 



```
from diffusers import AutoencoderKL

url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be local file
model = AutoencoderKL.from_single_file(url)
```