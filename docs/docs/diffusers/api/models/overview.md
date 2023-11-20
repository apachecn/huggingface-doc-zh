# Models

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/models/overview>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/models/overview>



 🤗 Diffusers provides pretrained models for popular algorithms and modules to create custom diffusion systems. The primary function of models is to denoise an input sample as modeled by the distribution
 




 p
 

 θ
 


 (
 


 x
 


 t
 

 −
 

 1
 



 ∣
 


 x
 

 t
 


 )
 


 p\_{	heta}(x\_{t-1}|x\_{t})
 



 p
 




 θ
 


 ​
 


 (
 


 x
 




 t
 

 −
 

 1
 


 ​
 


 ∣
 


 x
 




 t
 


 ​
 


 )
 




.
 



 All models are built from the base
 [ModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin) 
 class which is a
 [`torch.nn.module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
 providing basic functionality for saving and loading models, locally and from the Hugging Face Hub.
 


## ModelMixin




### 




 class
 

 diffusers.
 

 ModelMixin




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_utils.py#L180)



 (
 

 )
 




 Base class for all models.
 



[ModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin) 
 takes care of storing the model configuration and provides methods for loading, downloading and
saving models.
 


* **config\_name** 
 (
 `str` 
 ) — Filename to save a model to when calling
 [save\_pretrained()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin.save_pretrained) 
.



#### 




 active\_adapters




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_utils.py#L428)



 (
 

 )
 




 Gets the current list of active adapters of the model.
 



 If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
official documentation:
 <https://huggingface.co/docs/peft>


#### 




 add\_adapter




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_utils.py#L299)



 (
 


 adapter\_config
 


 adapter\_name
 
 : str = 'default'
 



 )
 


 Parameters
 




* **adapter\_config** 
 (
 `[~peft.PeftConfig]` 
 ) —
The configuration of the adapter to add; supported adapters are non-prefix tuning and adaption prompt
methods.
* **adapter\_name** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"default"` 
 ) —
The name of the adapter to add. If no name is passed, a default name is assigned to the adapter.


 Adds a new adapter to the current model for training. If no adapter name is passed, a default name is assigned
to the adapter to follow the convention of the PEFT library.
 



 If you are not familiar with adapters and PEFT methods, we invite you to read more about them in the PEFT
 [documentation](https://huggingface.co/docs/peft) 
.
 




#### 




 disable\_adapters




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_utils.py#L383)



 (
 

 )
 




 Disable all adapters attached to the model and fallback to inference with the base model only.
 



 If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
official documentation:
 <https://huggingface.co/docs/peft>


#### 




 disable\_gradient\_checkpointing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_utils.py#L232)



 (
 

 )
 




 Deactivates gradient checkpointing for the current model (may be referred to as
 *activation checkpointing* 
 or
 *checkpoint activations* 
 in other frameworks).
 




#### 




 disable\_xformers\_memory\_efficient\_attention




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_utils.py#L293)



 (
 

 )
 




 Disable memory efficient attention from
 [xFormers](https://facebookresearch.github.io/xformers/) 
.
 




#### 




 enable\_adapters




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_utils.py#L405)



 (
 

 )
 




 Enable adapters that are attached to the model. The model will use
 `self.active_adapters()` 
 to retrieve the
list of adapters to enable.
 



 If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
official documentation:
 <https://huggingface.co/docs/peft>


#### 




 enable\_gradient\_checkpointing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_utils.py#L223)



 (
 

 )
 




 Activates gradient checkpointing for the current model (may be referred to as
 *activation checkpointing* 
 or
 *checkpoint activations* 
 in other frameworks).
 




#### 




 enable\_xformers\_memory\_efficient\_attention




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_utils.py#L257)



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
.
 



 When this option is enabled, you should observe lower GPU memory usage and a potential speed up during
inference. Speed up during training is not guaranteed.
 




 ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient attention takes
precedent.
 



 Examples:
 



```
>>> import torch
>>> from diffusers import UNet2DConditionModel
>>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

>>> model = UNet2DConditionModel.from_pretrained(
...     "stabilityai/stable-diffusion-2-1", subfolder="unet", torch_dtype=torch.float16
... )
>>> model = model.to("cuda")
>>> model.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
```


#### 




 from\_pretrained




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_utils.py#L529)



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
	 [save\_pretrained()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin.save_pretrained) 
	.
* **cache\_dir** 
 (
 `Union[str, os.PathLike]` 
 ,
 *optional* 
 ) —
Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
is not used.
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
* **output\_loading\_info** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(
 `bool` 
 ,** 
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
* **from\_flax** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Load the model weights from a Flax checkpoint save file.
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
The path to offload weights if
 `device_map` 
 contains the value
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
* **variant** 
 (
 `str` 
 ,
 *optional* 
 ) —
Load weights from a specified
 `variant` 
 filename such as
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
 , the
 `safetensors` 
 weights are downloaded if they’re available
 **and** 
 if the
 `safetensors` 
 library is installed. If set to
 `True` 
 , the model is forcibly loaded from
 `safetensors` 
 weights. If set to
 `False` 
 ,
 `safetensors` 
 weights are not loaded.


 Instantiate a pretrained PyTorch model from a pretrained model configuration.
 



 The model is set in evaluation mode -
 `model.eval()` 
 - by default, and dropout modules are deactivated. To
train the model, set it back in training mode with
 `model.train()` 
.
 




 To use private or
 [gated models](https://huggingface.co/docs/hub/models-gated#gated-models) 
 , log-in with
 `huggingface-cli login` 
. You can also activate the special
 [“offline-mode”](https://huggingface.co/diffusers/installation.html#offline-mode) 
 to use this method in a
firewalled environment.
 



 Example:
 



```
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
```



 If you get the error message below, you need to finetune the weights for your downstream task:
 



```
Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
- conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```


#### 




 num\_parameters




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_utils.py#L1028)



 (
 


 only\_trainable
 
 : bool = False
 




 exclude\_embeddings
 
 : bool = False
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`int` 


 Parameters
 




* **only\_trainable** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether or not to return only the number of trainable parameters.
* **exclude\_embeddings** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether or not to return only the number of non-embedding parameters.




 Returns
 




 export const metadata = 'undefined';
 

`int` 




 export const metadata = 'undefined';
 




 The number of parameters.
 



 Get number of (trainable or non-embedding) parameters in the module.
 


 Example:
 



```
from diffusers import UNet2DConditionModel

model_id = "runwayml/stable-diffusion-v1-5"
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
unet.num_parameters(only_trainable=True)
859520964
```


#### 




 save\_pretrained




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_utils.py#L446)



 (
 


 save\_directory
 
 : typing.Union[str, os.PathLike]
 




 is\_main\_process
 
 : bool = True
 




 save\_function
 
 : typing.Callable = None
 




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
Directory to save a model and its configuration file to. Will be created if it doesn’t exist.
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
Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
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


 Save a model and its configuration file to a directory so that it can be reloaded using the
 [from\_pretrained()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin.from_pretrained) 
 class method.
 




#### 




 set\_adapter




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_utils.py#L334)



 (
 


 adapter\_name
 
 : typing.Union[str, typing.List[str]]
 



 )
 


 Parameters
 




* **adapter\_name** 
 (Union[str, List[str]])) —
The list of adapters to set or the adapter name in case of single adapter.


 Sets a specific adapter by forcing the model to only use that adapter and disables the other adapters.
 



 If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
official documentation:
 <https://huggingface.co/docs/peft>


## FlaxModelMixin




### 




 class
 

 diffusers.
 

 FlaxModelMixin




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_flax_utils.py#L46)



 (
 

 )
 




 Base class for all Flax models.
 



[FlaxModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.FlaxModelMixin) 
 takes care of storing the model configuration and provides methods for loading, downloading and
saving models.
 


* **config\_name** 
 (
 `str` 
 ) — Filename to save a model to when calling
 [save\_pretrained()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.FlaxModelMixin.save_pretrained) 
.



#### 




 from\_pretrained




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_flax_utils.py#L198)



 (
 


 pretrained\_model\_name\_or\_path
 
 : typing.Union[str, os.PathLike]
 




 dtype
 
 : dtype = <class 'jax.numpy.float32'>
 




 \*model\_args
 


 \*\*kwargs
 




 )
 


 Parameters
 




* **pretrained\_model\_name\_or\_path** 
 (
 `str` 
 or
 `os.PathLike` 
 ) —
Can be either:
 
	+ A string, the
	 *model id* 
	 (for example
	 `runwayml/stable-diffusion-v1-5` 
	 ) of a pretrained model
	hosted on the Hub.
	+ A path to a
	 *directory* 
	 (for example
	 `./my_model_directory` 
	 ) containing the model weights saved
	using
	 [save\_pretrained()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.FlaxModelMixin.save_pretrained) 
	.
* **dtype** 
 (
 `jax.numpy.dtype` 
 ,
 *optional* 
 , defaults to
 `jax.numpy.float32` 
 ) —
The data type of the computation. Can be one of
 `jax.numpy.float32` 
 ,
 `jax.numpy.float16` 
 (on GPUs) and
 `jax.numpy.bfloat16` 
 (on TPUs).
 
 This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified, all the computation will be performed with the given
 `dtype` 
.
 




 This only specifies the dtype of the
 *computation* 
 and does not influence the dtype of model
parameters.
 



 If you wish to change the dtype of the model parameters, see
 [to\_fp16()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.FlaxModelMixin.to_fp16) 
 and
 [to\_bf16()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.FlaxModelMixin.to_bf16) 
.
* **model\_args** 
 (sequence of positional arguments,
 *optional* 
 ) —
All remaining positional arguments are passed to the underlying model’s
 `__init__` 
 method.
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
* **local\_files\_only(
 `bool` 
 ,** 
*optional* 
 , defaults to
 `False` 
 ) —
Whether to only load local model weights and configuration files or not. If set to
 `True` 
 , the model
won’t be downloaded from the Hub.
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
* **from\_pt** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Load the model weights from a PyTorch checkpoint save file.
* **kwargs** 
 (remaining dictionary of keyword arguments,
 *optional* 
 ) —
Can be used to update the configuration object (after it is loaded) and initiate the model (for
example,
 `output_attentions=True` 
 ). Behaves differently depending on whether a
 `config` 
 is provided or
automatically loaded:
 
	+ If a configuration is provided with
	 `config` 
	 ,
	 `kwargs` 
	 are directly passed to the underlying
	model’s
	 `__init__` 
	 method (we assume all relevant updates to the configuration have already been
	done).
	+ If a configuration is not provided,
	 `kwargs` 
	 are first passed to the configuration class
	initialization function
	 [from\_config()](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin.from_config) 
	. Each key of the
	 `kwargs` 
	 that corresponds
	to a configuration attribute is used to override said attribute with the supplied
	 `kwargs` 
	 value.
	Remaining keys that do not correspond to any configuration attribute are passed to the underlying
	model’s
	 `__init__` 
	 function.


 Instantiate a pretrained Flax model from a pretrained model configuration.
 


 Examples:
 



```
>>> from diffusers import FlaxUNet2DConditionModel

>>> # Download model and configuration from huggingface.co and cache.
>>> model, params = FlaxUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
>>> # Model was saved using \*save\_pretrained('./test/saved\_model/')\* (for example purposes, not runnable).
>>> model, params = FlaxUNet2DConditionModel.from_pretrained("./test/saved\_model/")
```



 If you get the error message below, you need to finetune the weights for your downstream task:
 



```
Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
- conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```


#### 




 save\_pretrained




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_flax_utils.py#L496)



 (
 


 save\_directory
 
 : typing.Union[str, os.PathLike]
 




 params
 
 : typing.Union[typing.Dict, flax.core.frozen\_dict.FrozenDict]
 




 is\_main\_process
 
 : bool = True
 




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
Directory to save a model and its configuration file to. Will be created if it doesn’t exist.
* **params** 
 (
 `Union[Dict, FrozenDict]` 
 ) —
A
 `PyTree` 
 of model parameters.
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
Additional key word arguments passed along to the
 [push\_to\_hub()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub) 
 method.


 Save a model and its configuration file to a directory so that it can be reloaded using the
 [from\_pretrained()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.FlaxModelMixin.from_pretrained) 
 class method.
 




#### 




 to\_bf16




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_flax_utils.py#L90)



 (
 


 params
 
 : typing.Union[typing.Dict, flax.core.frozen\_dict.FrozenDict]
 




 mask
 
 : typing.Any = None
 



 )
 


 Parameters
 




* **params** 
 (
 `Union[Dict, FrozenDict]` 
 ) —
A
 `PyTree` 
 of model parameters.
* **mask** 
 (
 `Union[Dict, FrozenDict]` 
 ) —
A
 `PyTree` 
 with same structure as the
 `params` 
 tree. The leaves should be booleans. It should be
 `True` 
 for params you want to cast, and
 `False` 
 for those you want to skip.


 Cast the floating-point
 `params` 
 to
 `jax.numpy.bfloat16` 
. This returns a new
 `params` 
 tree and does not cast
the
 `params` 
 in place.
 



 This method can be used on a TPU to explicitly convert the model parameters to bfloat16 precision to do full
half-precision training or to save weights in bfloat16 for inference in order to save memory and improve speed.
 


 Examples:
 



```
>>> from diffusers import FlaxUNet2DConditionModel

>>> # load model
>>> model, params = FlaxUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
>>> # By default, the model parameters will be in fp32 precision, to cast these to bfloat16 precision
>>> params = model.to_bf16(params)
>>> # If you don't want to cast certain parameters (for example layer norm bias and scale)
>>> # then pass the mask as follows
>>> from flax import traverse_util

>>> model, params = FlaxUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
>>> flat_params = traverse_util.flatten_dict(params)
>>> mask = {
...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
...     for path in flat_params
... }
>>> mask = traverse_util.unflatten_dict(mask)
>>> params = model.to_bf16(params, mask)
```


#### 




 to\_fp16




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_flax_utils.py#L156)



 (
 


 params
 
 : typing.Union[typing.Dict, flax.core.frozen\_dict.FrozenDict]
 




 mask
 
 : typing.Any = None
 



 )
 


 Parameters
 




* **params** 
 (
 `Union[Dict, FrozenDict]` 
 ) —
A
 `PyTree` 
 of model parameters.
* **mask** 
 (
 `Union[Dict, FrozenDict]` 
 ) —
A
 `PyTree` 
 with same structure as the
 `params` 
 tree. The leaves should be booleans. It should be
 `True` 
 for params you want to cast, and
 `False` 
 for those you want to skip.


 Cast the floating-point
 `params` 
 to
 `jax.numpy.float16` 
. This returns a new
 `params` 
 tree and does not cast the
 `params` 
 in place.
 



 This method can be used on a GPU to explicitly convert the model parameters to float16 precision to do full
half-precision training or to save weights in float16 for inference in order to save memory and improve speed.
 


 Examples:
 



```
>>> from diffusers import FlaxUNet2DConditionModel

>>> # load model
>>> model, params = FlaxUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
>>> # By default, the model params will be in fp32, to cast these to float16
>>> params = model.to_fp16(params)
>>> # If you want don't want to cast certain parameters (for example layer norm bias and scale)
>>> # then pass the mask as follows
>>> from flax import traverse_util

>>> model, params = FlaxUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
>>> flat_params = traverse_util.flatten_dict(params)
>>> mask = {
...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
...     for path in flat_params
... }
>>> mask = traverse_util.unflatten_dict(mask)
>>> params = model.to_fp16(params, mask)
```


#### 




 to\_fp32




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/modeling_flax_utils.py#L129)



 (
 


 params
 
 : typing.Union[typing.Dict, flax.core.frozen\_dict.FrozenDict]
 




 mask
 
 : typing.Any = None
 



 )
 


 Parameters
 




* **params** 
 (
 `Union[Dict, FrozenDict]` 
 ) —
A
 `PyTree` 
 of model parameters.
* **mask** 
 (
 `Union[Dict, FrozenDict]` 
 ) —
A
 `PyTree` 
 with same structure as the
 `params` 
 tree. The leaves should be booleans. It should be
 `True` 
 for params you want to cast, and
 `False` 
 for those you want to skip.


 Cast the floating-point
 `params` 
 to
 `jax.numpy.float32` 
. This method can be used to explicitly convert the
model parameters to fp32 precision. This returns a new
 `params` 
 tree and does not cast the
 `params` 
 in place.
 


 Examples:
 



```
>>> from diffusers import FlaxUNet2DConditionModel

>>> # Download model and configuration from huggingface.co
>>> model, params = FlaxUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
>>> # By default, the model params will be in fp32, to illustrate the use of this method,
>>> # we'll first cast to fp16 and back to fp32
>>> params = model.to_f16(params)
>>> # now cast back to fp32
>>> params = model.to_fp32(params)
```


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