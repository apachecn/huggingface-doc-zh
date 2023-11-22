# AutoPipeline

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/auto_pipeline>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/auto_pipeline>



`AutoPipeline` 
 is designed to:
 


1. make it easy for you to load a checkpoint for a task without knowing the specific pipeline class to use
2. use multiple pipelines in your workflow



 Based on the task, the
 `AutoPipeline` 
 class automatically retrieves the relevant pipeline given the name or path to the pretrained weights with the
 `from_pretrained()` 
 method.
 



 To seamlessly switch between tasks with the same checkpoint without reallocating additional memory, use the
 `from_pipe()` 
 method to transfer the components from the original pipeline to the new one.
 



```
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

image = pipeline(prompt, num_inference_steps=25).images[0]
```




 Check out the
 [AutoPipeline](/tutorials/autopipeline) 
 tutorial to learn how to use this API!
 




`AutoPipeline` 
 supports text-to-image, image-to-image, and inpainting for the following diffusion models:
 


* [Stable Diffusion](./stable_diffusion)
* [ControlNet](./controlnet)
* [Stable Diffusion XL (SDXL)](./stable_diffusion/stable_diffusion_xl)
* [DeepFloyd IF](./if)
* [Kandinsky](./kandinsky)
* [Kandinsky 2.2](./kandinsky#kandinsky-22)


## AutoPipelineForText2Image




### 




 class
 

 diffusers.
 

 AutoPipelineForText2Image




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/auto_pipeline.py#L169)



 (
 


 \*args
 


 \*\*kwargs
 




 )
 




[AutoPipelineForText2Image](/docs/diffusers/v0.23.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForText2Image) 
 is a generic pipeline class that instantiates a text-to-image pipeline class. The
specific underlying pipeline class is automatically selected from either the
 [from\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForText2Image.from_pretrained) 
 or
 [from\_pipe()](/docs/diffusers/v0.23.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForText2Image.from_pipe) 
 methods.
 



 This class cannot be instantiated using
 `__init__()` 
 (throws an error).
 



 Class attributes:
 


* **config\_name** 
 (
 `str` 
 ) — The configuration filename that stores the class and module names of all the
diffusion pipeline’s components.



#### 




 from\_pretrained




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/auto_pipeline.py#L193)



 (
 


 pretrained\_model\_or\_path
 


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
	 *repo id* 
	 (for example
	 `CompVis/ldm-text2im-large-256` 
	 ) of a pretrained pipeline
	hosted on the Hub.
	+ A path to a
	 *directory* 
	 (for example
	 `./my_pipeline_directory/` 
	 ) containing pipeline weights
	saved using
	 [save\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.save_pretrained) 
	.
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
 and load the model with another dtype. If “auto” is passed, the
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
* **output\_loading\_info(
 `bool` 
 ,** 
*optional* 
 , defaults to
 `False` 
 ) —
Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
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
* **custom\_revision** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"main"` 
 ) —
The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
 `revision` 
 when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
custom pipeline from GitHub, otherwise it defaults to
 `"main"` 
 when loading from the Hub.
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
The path to offload weights if device\_map contains the value
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
* **kwargs** 
 (remaining dictionary of keyword arguments,
 *optional* 
 ) —
Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
class). The overwritten components are passed directly to the pipelines
 `__init__` 
 method. See example
below for more information.
* **variant** 
 (
 `str` 
 ,
 *optional* 
 ) —
Load weights from a specified variant filename such as
 `"fp16"` 
 or
 `"ema"` 
. This is ignored when
loading
 `from_flax` 
.


 Instantiates a text-to-image Pytorch diffusion pipeline from pretrained pipeline weight.
 



 The from\_pretrained() method takes care of returning the correct pipeline class instance by:
 


1. Detect the pipeline class of the pretrained\_model\_or\_path based on the \_class\_name property of its
config object
2. Find the text-to-image pipeline linked to the pipeline class using pattern matching on pipeline class
name.



 If a
 `controlnet` 
 argument is passed, it will instantiate a
 [StableDiffusionControlNetPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetPipeline) 
 object.
 



 The pipeline is set in evaluation mode (
 `model.eval()` 
 ) by default.
 


 If you get the error message below, you need to finetune the weights for your downstream task:
 



```
Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
- conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```


 To use private or
 [gated](https://huggingface.co/docs/hub/models-gated#gated-models) 
 models, log-in with
 `huggingface-cli login` 
.
 



 Examples:
 



```
>>> from diffusers import AutoPipelineForText2Image

>>> pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5")
>>> image = pipeline(prompt).images[0]
```


#### 




 from\_pipe




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/auto_pipeline.py#L338)



 (
 


 pipeline
 


 \*\*kwargs
 




 )
 


 Parameters
 




* **pipeline** 
 (
 `DiffusionPipeline` 
 ) —
an instantiated
 `DiffusionPipeline` 
 object


 Instantiates a text-to-image Pytorch diffusion pipeline from another instantiated diffusion pipeline class.
 



 The from\_pipe() method takes care of returning the correct pipeline class instance by finding the text-to-image
pipeline linked to the pipeline class using pattern matching on pipeline class name.
 



 All the modules the pipeline contains will be used to initialize the new pipeline without reallocating
additional memoery.
 



 The pipeline is set in evaluation mode (
 `model.eval()` 
 ) by default.
 



```
>>> from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

>>> pipe_i2i = AutoPipelineForImage2Image.from_pretrained(
...     "runwayml/stable-diffusion-v1-5", requires_safety_checker=False
... )

>>> pipe_t2i = AutoPipelineForText2Image.from_pipe(pipe_i2i)
>>> image = pipe_t2i(prompt).images[0]
```


## AutoPipelineForImage2Image




### 




 class
 

 diffusers.
 

 AutoPipelineForImage2Image




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/auto_pipeline.py#L439)



 (
 


 \*args
 


 \*\*kwargs
 




 )
 




[AutoPipelineForImage2Image](/docs/diffusers/v0.23.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForImage2Image) 
 is a generic pipeline class that instantiates an image-to-image pipeline class. The
specific underlying pipeline class is automatically selected from either the
 [from\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForImage2Image.from_pretrained) 
 or
 [from\_pipe()](/docs/diffusers/v0.23.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForImage2Image.from_pipe) 
 methods.
 



 This class cannot be instantiated using
 `__init__()` 
 (throws an error).
 



 Class attributes:
 


* **config\_name** 
 (
 `str` 
 ) — The configuration filename that stores the class and module names of all the
diffusion pipeline’s components.



#### 




 from\_pretrained




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/auto_pipeline.py#L463)



 (
 


 pretrained\_model\_or\_path
 


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
	 *repo id* 
	 (for example
	 `CompVis/ldm-text2im-large-256` 
	 ) of a pretrained pipeline
	hosted on the Hub.
	+ A path to a
	 *directory* 
	 (for example
	 `./my_pipeline_directory/` 
	 ) containing pipeline weights
	saved using
	 [save\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.save_pretrained) 
	.
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
 and load the model with another dtype. If “auto” is passed, the
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
* **output\_loading\_info(
 `bool` 
 ,** 
*optional* 
 , defaults to
 `False` 
 ) —
Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
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
* **custom\_revision** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"main"` 
 ) —
The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
 `revision` 
 when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
custom pipeline from GitHub, otherwise it defaults to
 `"main"` 
 when loading from the Hub.
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
The path to offload weights if device\_map contains the value
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
* **kwargs** 
 (remaining dictionary of keyword arguments,
 *optional* 
 ) —
Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
class). The overwritten components are passed directly to the pipelines
 `__init__` 
 method. See example
below for more information.
* **variant** 
 (
 `str` 
 ,
 *optional* 
 ) —
Load weights from a specified variant filename such as
 `"fp16"` 
 or
 `"ema"` 
. This is ignored when
loading
 `from_flax` 
.


 Instantiates a image-to-image Pytorch diffusion pipeline from pretrained pipeline weight.
 



 The from\_pretrained() method takes care of returning the correct pipeline class instance by:
 


1. Detect the pipeline class of the pretrained\_model\_or\_path based on the \_class\_name property of its
config object
2. Find the image-to-image pipeline linked to the pipeline class using pattern matching on pipeline class
name.



 If a
 `controlnet` 
 argument is passed, it will instantiate a
 [StableDiffusionControlNetImg2ImgPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetImg2ImgPipeline) 
 object.
 



 The pipeline is set in evaluation mode (
 `model.eval()` 
 ) by default.
 


 If you get the error message below, you need to finetune the weights for your downstream task:
 



```
Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
- conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```


 To use private or
 [gated](https://huggingface.co/docs/hub/models-gated#gated-models) 
 models, log-in with
 `huggingface-cli login` 
.
 



 Examples:
 



```
>>> from diffusers import AutoPipelineForImage2Image

>>> pipeline = AutoPipelineForImage2Image.from_pretrained("runwayml/stable-diffusion-v1-5")
>>> image = pipeline(prompt, image).images[0]
```


#### 




 from\_pipe




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/auto_pipeline.py#L609)



 (
 


 pipeline
 


 \*\*kwargs
 




 )
 


 Parameters
 




* **pipeline** 
 (
 `DiffusionPipeline` 
 ) —
an instantiated
 `DiffusionPipeline` 
 object


 Instantiates a image-to-image Pytorch diffusion pipeline from another instantiated diffusion pipeline class.
 



 The from\_pipe() method takes care of returning the correct pipeline class instance by finding the
image-to-image pipeline linked to the pipeline class using pattern matching on pipeline class name.
 



 All the modules the pipeline contains will be used to initialize the new pipeline without reallocating
additional memoery.
 



 The pipeline is set in evaluation mode (
 `model.eval()` 
 ) by default.
 


 Examples:
 



```
>>> from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

>>> pipe_t2i = AutoPipelineForText2Image.from_pretrained(
...     "runwayml/stable-diffusion-v1-5", requires_safety_checker=False
... )

>>> pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i)
>>> image = pipe_i2i(prompt, image).images[0]
```


## AutoPipelineForInpainting




### 




 class
 

 diffusers.
 

 AutoPipelineForInpainting




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/auto_pipeline.py#L714)



 (
 


 \*args
 


 \*\*kwargs
 




 )
 




[AutoPipelineForInpainting](/docs/diffusers/v0.23.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForInpainting) 
 is a generic pipeline class that instantiates an inpainting pipeline class. The
specific underlying pipeline class is automatically selected from either the
 [from\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForInpainting.from_pretrained) 
 or
 [from\_pipe()](/docs/diffusers/v0.23.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForInpainting.from_pipe) 
 methods.
 



 This class cannot be instantiated using
 `__init__()` 
 (throws an error).
 



 Class attributes:
 


* **config\_name** 
 (
 `str` 
 ) — The configuration filename that stores the class and module names of all the
diffusion pipeline’s components.



#### 




 from\_pretrained




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/auto_pipeline.py#L738)



 (
 


 pretrained\_model\_or\_path
 


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
	 *repo id* 
	 (for example
	 `CompVis/ldm-text2im-large-256` 
	 ) of a pretrained pipeline
	hosted on the Hub.
	+ A path to a
	 *directory* 
	 (for example
	 `./my_pipeline_directory/` 
	 ) containing pipeline weights
	saved using
	 [save\_pretrained()](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.save_pretrained) 
	.
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
 and load the model with another dtype. If “auto” is passed, the
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
* **output\_loading\_info(
 `bool` 
 ,** 
*optional* 
 , defaults to
 `False` 
 ) —
Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
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
* **custom\_revision** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"main"` 
 ) —
The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
 `revision` 
 when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
custom pipeline from GitHub, otherwise it defaults to
 `"main"` 
 when loading from the Hub.
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
The path to offload weights if device\_map contains the value
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
* **kwargs** 
 (remaining dictionary of keyword arguments,
 *optional* 
 ) —
Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
class). The overwritten components are passed directly to the pipelines
 `__init__` 
 method. See example
below for more information.
* **variant** 
 (
 `str` 
 ,
 *optional* 
 ) —
Load weights from a specified variant filename such as
 `"fp16"` 
 or
 `"ema"` 
. This is ignored when
loading
 `from_flax` 
.


 Instantiates a inpainting Pytorch diffusion pipeline from pretrained pipeline weight.
 



 The from\_pretrained() method takes care of returning the correct pipeline class instance by:
 


1. Detect the pipeline class of the pretrained\_model\_or\_path based on the \_class\_name property of its
config object
2. Find the inpainting pipeline linked to the pipeline class using pattern matching on pipeline class name.



 If a
 `controlnet` 
 argument is passed, it will instantiate a
 [StableDiffusionControlNetInpaintPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetInpaintPipeline) 
 object.
 



 The pipeline is set in evaluation mode (
 `model.eval()` 
 ) by default.
 


 If you get the error message below, you need to finetune the weights for your downstream task:
 



```
Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
- conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```


 To use private or
 [gated](https://huggingface.co/docs/hub/models-gated#gated-models) 
 models, log-in with
 `huggingface-cli login` 
.
 



 Examples:
 



```
>>> from diffusers import AutoPipelineForInpainting

>>> pipeline = AutoPipelineForInpainting.from_pretrained("runwayml/stable-diffusion-v1-5")
>>> image = pipeline(prompt, image=init_image, mask_image=mask_image).images[0]
```


#### 




 from\_pipe




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/auto_pipeline.py#L883)



 (
 


 pipeline
 


 \*\*kwargs
 




 )
 


 Parameters
 




* **pipeline** 
 (
 `DiffusionPipeline` 
 ) —
an instantiated
 `DiffusionPipeline` 
 object


 Instantiates a inpainting Pytorch diffusion pipeline from another instantiated diffusion pipeline class.
 



 The from\_pipe() method takes care of returning the correct pipeline class instance by finding the inpainting
pipeline linked to the pipeline class using pattern matching on pipeline class name.
 



 All the modules the pipeline class contain will be used to initialize the new pipeline without reallocating
additional memoery.
 



 The pipeline is set in evaluation mode (
 `model.eval()` 
 ) by default.
 


 Examples:
 



```
>>> from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting

>>> pipe_t2i = AutoPipelineForText2Image.from_pretrained(
...     "DeepFloyd/IF-I-XL-v1.0", requires_safety_checker=False
... )

>>> pipe_inpaint = AutoPipelineForInpainting.from_pipe(pipe_t2i)
>>> image = pipe_inpaint(prompt, image=init_image, mask_image=mask_image).images[0]
```