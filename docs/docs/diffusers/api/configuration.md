# Configuration

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/configuration>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/configuration>



 Schedulers from
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 and models from
 [ModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin) 
 inherit from
 [ConfigMixin](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin) 
 which stores all the parameters that are passed to their respective
 `__init__` 
 methods in a JSON-configuration file.
 




 To use private or
 [gated](https://huggingface.co/docs/hub/models-gated#gated-models) 
 models, log-in with
 `huggingface-cli login` 
.
 


## ConfigMixin




### 




 class
 

 diffusers.
 

 ConfigMixin




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/configuration_utils.py#L82)



 (
 

 )
 




 Base class for all configuration classes. All configuration parameters are stored under
 `self.config` 
. Also
provides the
 [from\_config()](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin.from_config) 
 and
 [save\_config()](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin.save_config) 
 methods for loading, downloading, and
saving classes that inherit from
 [ConfigMixin](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin) 
.
 



 Class attributes:
 


* **config\_name** 
 (
 `str` 
 ) — A filename under which the config should stored when calling
 [save\_config()](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin.save_config) 
 (should be overridden by parent class).
* **ignore\_for\_config** 
 (
 `List[str]` 
 ) — A list of attributes that should not be saved in the config (should be
overridden by subclass).
* **has\_compatibles** 
 (
 `bool` 
 ) — Whether the class has compatible classes (should be overridden by subclass).
* **\_deprecated\_kwargs** 
 (
 `List[str]` 
 ) — Keyword arguments that are deprecated. Note that the
 `init` 
 function
should only have a
 `kwargs` 
 argument if at least one argument is deprecated (should be overridden by
subclass).



#### 




 load\_config




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/configuration_utils.py#L276)



 (
 


 pretrained\_model\_name\_or\_path
 
 : typing.Union[str, os.PathLike]
 




 return\_unused\_kwargs
 
 = False
 




 return\_commit\_hash
 
 = False
 




 \*\*kwargs
 




 )
 

 →
 



 export const metadata = 'undefined';
 

`dict` 


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
	 ) containing model weights saved with
	 [save\_config()](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin.save_config) 
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
* **subfolder** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `""` 
 ) —
The subfolder location of a model file within a larger model repository on the Hub or locally.
* **return\_unused\_kwargs** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to `False) —
Whether unused keyword arguments of the config are returned.
* **return\_commit\_hash** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False) -- Whether the` 
 commit\_hash` of the loaded configuration are returned.




 Returns
 




 export const metadata = 'undefined';
 

`dict` 




 export const metadata = 'undefined';
 




 A dictionary of all the parameters stored in a JSON configuration file.
 



 Load a model or scheduler configuration.
 




#### 




 from\_config




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/configuration_utils.py#L181)



 (
 


 config
 
 : typing.Union[diffusers.configuration\_utils.FrozenDict, typing.Dict[str, typing.Any]] = None
 




 return\_unused\_kwargs
 
 = False
 




 \*\*kwargs
 




 )
 

 →
 



 export const metadata = 'undefined';
 

[ModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin) 
 or
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 


 Parameters
 




* **config** 
 (
 `Dict[str, Any]` 
 ) —
A config dictionary from which the Python class is instantiated. Make sure to only load configuration
files of compatible classes.
* **return\_unused\_kwargs** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether kwargs that are not consumed by the Python class should be returned or not.
* **kwargs** 
 (remaining dictionary of keyword arguments,
 *optional* 
 ) —
Can be used to update the configuration object (after it is loaded) and initiate the Python class.
 `**kwargs` 
 are passed directly to the underlying scheduler/model’s
 `__init__` 
 method and eventually
overwrite the same named arguments in
 `config` 
.




 Returns
 




 export const metadata = 'undefined';
 

[ModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin) 
 or
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 




 export const metadata = 'undefined';
 




 A model or scheduler object instantiated from a config dictionary.
 



 Instantiate a Python class from a config dictionary.
 


 Examples:
 



```
>>> from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler

>>> # Download scheduler from huggingface.co and cache.
>>> scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")

>>> # Instantiate DDIM scheduler class with same config as DDPM
>>> scheduler = DDIMScheduler.from_config(scheduler.config)

>>> # Instantiate PNDM scheduler class with same config as DDPM
>>> scheduler = PNDMScheduler.from_config(scheduler.config)
```


#### 




 save\_config




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/configuration_utils.py#L139)



 (
 


 save\_directory
 
 : typing.Union[str, os.PathLike]
 




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
Directory where the configuration JSON file is saved (will be created if it does not exist).
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


 Save a configuration object to the directory specified in
 `save_directory` 
 so that it can be reloaded using the
 [from\_config()](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin.from_config) 
 class method.
 




#### 




 to\_json\_file




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/configuration_utils.py#L587)



 (
 


 json\_file\_path
 
 : typing.Union[str, os.PathLike]
 



 )
 


 Parameters
 




* **json\_file\_path** 
 (
 `str` 
 or
 `os.PathLike` 
 ) —
Path to the JSON file to save a configuration instance’s parameters.


 Save the configuration instance’s parameters to a JSON file.
 




#### 




 to\_json\_string




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/configuration_utils.py#L561)



 (
 

 )
 

 →
 



 export const metadata = 'undefined';
 

`str` 



 Returns
 




 export const metadata = 'undefined';
 

`str` 




 export const metadata = 'undefined';
 




 String containing all the attributes that make up the configuration instance in JSON format.
 



 Serializes the configuration instance to a JSON string.