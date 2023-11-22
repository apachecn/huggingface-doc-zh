# Outputs

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/outputs>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/outputs>



 All models outputs are subclasses of
 [BaseOutput](/docs/diffusers/v0.23.0/en/api/outputs#diffusers.utils.BaseOutput) 
 , data structures containing all the information returned by the model. The outputs can also be used as tuples or dictionaries.
 



 For example:
 



```
from diffusers import DDIMPipeline

pipeline = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32")
outputs = pipeline()
```



 The
 `outputs` 
 object is a
 [ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 which means it has an image attribute.
 



 You can access each attribute as you normally would or with a keyword lookup, and if that attribute is not returned by the model, you will get
 `None` 
 :
 



```
outputs.images
outputs["images"]
```



 When considering the
 `outputs` 
 object as a tuple, it only considers the attributes that don’t have
 `None` 
 values.
For instance, retrieving an image by indexing into it returns the tuple
 `(outputs.images)` 
 :
 



```
outputs[:1]
```




 To check a specific pipeline or model output, refer to its corresponding API documentation.
 


## BaseOutput




### 




 class
 

 diffusers.utils.
 

 BaseOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/utils/outputs.py#L40)



 (
 

 )
 




 Base class for all model outputs as dataclass. Has a
 `__getitem__` 
 that allows indexing by integer or slice (like a
tuple) or strings (like a dictionary) that will ignore the
 `None` 
 attributes. Otherwise behaves like a regular
Python dictionary.
 




 You can’t unpack a
 `BaseOutput` 
 directly. Use the
 [to\_tuple()](/docs/diffusers/v0.23.0/en/api/outputs#diffusers.utils.BaseOutput.to_tuple) 
 method to convert it to a tuple
first.
 




#### 




 to\_tuple




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/utils/outputs.py#L126)



 (
 

 )
 




 Convert self to a tuple containing all the attributes/keys that are not
 `None` 
.
 


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
 


## FlaxImagePipelineOutput




### 




 class
 

 diffusers.pipelines.pipeline\_flax\_utils.
 

 FlaxImagePipelineOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_flax_utils.py#L88)



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
 



#### 




 replace




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/flax/struct.py#L111)



 (
 


 \*\*updates
 




 )
 




 “Returns a new object replacing the specified fields with new values.
 


## AudioPipelineOutput




### 




 class
 

 diffusers.
 

 AudioPipelineOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L124)



 (
 


 audios
 
 : ndarray
 



 )
 


 Parameters
 




* **audios** 
 (
 `np.ndarray` 
 ) —
List of denoised audio samples of a NumPy array of shape
 `(batch_size, num_channels, sample_rate)` 
.


 Output class for audio pipelines.
 


## ImageTextPipelineOutput




### 




 class
 

 diffusers.
 

 ImageTextPipelineOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py#L34)



 (
 


 images
 
 : typing.Union[typing.List[PIL.Image.Image], numpy.ndarray, NoneType]
 




 text
 
 : typing.Union[typing.List[str], typing.List[typing.List[str]], NoneType]
 



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
* **text** 
 (
 `List[str]` 
 or
 `List[List[str]]` 
 ) —
List of generated text strings of length
 `batch_size` 
 or a list of list of strings whose outer list has
length
 `batch_size` 
.


 Output class for joint image-text pipelines.