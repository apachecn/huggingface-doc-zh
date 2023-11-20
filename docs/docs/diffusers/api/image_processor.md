# VAE Image Processor

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/image_processor>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/image_processor>



 The
 `VaeImageProcessor` 
 provides a unified API for
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline) 
 ’s to prepare image inputs for VAE encoding and post-processing outputs once they’re decoded. This includes transformations such as resizing, normalization, and conversion between PIL Image, PyTorch, and NumPy arrays.
 



 All pipelines with
 `VaeImageProcessor` 
 accepts PIL Image, PyTorch tensor, or NumPy arrays as image inputs and returns outputs based on the
 `output_type` 
 argument by the user. You can pass encoded image latents directly to the pipeline and return latents from the pipeline as a specific output with the
 `output_type` 
 argument (for example
 `output_type="pt"` 
 ). This allows you to take the generated latents from one pipeline and pass it to another pipeline as input without leaving the latent space. It also makes it much easier to use multiple pipelines together by passing PyTorch tensors directly between different pipelines.
 


## VaeImageProcessor




### 




 class
 

 diffusers.image\_processor.
 

 VaeImageProcessor




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L37)



 (
 


 do\_resize
 
 : bool = True
 




 vae\_scale\_factor
 
 : int = 8
 




 resample
 
 : str = 'lanczos'
 




 do\_normalize
 
 : bool = True
 




 do\_binarize
 
 : bool = False
 




 do\_convert\_rgb
 
 : bool = False
 




 do\_convert\_grayscale
 
 : bool = False
 



 )
 


 Parameters
 




* **do\_resize** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether to downscale the image’s (height, width) dimensions to multiples of
 `vae_scale_factor` 
. Can accept
 `height` 
 and
 `width` 
 arguments from
 [image\_processor.VaeImageProcessor.preprocess()](/docs/diffusers/v0.23.0/en/api/image_processor#diffusers.image_processor.VaeImageProcessor.preprocess) 
 method.
* **vae\_scale\_factor** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `8` 
 ) —
VAE scale factor. If
 `do_resize` 
 is
 `True` 
 , the image is automatically resized to multiples of this factor.
* **resample** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `lanczos` 
 ) —
Resampling filter to use when resizing the image.
* **do\_normalize** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether to normalize the image to [-1,1].
* **do\_binarize** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether to binarize the image to 0/1.
* **do\_convert\_rgb** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to be
 `False` 
 ) —
Whether to convert the images to RGB format.
* **do\_convert\_grayscale** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to be
 `False` 
 ) —
Whether to convert the images to grayscale format.


 Image processor for VAE.
 



#### 




 binarize




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L228)



 (
 


 image
 
 : Image
 



 )
 




 create a mask
 




#### 




 convert\_to\_grayscale




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L151)



 (
 


 image
 
 : Image
 



 )
 




 Converts a PIL image to grayscale format.
 




#### 




 convert\_to\_rgb




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L142)



 (
 


 image
 
 : Image
 



 )
 




 Converts a PIL image to RGB format.
 




#### 




 denormalize




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L135)



 (
 


 images
 




 )
 




 Denormalize an image array to [0,1].
 




#### 




 get\_default\_height\_width




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L160)



 (
 


 image
 
 : [<class 'PIL.Image.Image'>, <class 'numpy.ndarray'>, <class 'torch.Tensor'>]
 




 height
 
 : typing.Optional[int] = None
 




 width
 
 : typing.Optional[int] = None
 



 )
 


 Parameters
 




* **image(
 `PIL.Image.Image` 
 ,** 
`np.ndarray` 
 or
 `torch.Tensor` 
 ) —
The image input, can be a PIL image, numpy array or pytorch tensor. if it is a numpy array, should have
shape
 `[batch, height, width]` 
 or
 `[batch, height, width, channel]` 
 if it is a pytorch tensor, should
have shape
 `[batch, channel, height, width]` 
.
* **height** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The height in preprocessed image. If
 `None` 
 , will use the height of
 `image` 
 input.
* **width** 
 (
 `int` 
 ,
 *optional* 
`, defaults to` 
 None
 `) -- The width in preprocessed. If` 
 None
 `, will use the width of the` 
 image` input.


 This function return the height and width that are downscaled to the next integer multiple of
 `vae_scale_factor` 
.
 




#### 




 normalize




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L128)



 (
 


 images
 




 )
 




 Normalize an image array to [-1,1].
 




#### 




 numpy\_to\_pil




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L81)



 (
 


 images
 
 : ndarray
 



 )
 




 Convert a numpy image or a batch of images to a PIL image.
 




#### 




 numpy\_to\_pt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L109)



 (
 


 images
 
 : ndarray
 



 )
 




 Convert a NumPy image to a PyTorch tensor.
 




#### 




 pil\_to\_numpy




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L97)



 (
 


 images
 
 : typing.Union[typing.List[PIL.Image.Image], PIL.Image.Image]
 



 )
 




 Convert a PIL image or a list of PIL images to NumPy arrays.
 




#### 




 preprocess




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L236)



 (
 


 image
 
 : typing.Union[torch.FloatTensor, PIL.Image.Image, numpy.ndarray]
 




 height
 
 : typing.Optional[int] = None
 




 width
 
 : typing.Optional[int] = None
 



 )
 




 Preprocess the image input. Accepted formats are PIL images, NumPy arrays or PyTorch tensors.
 




#### 




 pt\_to\_numpy




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L120)



 (
 


 images
 
 : FloatTensor
 



 )
 




 Convert a PyTorch tensor to a NumPy image.
 




#### 




 resize




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L203)



 (
 


 image
 
 : [<class 'PIL.Image.Image'>, <class 'numpy.ndarray'>, <class 'torch.Tensor'>]
 




 height
 
 : typing.Optional[int] = None
 




 width
 
 : typing.Optional[int] = None
 



 )
 




 Resize image.
 


## VaeImageProcessorLDM3D




 The
 `VaeImageProcessorLDM3D` 
 accepts RGB and depth inputs and returns RGB and depth outputs.
 



### 




 class
 

 diffusers.image\_processor.
 

 VaeImageProcessorLDM3D




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L365)



 (
 


 do\_resize
 
 : bool = True
 




 vae\_scale\_factor
 
 : int = 8
 




 resample
 
 : str = 'lanczos'
 




 do\_normalize
 
 : bool = True
 



 )
 


 Parameters
 




* **do\_resize** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether to downscale the image’s (height, width) dimensions to multiples of
 `vae_scale_factor` 
.
* **vae\_scale\_factor** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `8` 
 ) —
VAE scale factor. If
 `do_resize` 
 is
 `True` 
 , the image is automatically resized to multiples of this factor.
* **resample** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `lanczos` 
 ) —
Resampling filter to use when resizing the image.
* **do\_normalize** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether to normalize the image to [-1,1].


 Image processor for VAE LDM3D.
 



#### 




 numpy\_to\_depth




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L419)



 (
 


 images
 




 )
 




 Convert a NumPy depth image or a batch of images to a PIL image.
 




#### 




 numpy\_to\_pil




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L392)



 (
 


 images
 




 )
 




 Convert a NumPy image or a batch of images to a PIL image.
 




#### 




 rgblike\_to\_depthmap




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/image_processor.py#L408)



 (
 


 image
 




 )
 




 Returns: depth map