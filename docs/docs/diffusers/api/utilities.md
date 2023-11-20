# Utilities

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/utilities>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/utilities>



 Utility and helper functions for working with 🤗 Diffusers.
 


## numpy\_to\_pil




#### 




 diffusers.utils.numpy\_to\_pil




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/utils/pil_utils.py#L37)



 (
 


 images
 




 )
 




 Convert a numpy image or a batch of images to a PIL image.
 


## pt\_to\_pil




#### 




 diffusers.utils.pt\_to\_pil




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/utils/pil_utils.py#L27)



 (
 


 images
 




 )
 




 Convert a torch image to a PIL image.
 


## load\_image




#### 




 diffusers.utils.load\_image




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/utils/loading_utils.py#L9)



 (
 


 image
 
 : typing.Union[str, PIL.Image.Image]
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`PIL.Image.Image` 


 Parameters
 




* **image** 
 (
 `str` 
 or
 `PIL.Image.Image` 
 ) —
The image to convert to the PIL Image format.




 Returns
 




 export const metadata = 'undefined';
 

`PIL.Image.Image` 




 export const metadata = 'undefined';
 




 A PIL Image.
 



 Loads
 `image` 
 to a PIL Image.
 


## export\_to\_gif




#### 




 diffusers.utils.export\_to\_gif




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/utils/export_utils.py#L31)



 (
 


 image
 
 : typing.List[PIL.Image.Image]
 




 output\_gif\_path
 
 : str = None
 



 )
 


## export\_to\_video




#### 




 diffusers.utils.export\_to\_video




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/utils/export_utils.py#L118)



 (
 


 video\_frames
 
 : typing.List[numpy.ndarray]
 




 output\_video\_path
 
 : str = None
 



 )
 


## make\_image\_grid




#### 




 diffusers.utils.make\_image\_grid




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/utils/pil_utils.py#L53)



 (
 


 images
 
 : typing.List[PIL.Image.Image]
 




 rows
 
 : int
 




 cols
 
 : int
 




 resize
 
 : int = None
 



 )
 




 Prepares a single grid of images. Useful for visualization purposes.