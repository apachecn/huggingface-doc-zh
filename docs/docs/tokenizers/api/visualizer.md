# Visualizer

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/api/visualizer>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/api/visualizer>


## Annotation




### 




 class
 

 tokenizers.tools.
 

 Annotation




[<
 

 source
 

 >](https://github.com/huggingface/tokenizers/blob/v0.13.4.rc2/src/tokenizers/tools/visualizer.py#L16)



 (
 


 start
 
 : int
 




 end
 
 : int
 




 label
 
 : str
 



 )
 


## EncodingVisualizer




### 




 class
 

 tokenizers.tools.
 

 EncodingVisualizer




[<
 

 source
 

 >](https://github.com/huggingface/tokenizers/blob/v0.13.4.rc2/src/tokenizers/tools/visualizer.py#L67)



 (
 


 tokenizer
 
 : Tokenizer
 




 default\_to\_notebook
 
 : bool = True
 




 annotation\_converter
 
 : typing.Union[typing.Callable[[typing.Any], tokenizers.tools.visualizer.Annotation], NoneType] = None
 



 )
 


 Parameters
 




* **tokenizer** 
 (
 [Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 
 ) —
A tokenizer instance
* **default\_to\_notebook** 
 (
 `bool` 
 ) —
Whether to render html output in a notebook by default
* **annotation\_converter** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
An optional (lambda) function that takes an annotation in any format and returns
an Annotation object


 Build an EncodingVisualizer
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/tokenizers/blob/v0.13.4.rc2/src/tokenizers/tools/visualizer.py#L108)



 (
 


 text
 
 : str
 




 annotations
 
 : typing.List[tokenizers.tools.visualizer.Annotation] = []
 




 default\_to\_notebook
 
 : typing.Optional[bool] = None
 



 )
 


 Parameters
 




* **text** 
 (
 `str` 
 ) —
The text to tokenize
* **annotations** 
 (
 `List[Annotation]` 
 ,
 *optional* 
 ) —
An optional list of annotations of the text. The can either be an annotation class
or anything else if you instantiated the visualizer with a converter function
* **default\_to\_notebook** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 *False* 
 ) —
If True, will render the html in a notebook. Otherwise returns an html string.


 Build a visualization of the given text