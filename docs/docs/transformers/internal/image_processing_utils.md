# 图像处理器实用程序

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/internal/image_processing_utils>
>
> 原始地址：<https://huggingface.co/docs/transformers/internal/image_processing_utils>


本页列出了图像处理器使用的所有实用函数，主要是功能
用于处理图像的变换。


其中大多数仅在您研究库中图像处理器的代码时才有用。


## 图像转换




#### 


变压器.image\_transforms.center\_crop


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_transforms.py#L402)


（


 图像
 
 ：ndarray


尺寸
 
 : 打字.Tuple[int, int]


数据\_格式
 
 : Typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None


输入数据格式
 
 : Typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None


返回\_numpy
 
 : 打字.Optional[bool] = None


）
 

 →


导出常量元数据='未定义';
 

`np.ndarray`


 参数


* **图像**
 （
 `np.ndarray`
 )—
要裁剪的图像。
* **尺寸**
 （
 `元组[int, int]`
 )—
裁剪图像的目标尺寸。
* **数据\_格式**
 （
 `str`
 或者
 `通道维度`
 ,
 *选修的*
 )—
输出图像的通道尺寸格式。可以是以下之一：
 
+ `“频道优先”`
或者
`ChannelDimension.FIRST`
：(num\_channels, height, width) 格式的图像。
+ `"channels_last"`
或者
`ChannelDimension.LAST`
：（高度，宽度，通道数）格式的图像。
如果未设置，将使用输入图像的推断格式。
* **输入\_数据\_格式**
 （
 `str`
 或者
 `通道维度`
 ,
 *选修的*
 )—
输入图像的通道尺寸格式。可以是以下之一：
 
+ `“频道优先”`
或者
`ChannelDimension.FIRST`
：(num\_channels, height, width) 格式的图像。
+ `"channels_last"`
或者
`ChannelDimension.LAST`
：（高度，宽度，通道数）格式的图像。
如果未设置，将使用输入图像的推断格式。
* **返回\_numpy**
 （
 `布尔`
 ,
 *选修的*
 )—
是否将裁剪后的图像作为 numpy 数组返回。用于向后兼容
之前的 ImageFeatureExtractionMixin 方法。
 
+ 取消设置：将返回与输入图像相同的类型。
+ `真实`
: 将返回一个 numpy 数组。
+ `假`
: 将返回一个
`PIL.Image.Image`
目的。


退货


导出常量元数据='未定义';
 

`np.ndarray`


导出常量元数据='未定义';


裁剪后的图像。


作物
 `图像`
 到指定的
 `尺寸`
 使用中心作物。请注意，如果图像太小而无法裁剪
给定的大小，它将被填充（因此返回的结果将始终是大小
 `尺寸`
 ）。




#### 


Transformers.image\_transforms.center\_to\_corners\_format


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_transforms.py#L532)


（


 bboxes\_center
 
 : 张量类型


）


将边界框从中心格式转换为角格式。


center 格式：包含盒子中心的坐标及其宽度、高度尺寸
（中心\_x，中心\_y，宽​​度，高度）
角格式：包含框左上角和右下角的坐标
（上\_左\_x、上\_左\_y、下\_右\_x、下\_右\_y）




#### 


Transformers.image\_transforms.corners\_to\_center\_format


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_transforms.py#L592)


（


 bboxes\_corners
 
 : 张量类型


）


将边界框从角格式转换为中心格式。


角格式：包含框的左上角和右下角的坐标
（上\_左\_x、上\_左\_y、下\_右\_x、下\_右\_y）
center 格式：包含盒子中心的坐标及其宽度、高度尺寸
（中心\_x，中心\_y，宽​​度，高度）




#### 


Transformers.image\_transforms.id\_to\_rgb


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_transforms.py#L626)


（


 id\_map


）


将唯一 ID 转换为 RGB 颜色。




#### 


Transformers.image\_transforms.normalize


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_transforms.py#L347)


（


 图像
 
 ：ndarray


意思是
 
 : 打字.Union[浮点, 打字.Iterable[浮点]]


标准
 
 : 打字.Union[浮点, 打字.Iterable[浮点]]


数据\_格式
 
 : 打字.可选[transformers.image\_utils.ChannelDimension] =无


输入数据格式
 
 : Typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None


）


 参数


* **图像**
 （
 `np.ndarray`
 )—
要标准化的图像。
* **意思是**
 （
 `浮动`
 或者
 `可迭代[浮点]`
 )—
用于标准化的平均值。
* **标准**
 （
 `浮动`
 或者
 `可迭代[浮点]`
 )—
用于标准化的标准差。
* **数据\_格式**
 （
 `通道维度`
 ,
 *选修的*
 )—
输出图像的通道尺寸格式。如果未设置，将使用从输入推断的格式。
* **输入\_数据\_格式**
 （
 `通道维度`
 ,
 *选修的*
 )—
输入图像的通道维度格式。如果未设置，将使用从输入推断的格式。


 标准化
 `图像`
 使用指定的平均值和标准差
 ‘意思’
 和
 `标准`
 。


图像 =（图像 - 平均值）/标准差




#### 


变压器.image\_transforms.pad


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_transforms.py#L656)


（


 图像
 
 ：ndarray


填充
 
 : 打字.Union[int, 打字.Tuple[int, int], 打字.Iterable[打字.Tuple[int, int]]]


模式
 
 : PaddingMode = <PaddingMode.CONSTANT: '常量'>


常数\_值
 
 : 打字.Union[浮点, 打字.Iterable[浮点]] = 0.0


数据\_格式
 
 : Typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None


输入数据格式
 
 : Typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None


）
 

 →


导出常量元数据='未定义';
 

`np.ndarray`


 参数


* **图像**
 （
 `np.ndarray`
 )—
要填充的图像。
* **填充**
 （
 `int`
 或者
 `元组[int, int]`
 或者
 `可迭代[元组[int, int]]`
 )—
应用于高度轴、宽度轴边缘的填充。可以是以下三种格式之一：
 
+ `((之前高度，之后高度)，(之前宽度，之后宽度))`
每个轴都有独特的焊盘宽度。
+ `((之前、之后),)`
pad 前后的高度和宽度相同。
+ `（垫，）`
或 int 是 before = after = 所有轴的焊盘宽度的快捷方式。
* **模式**
 （
 `填充模式`
 )—
要使用的填充模式。可以是以下之一：
 
+ `“常数”`
：具有恒定值的焊盘。
+ `“反映”`
：具有镜像到第一个和最后一个值的矢量反射的焊盘
沿每个轴的向量。
+ `“复制”`
：沿每个轴复制数组边缘的最后一个值进行填充。
+ `“对称”`
：沿阵列边缘镜像矢量反射的焊盘。
* **常数\_值**
 （
 `浮动`
 或者
 `可迭代[浮点]`
 ,
 *选修的*
 )—
用于填充的值，如果
 `模式`
 是
 `“常数”`
 。
* **数据\_格式**
 （
 `str`
 或者
 `通道维度`
 ,
 *选修的*
 )—
输出图像的通道尺寸格式。可以是以下之一：
 
+ `“频道优先”`
或者
`ChannelDimension.FIRST`
：(num\_channels, height, width) 格式的图像。
+ `"channels_last"`
或者
`ChannelDimension.LAST`
：（高度，宽度，通道数）格式的图像。
如果未设置，将使用与输入图像相同的图像。
* **输入\_数据\_格式**
 （
 `str`
 或者
 `通道维度`
 ,
 *选修的*
 )—
输入图像的通道尺寸格式。可以是以下之一：
 
+ `“频道优先”`
或者
`ChannelDimension.FIRST`
：(num\_channels, height, width) 格式的图像。
+ `"channels_last"`
或者
`ChannelDimension.LAST`
：（高度，宽度，通道数）格式的图像。
如果未设置，将使用输入图像的推断格式。


退货


导出常量元数据='未定义';
 

`np.ndarray`


导出常量元数据='未定义';


填充后的图像。


垫
 `图像`
 具有指定的（高度，宽度）
 `填充`
 和
 `模式`
 。




#### 


Transformers.image\_transforms.rgb\_to\_id


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_transforms.py#L615)


（


 颜色


）


将 RGB 颜色转换为唯一 ID。




#### 


Transformers.image\_transforms.rescale


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_transforms.py#L92)


（


 图像
 
 ：ndarray


规模
 
 ： 漂浮


数据\_格式
 
 : 打字.可选[transformers.image\_utils.ChannelDimension] =无


数据类型
 
 : dtype = <类'numpy.float32'>


输入数据格式
 
 : Typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None


）
 

 →


导出常量元数据='未定义';
 

`np.ndarray`


 参数


* **图像**
 （
 `np.ndarray`
 )—
要重新缩放的图像。
* **规模**
 （
 `浮动`
 )—
用于重新缩放图像的比例。
* **数据\_格式**
 （
 `通道维度`
 ,
 *选修的*
 )—
图像的通道尺寸格式。如果未提供，它将与输入图像相同。
* **数据类型**
 （
 `np.dtype`
 ,
 *选修的*
 ，默认为
 `np.float32`
 )—
输出图像的数据类型。默认为
 `np.float32`
 。用于向后兼容功能
提取器。
* **输入\_数据\_格式**
 （
 `通道维度`
 ,
 *选修的*
 )—
输入图像的通道维度格式。如果未提供，将从输入图像推断。


退货


导出常量元数据='未定义';
 

`np.ndarray`


导出常量元数据='未定义';


重新缩放后的图像。


重新缩放
 `图像`
 经过
 `规模`
 。




#### 


变压器.image\_transforms.resize


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_transforms.py#L276)


（


 图像
 
 ：ndarray


尺寸
 
 : 打字.Tuple[int, int]


重新采样
 
 : PILImageResampling = 无


减少差距
 
 : 打字.Optional[int] = None


数据\_格式
 
 : 打字.可选[transformers.image\_utils.ChannelDimension] =无


返回\_numpy
 
 ：布尔=真


输入数据格式
 
 : Typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None


）
 

 →


导出常量元数据='未定义';
 

`np.ndarray`


 参数


* **图像**
 （
 `np.ndarray`
 )—
要调整大小的图像。
* **尺寸**
 （
 `元组[int, int]`
 )—
用于调整图像大小的大小。
* **重新采样**
 （
 `int`
 ,
 *选修的*
 ，默认为
 `PILImageResampling.BILINEAR`
 )—
过滤器供用户重新采样。
* **减少差距**
 （
 `int`
 ,
 *选修的*
 )—
通过分两步调整图像大小来应用优化。更大的
 `减少间隙`
 ，结果越接近
公平的重采样。有关更多详细信息，请参阅相应的 Pillow 文档。
* **数据\_格式**
 （
 `通道维度`
 ,
 *选修的*
 )—
输出图像的通道尺寸格式。如果未设置，将使用从输入推断的格式。
* **返回\_numpy**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 '真实'
 )—
是否将调整大小的图像作为 numpy 数组返回。如果为假
 `PIL.Image.Image`
 对象是
回。
* **输入\_数据\_格式**
 （
 `通道维度`
 ,
 *选修的*
 )—
输入图像的通道维度格式。如果未设置，将使用从输入推断的格式。


退货


导出常量元数据='未定义';
 

`np.ndarray`


导出常量元数据='未定义';


调整大小的图像。


调整大小
 `图像`
 到
 `（高度，宽度）`
 由指定
 `尺寸`
 使用 PIL 库。




#### 


Transformers.image\_transforms.to\_pil\_image


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_transforms.py#L157)



 (
 


 图像
 
 ：打字.Union [numpy.ndarray，ForwardRef（'PIL.Image.Image'），ForwardRef（'torch.Tensor'），ForwardRef（'tf.Tensor'），ForwardRef（'jnp.ndarray'）]


重新缩放
 
 : 打字.Optional[bool] = None


输入数据格式
 
 : Typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None


）
 

 →


导出常量元数据='未定义';
 

`PIL.Image.Image`


 参数


* **图像**
 （
 `PIL.Image.Image`
 或者
 `numpy.ndarray`
 或者
 `火炬.张量`
 或者
 `tf.张量`
 )—
要转换为的图像
 `PIL.Image`
 格式。
* **执行\_重新缩放**
 （
 `布尔`
 ,
 *选修的*
 )—
是否应用缩放因子（使像素值成为 0 到 255 之间的整数）。会默认的
到
 '真实'
 如果图像类型是浮动类型并转换为
 `int`
 会导致精度损失，
和
 ‘假’
 否则。
* **输入\_数据\_格式**
 （
 `通道维度`
 ,
 *选修的*
 )—
输入图像的通道维度格式。如果未设置，将使用从输入推断的格式。


退货


导出常量元数据='未定义';
 

`PIL.Image.Image`


导出常量元数据='未定义';


转换后的图像。


转换
 `图像`
 到 PIL 图像。如果满足以下条件，则可以选择重新缩放它并将通道尺寸放回到最后一个轴
需要。


## 图像处理混合




### 


班级
 

 变压器。
 

 图像处理混合


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_processing_utils.py#L68)



 (
 


 \*\*夸格




 )
 


这是一个图像处理器 mixin，用于提供顺序和图像特征的保存/加载功能
提取器。



#### 


获取\_图像


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_processing_utils.py#L520)



 (
 


 图片\_url\_或\_url
 
 : 打字.Union[str, 打字.List[str]]



 )
 


将单个或一系列 url 转换为相应的
 `PIL.Image`
 对象。


如果传递单个 url，则返回值将是单个对象。如果传递一个列表，则对象列表是
回。




#### 


来自\_dict


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_processing_utils.py#L383)



 (
 


 图像\_处理器\_dict
 
 : 打字.Dict[str, 打字.Any]


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

[ImageProcessingMixin](/docs/transformers/v4.35.2/en/main_classes/image_processor#transformers.ImageProcessingMixin)


 参数


* **图像\_处理器\_dict**
 （
 `字典[str，任意]`
 )—
将用于实例化图像处理器对象的字典。这样的字典可以是
通过利用从预先训练的检查点检索
 [to\_dict()](/docs/transformers/v4.35.2/en/internal/image_processing_utils#transformers.ImageProcessingMixin.to_dict)
 方法。
* **夸格斯**
 （
 `字典[str，任意]`
 )—
用于初始化图像处理器对象的附加参数。


退货


导出常量元数据='未定义';
 

[ImageProcessingMixin](/docs/transformers/v4.35.2/en/main_classes/image_processor#transformers.ImageProcessingMixin)


导出常量元数据='未定义';


从这些实例化的图像处理器对象
参数。


实例化一个类型
 [ImageProcessingMixin](/docs/transformers/v4.35.2/en/main_classes/image_processor#transformers.ImageProcessingMixin)
 来自 Python 参数字典。




#### 


来自\_json\_file


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_processing_utils.py#L440)



 (
 


 json\_file
 
 : 打字.Union[str, os.PathLike]


）
 

 →


导出常量元数据='未定义';
 

 类型的图像处理器
 [ImageProcessingMixin](/docs/transformers/v4.35.2/en/main_classes/image_processor#transformers.ImageProcessingMixin)


 参数


* **json\_file**
 （
 `str`
 或者
 `os.PathLike`
 )—
包含参数的 JSON 文件的路径。


退货


导出常量元数据='未定义';
 

 类型的图像处理器
 [ImageProcessingMixin](/docs/transformers/v4.35.2/en/main_classes/image_processor#transformers.ImageProcessingMixin)


导出常量元数据='未定义';


图像处理器对象
从该 JSON 文件实例化。


实例化类型的图像处理器
 [ImageProcessingMixin](/docs/transformers/v4.35.2/en/main_classes/image_processor#transformers.ImageProcessingMixin)
 从 JSON 的路径
参数文件。




#### 


来自\_预训练


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_processing_utils.py#L92)


（


 预训练\_model\_name\_or\_path
 
 : 打字.Union[str, os.PathLike]


缓存\_dir
 
 : 打字.Union[str, os.PathLike, NoneType] = None


强制\_下载
 
 ：布尔=假


仅本地\_文件\_
 
 ：布尔=假


代币
 
 : 打字.Union[str, bool, NoneType] = None


修订
 
 : str = '主'


\*\*夸格


）


 参数


* **预训练\_model\_name\_or\_path**
 （
 `str`
 或者
 `os.PathLike`
 )—
这可以是：
 
+ 一个字符串，
*型号编号*
托管在模型存储库内的预训练图像\_处理器
拥抱脸.co。有效的模型 ID 可以位于根级别，例如
`bert-base-uncased`
， 或者
以用户或组织名称命名，例如
`dbmdz/bert-base-德语-大小写`
。
+ 一条通往 a 的路径
*目录*
包含使用保存的图像处理器文件
[save\_pretrained()](/docs/transformers/v4.35.2/en/main_classes/image_processor#transformers.ImageProcessingMixin.save_pretrained)
方法，例如
`./my_model_directory/`
。
+ 已保存图像处理器 JSON 的路径或 url
*文件*
，例如，
`./my_model_directory/preprocessor_config.json`
。
* **缓存\_dir**
 （
 `str`
 或者
 `os.PathLike`
 ,
 *选修的*
 )—
如果是，则应缓存下载的预训练模型图像处理器的目录路径
不应使用标准缓存。
* **强制\_下载**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
如果出现以下情况，是否强制（重新）下载图像处理器文件并覆盖缓存的版本
它们存在。
* **继续\_下载**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否删除未完整接收的文件。如果存在此类文件，则尝试恢复下载
存在。
* **代理**
 （
 `字典[str, str]`
 ,
 *选修的*
 )—
由协议或端点使用的代理服务器字典，例如，
 `{'http': 'foo.bar:3128', 'http://主机名': 'foo.bar:4012'}。`
 每个请求都会使用代理。
* **代币**
 （
 `str`
 或者
 `布尔`
 ,
 *选修的*
 )—
用作远程文件的 HTTP 承载授权的令牌。如果
 '真实'
 ，或未指定，将使用
运行时生成的token
 `huggingface-cli 登录`
 （存储在
 `~/.huggingface`
 ）。
* **修订**
 （
 `str`
 ,
 *选修的*
 ，默认为
 `“主要”`
 )—
要使用的具体型号版本。它可以是分支名称、标签名称或提交 ID，因为我们使用
基于 git 的系统，用于在 Huggingface.co 上存储模型和其他工件，因此
 `修订`
 可以是任何
git 允许的标识符。


 实例化一个类型
 [ImageProcessingMixin](/docs/transformers/v4.35.2/en/main_classes/image_processor#transformers.ImageProcessingMixin)
 来自图像处理器。


 例子：



```
# We can't instantiate directly the base class \*ImageProcessingMixin\* so let's show the examples on a
# derived class: \*CLIPImageProcessor\*
image_processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)  # Download image\_processing\_config from huggingface.co and cache.
image_processor = CLIPImageProcessor.from_pretrained(
    "./test/saved\_model/"
)  # E.g. image processor (or model) was saved using \*save\_pretrained('./test/saved\_model/')\*
image_processor = CLIPImageProcessor.from_pretrained("./test/saved\_model/preprocessor\_config.json")
image_processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", do_normalize=False, foo=False
)
assert image_processor.do_normalize is False
image_processor, unused_kwargs = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", do_normalize=False, foo=False, return_unused_kwargs=True
)
assert image_processor.do_normalize is False
assert unused_kwargs == {"foo": False}
```


#### 


获取\_image\_processor\_dict


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_processing_utils.py#L268)


（


 预训练\_model\_name\_or\_path
 
 : 打字.Union[str, os.PathLike]


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

`元组[字典，字典]`


 参数


* **预训练\_model\_name\_or\_path**
 （
 `str`
 或者
 `os.PathLike`
 )—
我们想要参数字典的预训练检查点的标识符。
* **子文件夹**
 （
 `str`
 ,
 *选修的*
 ，默认为
 `````
 )—
如果相关文件位于 Huggingface.co 上模型存储库的子文件夹内，您可以
在此指定文件夹名称。


退货


导出常量元数据='未定义';
 

`元组[字典，字典]`


导出常量元数据='未定义';


将用于实例化图像处理器对象的字典。


来自一个
 `预训练模型名称或路径`
 ，解析为参数字典，用于实例化
图像处理器类型
 `~image_processor_utils.ImageProcessingMixin`
 使用
 `from_dict`
 。




#### 


推送\_到\_集线器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/utils/hub.py#L791)


（


 仓库\_id
 
 : 字符串


使用\_temp\_dir
 
 : 打字.Optional[bool] = None


提交\_消息
 
 : 打字.可选[str] =无


私人的
 
 : 打字.Optional[bool] = None


代币
 
 : 打字.Union[bool, str, NoneType] = None


最大分片大小
 
 : 打字.Union[int, str, NoneType] = '5GB'


创建\_pr
 
 ：布尔=假


安全\_序列化
 
 ：布尔=真


修订
 
 ：str=无


提交\_描述
 
 ：str=无


\*\*已弃用\_kwargs


）


 参数


* **仓库\_id**
 （
 `str`
 )—
您要将图像处理器推送到的存储库的名称。它应包含您的组织名称
当推送到给定组织时。
* **使用\_temp\_dir**
 （
 `布尔`
 ,
 *选修的*
 )—
是否使用临时目录来存储推送到 Hub 之前保存的文件。
将默认为
 '真实'
 如果没有类似名称的目录
 `repo_id`
 ,
 ‘假’
 否则。
* **提交\_消息**
 （
 `str`
 ,
 *选修的*
 )—
推送时要提交的消息。将默认为
 `“上传图像处理器”`
 。
* **私人的**
 （
 `布尔`
 ,
 *选修的*
 )—
创建的存储库是否应该是私有的。
* **代币**
 （
 `布尔`
 或者
 `str`
 ,
 *选修的*
 )—
用作远程文件的 HTTP 承载授权的令牌。如果
 '真实'
 ，将使用生成的令牌
跑步时
 `huggingface-cli 登录`
 （存储在
 `~/.huggingface`
 ）。将默认为
 '真实'
 如果
 `repo_url`
 没有指定。
* **最大\_分片\_大小**
 （
 `int`
 或者
 `str`
 ,
 *选修的*
 ，默认为
 `“5GB”`
 )—
仅适用于型号。分片之前检查点的最大大小。检查点分片
那么每个尺寸都会小于这个尺寸。如果表示为字符串，则后面需要跟数字
按一个单位（例如
 `“5MB”`
 ）。我们默认为
 `“5GB”`
 以便用户可以轻松地在免费层上加载模型
Google Colab 实例没有任何 CPU OOM 问题。
* **创建\_pr**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否使用上传的文件创建 PR 或直接提交。
* **安全\_序列化**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 '真实'
 )—
是否将模型权重转换为安全张量格式以实现更安全的序列化。
* **修订**
 （
 `str`
 ,
 *选修的*
 )—
将上传的文件推送到的分支。
* **提交\_描述**
 （
 `str`
 ,
 *选修的*
 )—
将创建的提交的描述


 将图像处理器文件上传到🤗模型中心。


 例子：



```
from transformers import AutoImageProcessor

image processor = AutoImageProcessor.from_pretrained("bert-base-cased")

# Push the image processor to your namespace with the name "my-finetuned-bert".
image processor.push_to_hub("my-finetuned-bert")

# Push the image processor to an organization with the name "my-finetuned-bert".
image processor.push_to_hub("huggingface/my-finetuned-bert")
```


#### 


注册\_for\_auto\_class


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_processing_utils.py#L494)


（


 自动\_类
 
 = '自动图像处理器'


）


 参数


* **汽车\_class**
 （
 `str`
 或者
 `类型`
 ,
 *选修的*
 ，默认为
 `“自动图像处理器”`
 )—
用于注册这个新图像处理器的自动类。


 使用给定的自动类注册此类。这应该只用于自定义图像处理器
库中已经映射了
 `自动图像处理器`
 。


此 API 是实验性的，在下一版本中可能会有一些轻微的重大更改。


#### 


保存\_预训练


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_processing_utils.py#L207)


（


 保存\_目录
 
 : 打字.Union[str, os.PathLike]


推送\_到\_集线器
 
 ：布尔=假


\*\*夸格


）


 参数


* **保存\_目录**
 （
 `str`
 或者
 `os.PathLike`
 )—
保存图像处理器 JSON 文件的目录（如果不存在则创建）。
* **推送\_到\_hub**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
保存模型后是否将其推送到 Hugging Face 模型中心。您可以指定
您想要推送到的存储库
 `repo_id`
 （将默认为名称
 `保存目录`
 在你的
命名空间）。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
传递给的附加关键字参数
 [push\_to\_hub()](/docs/transformers/v4.35.2/en/main_classes/configuration#transformers.PretrainedConfig.push_to_hub)
 方法。


 将图像处理器对象保存到目录中
 `保存目录`
 ，以便可以使用重新加载
 [来自\_pretrained()](/docs/transformers/v4.35.2/en/main_classes/image_processor#transformers.ImageProcessingMixin.from_pretrained)
 类方法。




#### 


到\_dict


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_processing_utils.py#L428)


（
 

 ）
 

 →


导出常量元数据='未定义';
 

`字典[str，任意]`


退货


导出常量元数据='未定义';
 

`字典[str，任意]`


导出常量元数据='未定义';


组成此图像处理器实例的所有属性的字典。


将此实例序列化为 Python 字典。




#### 


到\_json\_file


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_processing_utils.py#L480)


（


 json\_文件\_路径
 
 : 打字.Union[str, os.PathLike]


）


 参数


* **json\_file\_path**
 （
 `str`
 或者
 `os.PathLike`
 )—
将保存此 image\_processor 实例参数的 JSON 文件的路径。


 将此实例保存到 JSON 文件。




#### 


到\_json\_string


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/image_processing_utils.py#L459)


（
 

 ）
 

 →


导出常量元数据='未定义';
 

`str`


退货


导出常量元数据='未定义';
 

`str`


导出常量元数据='未定义';


包含组成此 feature\_extractor 实例的 JSON 格式的所有属性的字符串。


将此实例序列化为 JSON 字符串。