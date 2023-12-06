# 特征提取器

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/main_classes/feature_extractor>
>
> 原始地址：<https://huggingface.co/docs/transformers/main_classes/feature_extractor>


特征提取器负责为音频或视觉模型准备输入特征。这包括从序列中提取特征，例如预处理音频文件以生成 Log-Mel 频谱图特征，从图像中提取特征，例如裁剪图像文件，还包括填充、标准化以及转换为 NumPy、PyTorch 和 TensorFlow 张量。


## 特征提取混合




### 


班级
 

 变压器。
 

 特征提取混合


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/feature_extraction_utils.py#L240)



 (
 


 \*\*夸格




 )