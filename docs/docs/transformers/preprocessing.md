# 预处理

> 译者：[BrightLi](https://github.com/brightli)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/preprocessing>
>
> 原始地址：<https://huggingface.co/docs/transformers/preprocessing>

在你可以对数据集进行模型训练之前，数据需要预处理为模型所期望的输入格式。无论你的数据是文本、图像还是音频，它们都需要被转换并组装成张量的批次。🤗 Transformers 提供了一组预处理类来帮助准备你的数据用于模型。在本教程中，你将学习以下内容：

* 对于文本，使用[Tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer)将文本转换为一系列的标记，创建标记的数值表示，并将它们组装成张量。
* 对于语音和音频，使用[Feature Extractor](https://huggingface.co/docs/transformers/main_classes/feature_extractor)从音频波形中提取序列特征并将其转换成张量。
* 对于图像输入，使用[ImageProcessor](https://huggingface.co/docs/transformers/main_classes/image_processor)将图像转换成张量。
* 对于多模态输入，使用[Processor](https://huggingface.co/docs/transformers/main_classes/processors)结合一个 tokenizer 和一个 feature extractor 或 image processor。



> AutoProcessor 总是有效，并且会自动为你所使用的模型选择正确的类，无论你是使用 tokenizer、image processor、feature extractor 还是 processor。

在开始之前，安装 🤗 Datasets 以便你可以加载一些数据集进行实验：

```py
pip install datasets
```

**自然语言处理**


<video src="https://youtu.be/Yffk5aydLzg" controls></video>
