# 故障排除

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/developer_guides/troubleshooting>
>
> 原始地址：<https://huggingface.co/docs/peft/developer_guides/troubleshooting>


如果您在使用 PEFT 时遇到任何问题，请查看以下常见问题列表及其解决方案。


## 例子不起作用



示例通常依赖于最新的软件包版本，因此请确保它们是最新的。特别是检查以下软件包的版本：


* `佩夫特`
*`变形金刚`
*`加速`
*`torch`


一般来说，您可以通过在 Python 环境中运行以下命令来更新软件包版本：



```
python -m pip install -U <package_name>
```


从源代码安装 PEFT 对于跟上最新发展很有用：



```
python -m pip install git+https://github.com/huggingface/peft
```


## 加载的 PEFT 模型产生不良结果



加载的 PEFT 模型得到较差结果可能有多种原因，如下所列。如果您仍然无法解决问题，请查看其他人是否有类似的情况
 [问题](https://github.com/huggingface/peft/issues)
 在 GitHub 上，如果找不到任何问题，请打开一个新问题。


当打开一个问题时，如果您提供一个重现该问题的最小代码示例，将会有很大帮助。另外，请报告加载的模型的性能是否与微调前的模型相同，是否以随机水平运行，或者是否仅比预期稍差。这些信息可以帮助我们更快地识别问题。


### 


 随机偏差


如果您的模型输出与之前的运行不完全相同，则可能存在随机元素问题。例如：


1.请确保它在
 `.eval()`
 模式，这很重要，例如，如果模型使用 dropout
2.如果你使用
 [生成](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_ Generation#transformers.GenerationMixin.generate)
 在语言模型上，可能存在随机采样，因此获得相同的结果需要设置随机种子
3. 如果您使用量化并合并权重，则由于舍入误差，预计会出现小偏差


### 


 模型加载不正确


请确保正确加载模型。一个常见的错误是尝试加载
 *受训*
 模型与
 `get_peft_model`
 ，这是不正确的。相反，加载代码应该如下所示：



```
from peft import PeftModel, PeftConfig

base_model = ...  # to load the base model, use the same code as when you trained it
config = PeftConfig.from_pretrained(peft_model_id)
peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
```


### 


 随机初始化层


对于某些任务，正确配置非常重要
 `要保存的模块`
 在配置中考虑随机初始化的层。


举个例子，如果您使用 LoRA 微调序列分类的语言模型，这是必要的，因为🤗 Transformers 在模型顶部添加了一个随机初始化的分类头。如果不添加该层
 `要保存的模块`
 ，分类头不会被保存。下次加载模型时，您将得到
 *不同的*
 随机初始化的分类头，导致完全不同的结果。


在 PEFT 中，我们尝试正确猜测
 `要保存的模块`
 如果您提供
 `任务类型`
 配置中的参数。这应该适用于遵循标准命名方案的变压器模型。不过，仔细检查总是一个好主意，因为我们不能保证所有模型都遵循命名方案。


当您加载具有随机初始化层的 Transformer 模型时，您应该看到如下警告：



```
Some weights of <MODEL> were not initialized from the model checkpoint at <ID> and are newly initialized: [<LAYER_NAMES>].
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```


应将提到的层添加到
 `要保存的模块`
 在配置中以避免所描述的问题。