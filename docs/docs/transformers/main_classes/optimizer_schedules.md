# 优化

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/main_classes/optimizer_schedules>
>
> 原始地址：<https://huggingface.co/docs/transformers/main_classes/optimizer_schedules>


这
 `.优化`
 模块提供：


* 修复了权重衰减的优化器，可用于微调模型，以及
* 继承自schedule对象形式的几个schedule
 `_LRSchedule`
 :
* 一个梯度累积类，用于累积多个batch的梯度


## AdamW (PyTorch)




### 


班级
 

 变压器。
 

 亚当·W


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/optimization.py#L378)



 (
 


 参数
 
 : 打字.Iterable[torch.nn.parameter.Parameter]


LR
 
 : 浮点数 = 0.001


贝塔斯
 
 : 打字.Tuple[浮点, 浮点] = (0.9, 0.999)


每股收益
 
 ：浮动= 1e-06


权重衰减
 
 ：浮动= 0.0


正确\_偏差
 
 ：布尔=真


无\_弃用\_警告
 
 ：布尔=假



 )
 


 参数


* **参数**
 （
 `可迭代[nn.参数.参数]`
 )—
用于优化的可迭代参数或定义参数组的字典。
* **lr**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 0.001) —
使用的学习率。
* **测试版**
 （
 `元组[浮点数，浮点数]`
 ,
 *选修的*
 ，默认为
 `（0.9，0.999）`
 )—
Adam 的 beta 参数 (b1, b2)。
* **每股收益**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 1e-06) —
Adam 的 epsilon 用于数值稳定性。
* **重量\_衰减**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 0.0) —
应用解耦权重衰减。
* **正确\_偏差**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 '真实'
 )—
是否纠正 Adam 中的偏见（例如，在他们使用的 Bert TF 存储库中）
 ‘假’
 ）。
* **无\_弃用\_警告**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
用于禁用弃用警告的标志（设置为
 '真实'
 禁用警告）。


 实现带有权重衰减修复的 Adam 算法，如中介绍的
 [解耦权重衰减
正则化](https://arxiv.org/abs/1711.05101)
 。



#### 


步


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/optimization.py#L429)



 (
 


 关闭
 
 : 打字.Callable = 无



 )
 


 参数


* **关闭**
 （
 `可调用`
 ,
 *选修的*
 ) — 重新评估模型并返回损失的闭包。


 执行单个优化步骤。


## AdaFactor (PyTorch)




### 


班级
 

 变压器。
 

 阿达法特


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/optimization.py#L492)



 (
 


 参数


 LR
 
 = 无


每股收益
 
 = (1e-30, 0.001)


剪辑\_阈值
 
 = 1.0


衰减\_率
 
 =-0.8


β1
 
 = 无


权重衰减
 
 = 0.0


比例\_参数
 
 = 正确


相对\_step
 
 = 正确


预热\_init
 
 = 假



 )
 


 参数


* **参数**
 （
 `可迭代[nn.参数.参数]`
 )—
用于优化的可迭代参数或定义参数组的字典。
* **lr**
 （
 `浮动`
 ,
 *选修的*
 )—
外部学习率。
* **每股收益**
 （
 `元组[浮点数，浮点数]`
 ,
 *选修的*
 ，默认为
 `(1e-30, 0.001)`
 )—
分别用于平方梯度和参数尺度的正则化常数
* **剪辑\_阈值**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 1.0) —
最终梯度更新均方根阈值
* **衰减率**
 （
 `浮动`
 ,
 *选修的*
 ，默认为-0.8) —
用于计算平方的运行平均值的系数
* **测试版1**
 （
 `浮动`
 ,
 *选修的*
 )—
用于计算梯度运行平均值的系数
* **重量\_衰减**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 0.0) —
权重衰减（L2 惩罚）
* **比例\_参数**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 '真实'
 )—
如果为真，则学习率按均方根缩放
* **相对\_step**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 '真实'
 )—
如果为 True，则计算与时间相关的学习率而不是外部学习率
* **预热\_init**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
与时间相关的学习率计算取决于是否使用预热初始化


 AdaFactor pytorch 实现可以用作 Adam 原始 fairseq 代码的替代品：
 <https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py>


纸：
 *Adafactor：具有次线性内存成本的自适应学习率*
<https://arxiv.org/abs/1804.04235>
 注意
该优化器根据以下情况在内部调整学习率
 `scale_parameter`
 ,
 `相对步骤`
 和
 `预热初始化`
 选项。要使用手动（外部）学习率计划，您应该设置
 `scale_parameter=False`
 和
 `relative_step=False`
 。


此实现处理低精度（FP16、bfloat）值，但我们尚未进行彻底测试。


推荐的 T5 微调设置（
 <https://discuss.huggingface.co/t/t5-finetuning-tips/684/3>
 ）：


* 不建议在没有 LR 预热或 Clip\_threshold 的情况下进行训练。


+ 使用预定的 LR 预热来固定 LR
+ 使用clip\_threshold=1.0 (
<https://arxiv.org/abs/1804.04235>
）
* 禁用相关更新
* 使用scale\_parameter=False
* 梯度裁剪等其他优化器操作不应与 Adafactor 一起使用


 例子：



```
Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
```


其他人报告说以下组合效果良好：



```
Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
```


使用时
 `lr=无`
 和
 [培训师](/docs/transformers/v4.35.2/en/main_classes/trainer#transformers.Trainer)
 你很可能需要使用
 `AdafactorSchedule`


调度程序如下：



```
from transformers.optimization import Adafactor, AdafactorSchedule

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)
trainer = Trainer(..., optimizers=(optimizer, lr_scheduler))
```


用法：



```
# replace AdamW with Adafactor
optimizer = Adafactor(
    model.parameters(),
    lr=1e-3,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False,
)
```




#### 


步


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/optimization.py#L638)



 (
 


 关闭
 
 = 无



 )
 


 参数


* **关闭**
 (可调用，可选) - 重新评估模型的闭包
并返还损失。


 执行单个优化步骤


## AdamWeightDecay (TensorFlow)




### 


班级
 

 变压器。
 

 Adam权重衰减


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/optimization_tf.py#L172)



 (
 


 学习率
 
 : 打字.Union[float, keras.src.optimizers.schedules.learning\_rate\_schedule.LearningRateSchedule] = 0.001


测试\_1
 
 ：浮动= 0.9


测试\_2
 
 ：浮动= 0.999


厄普西隆
 
 ：浮动= 1e-07


阿姆斯格勒
 
 ：布尔=假


重量衰减率
 
 ：浮动= 0.0


包括\_in\_weight\_decay
 
 : 打字.可选[打字.列表[str]] =无


从权重衰减中排除\_
 
 : 打字.可选[打字.列表[str]] =无


姓名
 
 : str = 'AdamWeightDecay'


\*\*夸格




 )
 


 参数


* **学习率**
 （
 `Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]`
 ,
 *选修的*
 ，默认为 0.001) —
使用的学习率或时间表。
* **测试版\_1**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 0.9) —
Adam 中的 beta1 参数，是第一个动量估计的指数衰减率。
* **测试版\_2**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 0.999) —
Adam 中的 beta2 参数，是第二次动量估计的指数衰减率。
* **厄普西隆**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 1e-07) —
Adam 中的 epsilon 参数，它是一个用于数值稳定性的小常数。
* **阿姆斯格勒**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否应用该算法的 AMSGrad 变体，请参见
 [关于亚当和
超越](https://arxiv.org/abs/1904.09237)
 。
* **权重\_衰减\_率**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 0.0) —
要应用的权重衰减。
* **包括\_in\_weight\_decay**
 （
 `列表[str]`
 ,
 *选修的*
 )—
要应用权重衰减的参数名称（或重新模式）的列表。如果没有通过，则权重衰减为
默认情况下应用于所有参数（除非它们位于
 `排除权重衰减`
 ）。
* **从\_权重\_衰减中排除\_**
 （
 `列表[str]`
 ,
 *选修的*
 )—
要排除应用权重衰减的参数名称（或重新模式）的列表。如果一个
 `include_in_weight_decay`
 通过后，其中的名称将取代此列表。
* **姓名**
 （
 `str`
 ,
 *选修的*
 ，默认为
 `“亚当权重衰变”`
 )—
应用渐变时创建的操作的可选名称。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
关键字参数。允许为{
 `剪辑规范`
 ,
 `剪辑值`
 ,
 `lr`
 ,
 ‘腐烂’
 }。
 `剪辑规范`
 是剪辑梯度
规范;
 `剪辑值`
 是按值剪辑梯度，
 ‘腐烂’
 包含向后兼容性以留出时间
学习率的逆衰减。
 `lr`
 包含向后兼容性，推荐使用
 `学习率`
 反而。


 Adam 在梯度上启用 L2 权重衰减和 Clip\_by\_global\_norm。只需将权重的平方添加到
损失函数是
 *不是*
 使用 Adam 的 L2 正则化/权重衰减的正确方法，因为这会相互作用
以奇怪的方式使用 m 和 v 参数，如下所示
 [解耦权重衰减
正则化](https://arxiv.org/abs/1711.05101)
 。


相反，我们希望以不与 m/v 参数相互作用的方式衰减权重。这相当于
使用普通（非动量）SGD 将权重的平方添加到损失中。



#### 


来自\_config


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/optimization_tf.py#L229)



 (
 


 配置




 )
 


使用 WarmUp 自定义对象从其配置创建优化器。


#### 


Transformers.create\_optimizer


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/optimization_tf.py#L88)



 (
 


 初始化\_lr
 
 ： 漂浮


训练步数
 
 ：整数


num\_warmup\_steps
 
 ：整数


最小\_lr\_比率
 
 ：浮动= 0.0


亚当\_beta1
 
 ：浮动= 0.9