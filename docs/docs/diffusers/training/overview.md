# 🧨 扩散器培训示例

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/overview>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/overview>


扩散器培训示例是一组脚本，用于演示如何有效地使用扩散器
 `扩散器`
 图书馆
适用于各种用例。


**笔记**
 : 如果你正在寻找
 **官方的**
 有关如何使用的示例
 `扩散器`
 为了推断，
请看一下
 [src/diffusers/pipelines](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines)


我们的榜样渴望成为
 **独立**
 ,
 **易于调整**
 ,
 **适合初学者**
 并为
 **只有一个目的**
 。
更具体地说，这意味着：


* **独立**
 ：示例脚本应仅依赖于可在以下位置找到的“pip-install-able”Python 包：
 `需求.txt`
 文件。示例脚本应
 **不是**
 依赖于任何本地文件。这意味着人们可以简单地下载一个示例脚本，
 *例如。*
[train\_unconditional.py](https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_ Generation/train_unconditional.py)
 ，安装所需的依赖项，
 *例如。*
[requirements.txt](https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_ Generation/requirements.txt)
 并执行示例脚本。
* **易于调整**
 ：虽然我们努力展示尽可能多的用例，但示例脚本只是示例。预计它们不会开箱即用地解决您的特定问题，并且您将需要更改几行代码才能使其适应您的需求。为了帮助您做到这一点，大多数示例都充分公开了数据的预处理和训练循环，以便您根据需要调整和编辑它们。
* **适合初学者**
 ：我们的目标不是为最新模型提供最先进的训练脚本，而是提供可以用来更好地理解扩散模型以及如何将它们与
 `扩散器`
 图书馆。如果我们认为某些最先进的方法对于初学者来说太复杂，我们经常会故意忽略它们。
* **只有一个目的**
 ：示例应仅显示一项任务且一项任务。即使任务来自建模
观点非常相似，
 *例如。*
 图像超分辨率和图像修改往往使用相同的模型和训练方法，我们希望示例仅展示一项任务，以使其尽可能具有可读性和易于理解。


我们提供
 **官方的**
 涵盖扩散模型最常见任务的示例。
 *官方的*
 例子是
 **积极**
 由维护
 `扩散器`
 维护者，我们尝试严格遵循上面定义的示例哲学。
如果您觉得应该存在另一个重要的例子，我们非常乐意欢迎
 [功能请求](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feature_request.md&title=)
 或直接
 [拉取请求](https://github.com/huggingface/diffusers/compare)
 来自你！


训练示例展示了如何针对各种任务预训练或微调扩散模型。目前我们支持：


* [无条件训练](./unconditional_training)
* [文本转图像训练](./text2image)
\*
* [文本反转](./text_inversion)
* [梦境](./梦境)
\*
* [LoRA 支持](./lora)
\*
* [控制网](./controlnet)
\*
* [InstructPix2Pix](./instructpix2pix)
\*
* [自定义扩散](./custom_diffusion)
* [T2I-适配器](./t2i_adapters)
\*


\*
 ：支持[稳定扩散XL](../api/pipelines/stable\_diffusion/stable\_diffusion\_xl)。
 
 如果可以的话请
 [安装 xFormers](../optimization/xformers)
 为了记忆有效的注意力。这可以帮助您加快训练速度并减少内存消耗。


| 	 Task	  | 	 🤗 Accelerate	  | 	 🤗 Datasets	  | 	 Colab	  |
| --- | --- | --- | --- |
| [**Unconditional Image Generation**](./unconditional_training) | 	 ✅	  | 	 ✅	  | [Open In Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) |
| [**Text-to-Image fine-tuning**](./text2image) | 	 ✅	  | 	 ✅	  |  |
| [**Textual Inversion**](./text_inversion) | 	 ✅	  | 	 -	  | [Open In Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb) |
| [**Dreambooth**](./dreambooth) | 	 ✅	  | 	 -	  | [Open In Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb) |
| [**Training with LoRA**](./lora) | 	 ✅	  | 	 -	  | 	 -	  |
| [**ControlNet**](./controlnet) | 	 ✅	  | 	 ✅	  | 	 -	  |
| [**InstructPix2Pix**](./instructpix2pix) | 	 ✅	  | 	 ✅	  | 	 -	  |
| [**Custom Diffusion**](./custom_diffusion) | 	 ✅	  | 	 ✅	  | 	 -	  |
| [**T2I Adapters**](./t2i_adapters) | 	 ✅	  | 	 ✅	  | 	 -	  |


## 社区



此外，我们还提供
 **社区**
 示例，这是我们社区添加和维护的示例。
社区示例可以包含两者
 *训练*
 例子或
 *推理*
 管道。
对于这样的例子，我们对上面定义的理念比较宽容，也不能保证为每个问题提供维护。
对社区有用但尚未被认为流行或尚未遵循我们上述理念的示例应放入
 [社区示例](https://github.com/huggingface/diffusers/tree/main/examples/community)
 文件夹。因此，社区文件夹包括训练示例和推理管道。
 **笔记**
 ：社区示例可以是
 [伟大的第一个贡献](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
 向社区展示您喜欢如何使用
 `扩散器`
 🪄。


## 重要的提示



为了确保您可以成功运行示例脚本的最新版本，您必须
 **从源安装库**
 并安装一些特定于示例的要求。为此，请在新的虚拟环境中执行以下步骤：



```
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```


然后 cd 到您选择的示例文件夹中并运行



```
pip install -r requirements.txt
```