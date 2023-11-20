# 如何为扩散器做出贡献 🧨

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/conceptual/contribution>
>
> 原始地址：<https://huggingface.co/docs/diffusers/conceptual/contribution>


我们❤️来自开源社区的贡献！欢迎所有人，所有类型的参与（而不仅仅是代码）都受到重视和赞赏。回答问题、帮助他人、伸出援手和改进文档对于社区来说都是非常有价值的，所以如果您愿意的话，请不要害怕并参与其中！


我们鼓励每个人首先在我们的公共 Discord 频道中说 👋。我们讨论扩散模型的最新趋势，提出问题，展示个人项目，互相帮助贡献，或者只是闲逛☕。
 [![加入我们的 Discord](https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white)](https://Discord.gg/G7tWnz98XR)


无论您选择哪种方式做出贡献，我们都努力成为开放、热情和友善社区的一部分。请阅读我们的
 [行为准则](https://github.com/huggingface/diffusers/blob/main/CODE_OF_CONDUCT.md)
 并在互动过程中注意尊重它。我们还建议您熟悉
 [道德准则](https://huggingface.co/docs/diffusers/conceptual/ethical_guidelines)
 它指导我们的项目，并要求您遵守同样的透明度和责任原则。


我们非常重视社区的反馈，因此，如果您认为自己有宝贵的反馈可以帮助改进库，请不要害怕说出来 - 每条消息、评论、问题和拉取请求 (PR) 都会被阅读和考虑。


## 概述



您可以通过多种方式做出贡献，从回答有关问题的问题到添加新的扩散模型
核心库。


下面，我们概述了不同的贡献方式，并按难度升序排列。所有这些对社区都很有价值。


* 1. 提出并回答问题
[Diffusers 讨论论坛](https://discuss.huggingface.co/c/discussion-lated-to-httpsgithubcomhuggingfacediffusers)
或上
[不和谐](https://discord.gg/G7tWnz98XR)
。
* 2. 打开新问题
[GitHub 问题选项卡](https://github.com/huggingface/diffusers/issues/new/choose)
。
* 3. 解答问题
[GitHub 问题选项卡](https://github.com/huggingface/diffusers/issues)
。
* 4. 修复一个简单问题，标有“Good first issues”标签，参见
[此处](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
。
* 5. 为
[文档](https://github.com/huggingface/diffusers/tree/main/docs/source)
。
* 6. 贡献一个
[社区管道](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3Acommunity-examples)
。
* 7. 为
[示例](https://github.com/huggingface/diffusers/tree/main/examples)
。
* 8.修复一个较难的问题，标有“Good Second Issue”标签，参见
[此处](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22Good+second+issue%22)
。
* 9. 添加新的管道、模型或调度程序，请参见
[“新管道/模型”](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
和
[“新调度程序”](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)
问题。对于此贡献，请查看
[设计理念](https://github.com/huggingface/diffusers/blob/main/PHILOSOPHY.md)
。


正如之前所说，
 **所有贡献对社区都很有价值**
 。
下面，我们将更详细地解释每项贡献。


对于所有贡献 4 - 9，您将需要创建一个 PR。详细解释了如何执行此操作
 [打开拉取请求](#how-to-open-a-pr)
 。


### 


 1. 在 Diffusers 论坛或 Diffusers Discord 上提问和回答问题


任何与 Diffusers 库相关的问题或评论都可以在
 [讨论论坛](https://discuss.huggingface.co/c/discussion-lated-to-httpsgithubcomhuggingfacediffusers/)
 或上
 [不和谐](https://discord.gg/G7tWnz98XR)
 。此类问题和评论包括（但不限于）：


* 试图分享知识的训练或推理实验报告
* 个人项目介绍
* 针对非官方训练示例的问题
* 项目建议书
* 一般反馈
* 论文摘要
* 寻求有关建立在 Diffusers 库之上的个人项目的帮助
* 一般的问题
* 有关扩散模型的伦理问题
*……


在论坛或 Discord 上提出的每个问题都积极鼓励社区公开
分享知识，很可能会帮助将来有与您相同问题的初学者
拥有。请提出您可能有的任何问题。
本着同样的精神，您通过回答这些问题对社区有巨大的帮助，因为这样您就可以公开记录知识供每个人学习。


**请**
 请记住，您提出或回答问题的努力越多，获得的回报就越高。
公开记录的知识的质量。同样，提出得当且回答得当的问题会创建一个可供每个人访问的高质量知识数据库，而提出得不好的问题或答案会降低公共知识数据库的整体质量。
简而言之，高质量的问题或答案是
 *精确的*
 ,
 *简洁的*
 ,
 *相关的*
 ,
 *容易明白*
 ,
 *无障碍*
 ， 和
 *格式良好/姿势良好*
 。欲了解更多信息，请浏览
 [如何写好期刊](#how-to-write-a-good-issue)
 部分。


**有关渠道的注意事项**
 :
 [*论坛*](https://discuss.huggingface.co/c/discussion-lated-to-httpsgithubcomhuggingfacediffusers/63)
 可以更好地被搜索引擎（例如 Google）索引。帖子按受欢迎程度而不是按时间顺序排名。因此，可以更轻松地查找我们不久前发布的问题和答案。
此外，可以轻松链接到论坛中发布的问题和答案。
相比之下，
 *不和谐*
 具有类似聊天的格式，可以进行快速的来回通信。
虽然您在 Discord 上得到问题的答案很可能会花费更少的时间，但您的
随着时间的推移，问题将不再可见。此外，要找到不久前发布在 Discord 上的信息要困难得多。因此，我们强烈建议使用该论坛来提供高质量的问题和答案，以期为社区创造持久的知识。如果 Discord 上的讨论得出了非常有趣的答案和结论，我们建议将结果发布在论坛上，以便将来的读者更容易获得这些信息。


### 


 2. 在 GitHub issues 选项卡上打开新问题


感谢通知我们的用户，🧨 Diffusers 库强大且可靠
他们遇到的问题。感谢您报告问题。


请记住，GitHub 问题保留用于与 Diffusers 库直接相关的技术问题、错误报告、功能请求或库设计反馈。


简而言之，这意味着一切
 **不是**
 相关于
 **扩散器库的代码**
 （包括文档）应该
 **不是**
 在 GitHub 上询问，而是在
 [论坛](https://discuss.huggingface.co/c/discussion-lated-to-httpsgithubcomhuggingfacediffusers/63)
 或者
 [不和谐](https://discord.gg/G7tWnz98XR)
 。


**打开新问题时请考虑以下准则**
 :


* 确保您已经搜索过您的问题之前是否已被询问过（使用 GitHub 上“问题”下的搜索栏）。
* 请勿在其他（相关）问题上报告新问题。如果另一个问题高度相关，请
尽管如此，还是打开一个新问题并链接到相关问题。
* 确保您的问题是用英文写的。请使用一项出色的免费在线翻译服务，例如
 [DeepL](https://www.deepl.com/translator)
 如果您不熟悉英语，可以将您的母语翻译成英语。
* 检查您的问题是否可以通过更新到最新的 Diffusers 版本来解决。在发布您的问题之前，请确保
 `python -c“导入扩散器；打印（扩散器.__version__）”`
 更高或与最新的扩散器版本匹配。
* 请记住，您在打开新问题时投入的精力越多，您的答案质量就越高，并且 Diffusers 问题的整体质量就越好。


新问题通常包括以下内容。


#### 


 2.1.可重复、最少的错误报告


错误报告应始终具有可重现的代码片段，并尽可能简洁。
更详细地说，这意味着：


* 尽可能缩小错误范围，
 **不要只是转储整个代码文件**
 。
* 格式化你的代码。
* 除了依赖于它们的 Diffuser 之外，不要包含任何外部库。
* **总是**
 提供有关您的环境的所有必要信息；为此，您可以运行：
 `diffusers-cli env`
 在您的 shell 中并将显示的信息复制粘贴到问题中。
* 解释一下这个问题。如果读者不知道问题是什么以及为什么会成为问题，那么她就无法解决它。
* **总是**
 确保读者可以尽可能轻松地重现您的问题。如果您的代码片段由于缺少库或未定义的变量而无法运行，读者无法为您提供帮助。确保可重现的代码片段尽可能少，并且可以复制粘贴到简单的 Python shell 中。
* 如果为了重现您的问题需要模型和/或数据集，请确保读者有权访问该模型或数据集。您随时可以将模型或数据集上传到
 [中心](https://huggingface.co)
 使其易于下载。尝试使模型和数据集尽可能小，以便尽可能轻松地重现问题。


欲了解更多信息，请浏览
 [如何写好期刊](#how-to-write-a-good-issue)
 部分。


您可以打开错误报告
 [此处](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=bug&projects=&template=bug-report.yml)
 。


#### 


 2.2.功能请求


世界级的功能请求解决以下几点：


1. 动机第一：


* 这是否与图书馆的问题/挫折有关？如果是这样，请解释一下
为什么。提供一个演示问题的代码片段是最好的。
* 它与您的项目需要的东西相关吗？我们很想听听
关于它！
* 这是您所从事并认为可以造福社区的事情吗？
惊人的！告诉我们它为您解决了什么问题。


2. 写一个
 *完整段落*
 描述该功能；
3. 提供一个
 **代码片段**
 展示其未来用途；
4.如与论文相关，请附上链接；
5. 附上您认为可能有帮助的任何其他信息（图纸、屏幕截图等）。


您可以打开功能请求
 [此处](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feature_request.md&title=)
 。


#### 


 2.3 反馈


关于库设计及其好坏的反馈极大地帮助核心维护人员构建一个用户友好的库。要了解当前设计理念背后的理念，请看一下
 [此处](https://huggingface.co/docs/diffusers/conceptual/philosophy)
 。如果您觉得某个设计选择不符合当前的设计理念，请解释原因以及应如何更改。如果某个设计选择过于遵循设计理念，从而限制了用例，请解释为什么以及如何更改它。
如果某个设计选择对您非常有用，也请留言，因为这对未来的设计决策来说是很好的反馈。


您可以提出有关反馈的问题
 [此处](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feedback.md&title=)
 。


#### 


 2.4 技术问题


技术问题主要是关于为什么库的某些代码以某种方式编写，或者代码的某一部分是做什么的。请确保链接到有问题的代码，并提供详细信息
为什么这部分代码很难理解。


您可以提出有关技术问题的问题
 [此处](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=bug&template=bug-report.yml)
 。


#### 


 2.5 添加新模型、调度程序或管道的建议


如果扩散模型社区发布了您希望在 Diffusers 库中看到的新模型、管道或调度程序，请提供以下信息：


* 扩散管道、模型或调度程序的简短描述以及论文或公开发布的链接。
* 链接到其任何开源实现。
* 模型权重的链接（如果有）。


如果您愿意自己为模型做出贡献，请告诉我们，以便我们最好地指导您。另外，别忘了
如果可以找到，请通过 GitHub 句柄标记组件（模型、调度程序、管道等）的原始作者。


您可以打开模型/管道/调度程序的请求
 [此处](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=New+model%2Fpipeline%2Fscheduler&template=new-model-addition.yml)
 。


### 


 3. 在 GitHub issues 选项卡上回答问题


在 GitHub 上回答问题可能需要一些 Diffusers 的技术知识，但我们鼓励每个人都尝试一下，即使您不能 100% 确定您的答案是正确的。
给出问题的高质量答案的一些技巧：


* 尽可能简洁和最小化。
* 紧扣主题。对问题的回答应该关注问题，并且只关注问题。
* 提供代码、论文或其他来源的链接，以证明或支持您的观点。
* 用代码回答。如果一个简单的代码片段可以解决问题或显示如何解决问题，请提供完全可重现的代码片段。


此外，许多问题往往只是偏离主题、与其他问题重复或无关紧要。这是伟大的
如果您可以回答此类问题，则可以为维护人员提供帮助，鼓励该问题的作者
更准确地说，提供重复问题的链接或将其重定向到
 [论坛](https://discuss.huggingface.co/c/discussion-lated-to-httpsgithubcomhuggingfacediffusers/63)
 或者
 [不和谐](https://discord.gg/G7tWnz98XR)
 。


如果您已验证发布的错误报告正确并需要在源代码中进行更正，
请看一下接下来的部分。


对于以下所有贡献，您将需要创建 PR。中详细解释了如何执行此操作
 [打开拉取请求](#how-to-open-a-pr)
 部分。


### 


 4. 解决“好第一个问题”


*好的第一个问题*
 被标记为
 [第一期好](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
 标签。通常，问题已经
解释了潜在的解决方案应该是什么样子，以便更容易修复。
如果问题尚未关闭，并且您想尝试解决此问题，可以留言“我想尝试一下这个问题”。通常有以下三种情况：


* a.) 问题描述已提出修复方案。在这种情况下，如果解决方案对您有意义，您可以打开 PR 或草稿 PR 来修复它。
* b.) 问题描述不建议修复。在这种情况下，您可以询问建议的修复方案，Diffusers 团队的人员应该很快就会回答。如果您知道如何修复它，请随时直接打开 PR。
* c.) 已经有一个公开的 PR 来解决该问题，但该问题尚未关闭。如果 PR 已过时，您只需打开一个新 PR 并链接到过时的 PR。如果想要解决问题的原始贡献者突然找不到时间继续处理，PR 通常会变得过时。这种情况在开源中经常发生，很正常。在这种情况下，如果您进行新的尝试并利用现有 PR 的知识，社区将会非常高兴。如果已经有 PR 并且处于活动状态，您可以通过提供建议、审查 PR 甚至询问您是否可以为 PR 做出贡献来帮助作者。


### 


 5. 为文档做出贡献


一个好的图书馆
 **总是**
 有很好的文档！官方文档通常是库的新用户的第一个接触点之一，因此为文档做出贡献是一个
 **高度
宝贵的贡献**
 。


对库的贡献可以有多种形式：


* 纠正拼写或语法错误。
* 更正文档字符串的错误格式。如果您发现官方文档显示奇怪或链接损坏，我们将非常高兴您花一些时间来纠正它。
* 更正文档字符串输入或输出张量的形状或尺寸。
* 澄清难以理解或不正确的文档。
* 更新过时的代码示例。
* 将文档翻译成另一种语言。


任何显示在
 [官方 Diffusers 文档页面](https://huggingface.co/docs/diffusers/index)
 是官方文档的一部分，可以在相应的地方进行更正、调整
 [文档来源](https://github.com/huggingface/diffusers/tree/main/docs/source)
 。


请看一下
 [本页](https://github.com/huggingface/diffusers/tree/main/docs)
 关于如何在本地验证对文档所做的更改。


### 


 6. 贡献社区管道


[管道](https://huggingface.co/docs/diffusers/api/pipelines/overview)
 通常是 Diffusers 库和用户之间的第一个接触点。
管道是如何使用扩散器的示例
 [模型](https://huggingface.co/docs/diffusers/api/models/overview)
 和
 [调度程序](https://huggingface.co/docs/diffusers/api/schedulers/overview)
 。
我们支持两种类型的管道：


* 官方管道
* 社区管道


官方和社区管道都遵循相同的设计并由相同类型的组件组成。


官方管道由 Diffusers 的核心维护者进行测试和维护。他们的代码
驻留在
 [src/diffusers/pipelines](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines)
 。
相比之下，社区管道纯粹由社区贡献和维护
 **社区**
 并且是
 **不是**
 已测试。
他们居住在
 [示例/社区](https://github.com/huggingface/diffusers/tree/main/examples/community)
 虽然它们可以通过
 [PyPI 扩散器包](https://pypi.org/project/diffusers/)
 ，他们的代码不属于 PyPI 发行版的一部分。


造成这种区别的原因是 Diffusers 库的核心维护者无法维护和测试所有
扩散模型可用于推理的可能方式，但其中一些可能是社区感兴趣的。
正式发布扩散管线，
例如稳定扩散被添加到核心 src/diffusers/pipelines 包中，确保
高质量的维护、无向后破坏的代码更改和测试。
应添加更多前沿管道作为社区管道。如果社区管道的使用率很高，可以根据社区的请求将管道移至官方管道。这是我们努力成为社区驱动的图书馆的方式之一。


要添加社区管道，应将 <name-of-the-community>.py 文件添加到
 [示例/社区](https://github.com/huggingface/diffusers/tree/main/examples/community)
 并调整
 [examples/community/README.md](https://github.com/huggingface/diffusers/tree/main/examples/community/README.md)
 包括新管道的示例。


可以看一个例子
 [此处](https://github.com/huggingface/diffusers/pull/2400)
 。


社区管道 PR 仅在表面上进行检查，理想情况下应由原作者维护。


贡献社区管道是了解 Diffusers 模型和调度程序如何工作的好方法。贡献社区管道通常是向社区贡献官方管道的第一步。
核心包。


### 


 7. 贡献培训示例


扩散器示例是位于以下位置的训练脚本的集合
 [示例](https://github.com/huggingface/diffusers/tree/main/examples)
 。


我们支持两种类型的训练示例：


* 官方训练示例
* 研究培训实例


研究培训示例位于
 [示例/研究\_projects](https://github.com/huggingface/diffusers/tree/main/examples/research_projects)
 而官方培训示例包括以下所有文件夹
 [示例](https://github.com/huggingface/diffusers/tree/main/examples)
 除了
 `研究项目`
 和
 `社区`
 文件夹。
官方培训示例由 Diffusers 的核心维护者维护，而研究培训示例由社区维护。
这是因为与中提出的相同原因
 [6.贡献社区管道](#6-contribute-a-community-pipeline)
 官方管道与社区管道：核心维护者维护扩散模型的所有可能的训练方法是不可行的。
如果 Diffusers 核心维护者和社区认为某种训练范式过于实验性或不够流行，则应将相应的训练代码放在
 `研究项目`
 文件夹并由作者维护。


官方培训和研究示例都包含一个目录，其中包含一个或多个培训脚本、requirements.txt 文件和 README.md 文件。为了让用户能够使用
训练示例，需要克隆存储库：



```
git clone https://github.com/huggingface/diffusers
```


以及安装训练所需的所有附加依赖项：



```
pip install -r /examples/<your-example-folder>/requirements.txt
```


因此，在添加示例时，
 `需求.txt`
 文件应定义训练示例所需的所有 pip 依赖项，以便安装所有这些依赖项后，用户可以运行示例的训练脚本。例如，参见
 [梦想展位
 `需求.txt`
 文件](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/requirements.txt)
 。


Diffusers 库的训练示例应遵循以下理念：


* 运行示例所需的所有代码应位于单个 Python 文件中。
* 人们应该能够从命令行运行该示例
 `python <您的示例>.py --args`
 。
* 示例应保持简单并作为
 **一个例子**
 关于如何使用扩散器进行培训。示例脚本的目的是
 **不是**
 创建最先进的扩散模型，而是在不添加太多自定义逻辑的情况下重现已知的训练方案。作为这一点的副产品，我们的示例也努力充当良好的教育材料。


要提供示例，强烈建议查看现有示例，例如
 [dreambooth](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)
 了解它们应该是什么样子。
我们强烈建议贡献者利用
 [加速库](https://github.com/huggingface/accelerate)
 因为它紧密集成
与扩散器。
一旦示例脚本起作用，请确保添加全面的
 `自述文件.md`
 这说明了如何准确使用该示例。本自述文件应包括：


* 有关如何运行示例脚本的示例命令，如图所示
 [此处](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth#running-locally-with-pytorch)
 。
* 一些训练结果（日志、模型等）的链接，显示用户可以期望的结果，如图所示
 [此处](https://api.wandb.ai/report/patrickvonplaten/xm6cd5q5)
 。
* 如果您要添加非官方/研究培训示例，
 **请不要忘记**
 添加一句您正在维护此训练示例，其中包括您的 git 句柄，如图所示
 [此处]（https://github.com/huggingface/diffusers/tree/main/examples/research_projects/intel_opts#diffusers-examples-with-intel-optimizations）
 。


如果您正在为官方培训示例做出贡献，还请确保添加测试
 [examples/test\_examples.py](https://github.com/huggingface/diffusers/blob/main/examples/test_examples.py)
 。对于非官方训练示例来说，这不是必需的。


### 


 8. 修复“好第二个问题”


*好第二期*
 被标记为
 [好第二期](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22Good+second+issue%22)
 标签。好的第二个问题是
通常比解决起来更复杂
 [好第一期](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
 。
问题描述通常对如何解决问题提供的指导较少，并且需要
感兴趣的贡献者对图书馆有很好的了解。
如果您有兴趣解决第二个问题，请随时打开 PR 来修复它，并将 PR 链接到该问题。如果您发现已为此问题打开 PR 但未合并，请查看以了解其未合并的原因并尝试打开改进的 PR。
与好的第一个问题相比，好的第二个问题通常更难合并，所以请毫不犹豫地向核心维护者寻求帮助。如果您的 PR 即将完成，核心维护者也可以跳入您的 PR 并对其进行承诺，以便将其合并。


### 


 9. 添加管道、模型、调度程序


管道、模型和调度程序是 Diffusers 库中最重要的部分。
它们可以轻松获取最先进的传播技术，从而使社区能够
构建强大的生成式人工智能应用程序。


通过添加新的模型、管道或调度程序，您可以为任何依赖 Diffuser 的用户界面启用新的强大用例，这对于整个生成 AI 生态系统具有巨大的价值。


Diffusers 对所有三个组件都有一些开放的功能请求 - 请随意掩盖它们
如果您还不知道要添加什么特定组件：


* [模型或管道](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
* [调度程序](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)


在添加这三个组件中的任何一个之前，强烈建议您给出
 [哲学指南]（哲学）
 阅读以更好地理解这三个组件中任何一个的设计。请注意
我们不能合并与我们的设计理念有很大差异的模型、调度程序或管道添加
因为这会导致 API 不一致。如果您从根本上不同意设计选择，请
打开一个
 [反馈问题](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feedback.md&title=)
 而是可以讨论某个设计是否
图书馆各处的图案/设计选择都会改变，我们是否会更新我们的设计理念。整个图书馆的一致性对我们来说非常重要。


请确保将原始代码库/论文的链接添加到 PR，并且最好还 ping
原作者直接在 PR 上，以便他们可以跟踪进度并可能帮助解决问题。


如果您不确定或陷入 PR 困境，请随时留言以寻求首次审核或帮助。


## 如何写出一篇好文章



**您的问题写得越好，快速解决的机会就越大。**


1. 确保您针对问题使用了正确的模板。您可以选择
 *错误报告*
 ,
 *功能要求*
 ,
 *有关API设计的反馈*
 ,
 *添加新模型/管道/调度程序*
 ,
 *论坛*
 ，或空白问题。打开时请务必选择正确的
 [一个新问题](https://github.com/huggingface/diffusers/issues/new/choose)
 。
2. **准确**
 ：给你的问题一个合适的标题。尝试尽可能简单地表述您的问题描述。提交问题时越精确，理解问题并解决问题所需的时间就越少。确保仅针对一个问题而不是针对多个问题打开问题。如果您发现多个问题，只需打开多个问题即可。如果您的问题是一个错误，请尝试尽可能准确地说明它是什么错误 - 您不应该只写“扩散器中的错误”。
3. **再现性**
 ：没有可重现的代码片段==没有解决方案。如果您遇到错误，维护人员
 **必须能够重现**
 它。确保包含可以复制粘贴到 Python 解释器中以重现问题的代码片段。确保您的代码片段有效，
 *IE。*
 没有缺少导入或缺少图像链接，...您的问题应该包含错误消息
 **和**
 可以复制粘贴而不进行任何更改以重现完全相同的错误消息的代码片段。如果您的问题是使用本地模型权重或读者无法访问的本地数据，则该问题无法解决。如果您无法共享数据或模型，请尝试制作虚拟模型或虚拟数据。
4. **简约**
 ：尽量保持简洁，尽可能帮助读者尽快理解问题。删除与问题无关的所有代码/所有信息。如果您发现了错误，请尝试创建最简单的代码示例来演示您的问题，不要在发现错误后立即将整个工作流程转储到问题中。例如，如果您训练一个模型并在训练过程中的某个时刻出现错误，您应该首先尝试了解训练代码的哪一部分导致了错误，并尝试用几行代码重现它。尝试使用虚拟数据而不是完整数据集。
5.添加链接。如果您引用某个命名、方法或模型，请确保提供链接，以便读者可以更好地理解您的意思。如果您指的是特定的 PR 或问题，请确保将其链接到您的问题。不要假设读者知道你在说什么。添加到问题中的链接越多越好。
6. 格式化。确保通过将代码格式化为 Python 代码语法，并将错误消息格式化为正常代码语法来很好地格式化您的问题。请参阅
 [官方 GitHub 格式化文档](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and -格式语法）
 了解更多信息。
7. 不要将您的问题视为需要解决的问题，而应将其视为一本写得很好的百科全书的漂亮条目。每个添加的问题都是对公开知识的贡献。通过添加一个写得好的问题，您不仅可以让维护人员更轻松地解决您的问题，而且可以帮助整个社区更好地理解库的某个方面。


## 如何撰写好的 PR



1. 成为变色龙。了解现有的设计模式和语法，并确保您添加的代码无缝地流入现有的代码库。与现有设计模式或用户界面显着不同的拉取请求将不会被合并。
2. 集中精力。拉取请求应该解决一个问题，并且只能解决一个问题。确保不要陷入“在添加问题的同时也解决另一个问题”的陷阱。审查同时解决多个不相关问题的拉取请求要困难得多。
3. 如果有帮助，请尝试添加一个代码片段，以显示如何使用添加的示例。
4. 拉取请求的标题应该是其贡献的摘要。
5. 如果您的拉取请求解决了问题，请在中提及问题编号
拉取请求描述以确保它们是链接的（并且人们
咨询该问题知道您正在解决该问题）；
6. 要表示正在进行的工作，请在标题前加上前缀
 `[WIP]`
 。这些
有助于避免重复工作，并将其与已准备好的 PR 区分开来
被合并；
7. 尝试按照中的说明来制定和格式化您的文本
 [如何写好期刊](#how-to-write-a-good-issue)
 。
8.确保现有测试通过；
9.添加高覆盖率测试。没有质量测试=没有合并。


* 如果您要添加新的
 `@慢`
 测试，确保它们通过使用
 `RUN_SLOW=1 python -m pytest 测试/test_my_new_model.py`
 。
CircleCI 不会运行缓慢的测试，但 GitHub Actions 每晚都会运行！


10. 所有公共方法都必须具有能够与 Markdown 很好地配合的信息丰富的文档字符串。看
 [`pipeline_latent_diffusion.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py)
 举个例子。
11. 由于存储库的快速增长，确保不添加任何会显着影响存储库的文件非常重要。这包括图像、视频和其他非文本文件。我们更喜欢利用 hf.co 托管
 `数据集`
 喜欢
 [`hf-internal-testing`](https://huggingface.co/hf-internal-testing)
 或者
 [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images)
 来放置这些文件。
如果是外部贡献，请随时将图像添加到您的 PR 并要求 Hugging Face 成员迁移您的图像
到这个数据集。


## 如何开设 PR



在编写代码之前，我们强烈建议您搜索现有的 PR 或
问题以确保没有人已经在做同样的事情。如果你是
不确定，提出问题以获得一些反馈总是一个好主意。


你将需要基本的
 `git`
 能够做出贡献的熟练程度
🧨 扩散器。
 `git`
 不是最容易使用的工具，但它具有最强大的功能
手动的。类型
 `git --help`
 装在壳里并享用。如果你更喜欢书籍，
 [专业版
Git](https://git-scm.com/book/en/v2)
 是一个很好的参考。


请按照以下步骤开始贡献（
 [支持的Python版本](https://github.com/huggingface/diffusers/blob/main/setup.py#L244)
 ）：


1.分叉
 [存储库](https://github.com/huggingface/diffusers)
 经过
单击存储库页面上的“分叉”按钮。这将创建代码的副本
在您的 GitHub 用户帐户下。
2. 将您的分支克隆到本地磁盘，并将基本存储库添加为远程存储库：



```
$ git clone git@github.com:<your GitHub handle>/diffusers.git
$ cd diffusers
$ git remote add upstream https://github.com/huggingface/diffusers.git
```
3. Create a new branch to hold your development changes:
 



```
$ git checkout -b a-descriptive-name-for-my-changes
```


**不要**
 致力于
 `主要`
 分支。


4. 在虚拟环境中运行以下命令来设置开发环境：



```
$ pip install -e ".[dev]"
```


如果您已经克隆了存储库，则可能需要
 `git pull`
 获取最新的变化
图书馆。


5. 开发您分支机构的功能。


当您处理这些功能时，您应该确保测试套件
通过。您应该运行受更改影响的测试，如下所示：



```
$ pytest tests/<TEST_TO_RUN>.py
```


在运行测试之前，请确保安装了测试所需的依赖项。你可以这样做
用这个命令：



```
$ pip install -e ".[test]"
```


您还可以使用以下命令运行完整的测试套件，但这需要
一台强大的机器可以在相当长的时间内产生结果
扩散器已经增长了很多。这是它的命令：



```
$ make test
```


🧨 扩散器依赖
 ‘黑色’
 和
 `分类`
 格式化其源代码
始终如一。进行更改后，应用自动样式更正和代码验证
无法一次性实现自动化：



```
$ make style
```


🧨 扩散器也使用
 `领子`
 以及一些用于检查编码错误的自定义脚本。质量
控制在 CI 中运行，但是，您也可以使用以下命令运行相同的检查：



```
$ make quality
```


一旦您对更改感到满意，请使用添加更改的文件
 `git 添加`
 和
做出承诺
 `git 提交`
 在本地记录您的更改：



```
$ git add modified_file.py
$ git commit -m "A descriptive message about your changes."
```


将代码副本与原始代码同步是个好主意
定期存储。这样您就可以快速考虑变化：



```
$ git pull upstream main
```


使用以下方式将更改推送到您的帐户：



```
$ git push -u origin a-descriptive-name-for-my-changes
```


6. 满意后，转到
您的 fork 在 GitHub 上的网页。单击“拉取请求”发送您的更改
供项目维护者审查。
7. 如果维护人员要求您进行更改也没关系。它发生在核心贡献者身上
也！这样每个人都可以看到 Pull 请求中的更改，在您本地工作
分支并将更改推送到您的分支。它们会自动出现在
拉取请求。


### 


 测试


包含一个广泛的测试套件来测试库行为和几个示例。库测试可以在以下位置找到
这
 [测试文件夹](https://github.com/huggingface/diffusers/tree/main/tests)
 。


我们喜欢
 `pytest`
 和
 `pytest-xdist`
 因为它更快。从根源上来说
存储库，以下是如何运行测试
 `pytest`
 对于图书馆：



```
$ python -m pytest -n auto --dist=loadfile -s -v ./tests/
```


事实上，就是这样
 `进行测试`
 已实施！


您可以指定较小的测试集以仅测试该功能
你正在努力。


默认情况下，会跳过慢速测试。设置
 `运行_慢`
 环境变量为
 `是的`
 运行它们。这将下载许多 GB 的模型 - 确保您
有足够的磁盘空间和良好的互联网连接，或者有足够的耐心！



```
$ RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./tests/
```


`单元测试`
 完全支持，以下是如何使用它运行测试：



```
$ python -m unittest discover -s tests -t . -v
$ python -m unittest discover -s examples -t examples -v
```


### 


 将分叉主程序与上游 (HuggingFace) 主程序同步


为了避免 ping 上游存储库，该存储库会向每个上游 PR 添加参考注释并向参与这些 PR 的开发人员发送不必要的通知，
同步分叉存储库的主分支时，请按照以下步骤操作：


1. 如果可能，请避免使用分叉存储库上的分支和 PR 与上游同步。相反，直接合并到分叉的主干中。
2. 如果绝对需要 PR，请在检查您的分支机构后执行以下步骤：



```
$ git checkout -b your-branch-for-syncing
$ git pull --squash --no-commit upstream main
$ git commit -m '<your message without GitHub references>'
$ git push --set-upstream origin your-branch-for-syncing
```


### 


 时尚指南


对于文档字符串，🧨 Diffusers 遵循
 [谷歌风格](https://google.github.io/styleguide/pyguide.html)
 。