# 🧨 扩香者的道德准则

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/conceptual/ethical_guidelines>
>
> 原始地址：<https://huggingface.co/docs/diffusers/conceptual/ethical_guidelines>


## 前言



[扩散器](https://huggingface.co/docs/diffusers/index)
 提供预训练的扩散模型，并作为推理和训练的模块化工具箱。


鉴于其在全球的真实案例应用以及对社会的潜在负面影响，我们认为为该项目提供道德准则来指导 Diffusers 库的开发、用户贡献和使用非常重要。


与使用该技术相关的风险仍在研究中，但仅举几例：艺术家的版权问题；深度造假利用；在不适当的情况下生成色情内容；未经同意的冒充；有害的社会偏见使边缘化群体长期受到压迫。
我们将持续跟踪风险，并根据社区的响应和宝贵反馈调整以下指南。


## 范围



Diffusers 社区将在项目的开发中应用以下道德准则，并帮助协调社区如何整合贡献，特别是涉及与道德问题相关的敏感话题。


## 道德准则



以下道德准则普遍适用，但我们将主要在处理道德敏感问题并做出技术选择时实施它们。此外，我们承诺在出现与相关技术最新水平相关的危害后，随着时间的推移调整这些道德原则。


* **透明度**
 ：我们致力于在管理 PR、向用户解释我们的选择以及做出技术决策方面保持透明。
* **一致性**
 ：我们致力于保证我们的用户在项目管理中得到同等程度的关注，保持技术上的稳定和一致。
* **简单**
 ：为了让 Diffusers 库易于使用和利用，我们致力于保持项目目标的精简和连贯。
* **辅助功能**
 ：Diffusers 项目有助于降低贡献者的进入门槛，即使没有技术专业知识，他们也可以帮助运行该项目。这样做可以使社区更容易获得研究成果。
* **再现性**
 ：我们的目标是在通过 Diffusers 库提供的上游代码、模型和数据集的可重复性方面保持透明。
* **责任**
 ：作为一个社区，通过团队合作，我们对用户承担集体责任，预测并减轻该技术的潜在风险和危险。


## 实施示例：安全功能和机制



该团队每天致力于提供技术和非技术工具，以应对与扩散技术相关的潜在道德和社会风险。此外，社区的意见对于确保这些功能的实施和提高我们的认识非常宝贵。


* [**社区选项卡**](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)
 ：它使社区能够讨论并更好地协作项目。
* **偏差探索和评估**
 ：Hugging Face 团队提供了
 [空间](https://huggingface.co/spaces/society-ethics/DiffusionBiasExplorer)
 以交互方式演示稳定扩散中的偏差。从这个意义上说，我们支持和鼓励偏见探索者和评估。
* **鼓励部署安全**


+ [**安全稳定扩散**](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_safe)
：它缓解了一个众所周知的问题，即在未经过滤的网络爬行数据集上训练的模型（例如稳定扩散）往往会遭受不适当的退化。相关论文：
[安全潜在扩散：减轻扩散模型中的不当退化](https://arxiv.org/abs/2211.05105)
。
+ [**安全检查器**](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py)
：它在生成图像后检查并比较嵌入空间中一组硬编码有害概念与图像的类别概率。故意隐藏有害概念以防止检查器的逆向工程。
* **在 Hub 上分阶段发布**
 ：在特别敏感的情况下，应限制对某些存储库的访问。此分阶段发布是一个中间步骤，允许存储库的作者对其使用有更多的控制。
* **许可**
 :
 [OpenRAIL](https://huggingface.co/blog/open_rail)
 ，一种新型许可，允许我们确保免费访问，同时具有一系列限制以确保更负责任的使用。