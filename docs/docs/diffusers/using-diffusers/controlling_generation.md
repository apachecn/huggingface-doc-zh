# 受控发电

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/controlling_generation>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/controlling_generation>


控制扩散模型产生的输出长期以来一直是社区所追求的，现在是一个活跃的研究课题。在许多流行的扩散模型中，输入（图像和文本提示）的细微变化可以极大地改变输出。在理想的世界中，我们希望能够控制语义的保存和更改方式。


大多数保留语义的例子都简化为能够准确地将输入的变化映射到输出的变化。 IE。在提示中向主题添加形容词会保留整个图像，仅修改更改的主题。或者，特定主体的图像变化保留了主体的姿势。


此外，我们还希望影响生成图像的语义保存以外的质量。 IE。一般来说，我们希望我们的输出具有良好的质量，坚持特定的风格，或者是现实的。


我们将记录一些技术
 `扩散器`
 支持控制扩散模型的生成。很多都是前沿研究，而且可能非常微妙。如果有什么需要澄清或者您有建议，请随时就该问题展开讨论
 [论坛](https://discuss.huggingface.co/c/discussion-lated-to-httpsgithubcomhuggingfacediffusers/63)
 或一个
 [GitHub 问题](https://github.com/huggingface/diffusers/issues)
 。


我们提供了如何控制发电的高级解释以及技术片段。对于更深入的技术解释，从管道链接的原始论文始终是最好的资源。


根据用例，应该相应地选择一种技术。在许多情况下，这些技术可以结合起来。例如，可以将 Textual Inversion 与 SEGA 结合起来，为使用 Textual Inversion 生成的输出提供更多语义指导。


除非另有说明，这些技术适用于现有模型，并且不需要自己的权重。


1. [InstructPix2Pix](#instruct-pix2pix)
2. [Pix2Pix 零](#pix2pix-zero)
3. [参加并激发](#attend-and-excite)
4. [语义指导](#semantic-guidance-sega)
5. [自我注意力指导](#self-attention-guidance-sag)
6. [深度2图像](#深度2图像)
7. [多重扩散全景](#多重扩散-全景)
8.[DreamBooth](#dreambooth)
9. [文本倒置](#textual-inversion)
10. [控制网络](#controlnet)
11. [提示加权](#prompt-weighting)
12.[自定义扩散](#custom-diffusion)
13. [模型编辑](#model-editing)
14. [差异编辑](#diffedit)
15. [T2I-适配器](#t2i-适配器)
16. [面料](#fabric)


为了方便起见，我们提供了一个表格来表示哪些方法仅用于推理，哪些方法需要微调/训练。


| **Method**  | **Inference only**  | **Requires training /	 	 fine-tuning**  | **Comments**  |
| --- | --- | --- | --- |
| [InstructPix2Pix](#instruct-pix2pix)  | 	 ✅	  | 	 ❌	  | 	 Can additionally be	 	 fine-tuned for better	 	 performance on specific	 	 edit instructions.	  |
| [Pix2Pix Zero](#pix2pix-zero)  | 	 ✅	  | 	 ❌	  |  |
| [Attend and Excite](#attend-and-excite)  | 	 ✅	  | 	 ❌	  |  |
| [Semantic Guidance](#semantic-guidance-sega)  | 	 ✅	  | 	 ❌	  |  |
| [Self-attention Guidance](#self-attention-guidance-sag)  | 	 ✅	  | 	 ❌	  |  |
| [Depth2Image](#depth2image)  | 	 ✅	  | 	 ❌	  |  |
| [MultiDiffusion Panorama](#multidiffusion-panorama)  | 	 ✅	  | 	 ❌	  |  |
| [DreamBooth](#dreambooth)  | 	 ❌	  | 	 ✅	  |  |
| [Textual Inversion](#textual-inversion)  | 	 ❌	  | 	 ✅	  |  |
| [ControlNet](#controlnet)  | 	 ✅	  | 	 ❌	  | 	 A ControlNet can be	 	 trained/fine-tuned on	 	 a custom conditioning.	  |
| [Prompt Weighting](#prompt-weighting)  | 	 ✅	  | 	 ❌	  |  |
| [Custom Diffusion](#custom-diffusion)  | 	 ❌	  | 	 ✅	  |  |
| [Model Editing](#model-editing)  | 	 ✅	  | 	 ❌	  |  |
| [DiffEdit](#diffedit)  | 	 ✅	  | 	 ❌	  |  |
| [T2I-Adapter](#t2i-adapter)  | 	 ✅	  | 	 ❌	  |  |
| [Fabric](#fabric)  | 	 ✅	  | 	 ❌	  |  |


## 指导Pix2Pix



[论文](https://arxiv.org/abs/2211.09800)


[InstructPix2Pix](../api/pipelines/pix2pix)
 从稳定扩散进行微调以支持编辑输入图像。它将图像和描述编辑的提示作为输入，然后输出编辑后的图像。
InstructPix2Pix 经过明确培训，可以很好地与
 [指导GPT](https://openai.com/blog/instruction-following/)
 - 类似提示。


## 像素2像素零



[论文](https://arxiv.org/abs/2302.03027)


[Pix2Pix 零](../api/pipelines/pix2pix_zero)
 允许修改图像，以便将一个概念或主题转换为另一个概念或主题，同时保留一般图像语义。


去噪过程从一种概念嵌入引导到另一种概念嵌入。在去噪过程中优化中间潜伏，将注意力图推向参考注意力图。参考注意力图来自输入图像的去噪过程，用于鼓励语义保留。


Pix2Pix Zero 既可用于编辑合成图像，也可用于编辑真实图像。


* 要编辑合成图像，首先生成一个带有标题的图像。
接下来，我们为要编辑的概念和新的目标概念生成图像标题。我们可以使用这样的模型
 [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)
 以此目的。然后，通过文本编码器创建源概念和目标概念的“平均”提示嵌入。最后使用pix2pix-zero算法对合成图像进行编辑。
* 要编辑真实图像，首先使用如下模型生成图像标题
 [BLIP](https://huggingface.co/docs/transformers/model_doc/blip)
 。然后，对提示和图像应用 DDIM 反转以生成“反转”潜在变量。与之前类似，创建源概念和目标概念的“平均”提示嵌入，最后使用 pix2pix-zero 算法与“逆”潜伏相结合来编辑图像。


Pix2Pix Zero 是第一个允许“零镜头”图像编辑的模型。这意味着该模型
可以在消费级 GPU 上用不到一分钟的时间编辑图像，如图所示
 [此处](../api/pipelines/pix2pix_zero#usage-example)
 。


如上所述，Pix2Pix Zero 包括优化潜在变量（而不是 UNet、VAE 或文本编码器中的任何一个），以引导生成过程朝着特定的概念发展。这意味着总体
管道可能需要比标准更多的内存
 [StableDiffusionPipeline](../api/pipelines/stable_diffusion/text2img)
 。


InstructPix2Pix 和 Pix2Pix Zero 等方法之间的一个重要区别是前者
涉及微调预训练的权重，而后者则不然。这意味着您可以
将 Pix2Pix Zero 应用于任何可用的稳定扩散模型。


## 参加并兴奋



[论文](https://arxiv.org/abs/2301.13826)


[参加并激发](../api/pipelines/attend_and_excite)
 允许提示中的主题在最终图像中忠实地呈现。


给出一组标记索引作为输入，对应于提示中需要出现在图像中的主题。在去噪期间，保证每个标记索引对于图像的至少一个块具有最小注意阈值。在去噪过程中迭代优化中间潜伏，以加强最被忽视的主题标记的注意力，直到所有主题标记都超过注意力阈值。


与 Pix2Pix Zero 一样，Attend 和 Excite 也在其管道中涉及一个小型优化循环（保持预先训练的权重不变），并且可能需要比平常更多的内存
 [StableDiffusionPipeline](../api/pipelines/stable_diffusion/text2img)
 。


## 语义指导（世嘉）



[论文](https://arxiv.org/abs/2301.12247)


[世嘉](../api/pipelines/semantic_stable_diffusion)
 允许从图像中应用或删除一个或多个概念。概念的强度也是可以控制的。 IE。微笑概念可用于逐渐增加或减少肖像的微笑。


与无分类器指导通过空提示输入提供指导类似，SEGA 提供概念提示指导。可以同时应用多个这些概念提示。每个概念提示都可以添加或删除其概念，具体取决于指导是积极应用还是消极应用。


与 Pix2Pix Zero 或 attend and Excite 不同，SEGA 直接与扩散过程交互，而不是执行任何显式的基于梯度的优化。


## 自注意力引导（SAG）



[论文](https://arxiv.org/abs/2210.00939)


[自我注意力指导](../api/pipelines/self_attention_guidance)
 提高图像的总体质量。


SAG 提供从不以高频细节为条件的预测到完全条件图像的指导。从 UNet 自注意力图中提取高频细节。


## 深度2图像



[项目](https://huggingface.co/stabilityai/stable-diffusion-2-深度)


[Depth2Image](../api/pipelines/stable_diffusion/depth2img)
 从稳定扩散进行微调，以更好地保留文本引导图像变化的语义。


它以原始图像的单眼深度估计为条件。


## 多重扩散全景



[论文](https://arxiv.org/abs/2302.08113)


[多重扩散全景](../api/pipelines/panorama)
 在预先训练的扩散模型上定义了新一代过程。该过程将多种扩散生成方法结合在一起，可以轻松应用于生成高质量和多样化的图像。结果遵循用户提供的控制，例如所需的纵横比（例如全景）和空间引导信号，范围从紧密的分割掩模到边界框。
MultiDiffusion Panorama 允许生成任意纵横比的高质量图像（例如全景图）。


## 微调您自己的模型



除了预先训练的模型之外，Diffusers 还具有用于根据用户提供的数据微调模型的训练脚本。


## 梦想展位



[项目](https://dreambooth.github.io/)


[DreamBooth](../培训/dreambooth)
 微调模型以教授新主题。 IE。一个人的几张照片可以用来生成该人不同风格的图像。


## 文本倒装



[论文](https://arxiv.org/abs/2208.01618)


[文本倒置](../training/text_inversion)
 微调模型以教授新概念。 IE。某种艺术风格的几张图片可用于生成该风格的图像。


## 控制网



[论文](https://arxiv.org/abs/2302.05543)


[ControlNet](../api/pipelines/controlnet)
 是一个添加了额外条件的辅助网络。
有 8 个规范的预训练 ControlNet，在不同的条件下进行训练，例如边缘检测、涂鸦、
深度图和语义分割。


## 提示加权



[提示权重](../using-diffusers/weighted_prompts)
 是一种简单的技术，可以将更多注意力放在文本的某些部分上
输入。


## 定制扩散



[论文](https://arxiv.org/abs/2212.04488)


[自定义扩散](../training/custom_diffusion)
 仅微调预训练的交叉注意力图
文本到图像的扩散模型。它还允许额外执行文本反转。它支持
设计多概念培训。与 DreamBooth 和 Textual Inversion 一样，自定义扩散也用于
教授预先训练的文本到图像扩散模型有关新概念的知识，以生成涉及以下内容的输出：
感兴趣的概念。


## 模型编辑



[论文](https://arxiv.org/abs/2303.08084)


这
 [文本到图像模型编辑管道](../api/pipelines/model_editing)
 帮助您减轻预训练文本到图像的一些不正确的隐式假设
扩散模型可能会影响输入提示中出现的主题。例如，如果提示稳定扩散生成“一包玫瑰”的图像，则生成图像中的玫瑰
更可能是红色的。该管道可以帮助您改变这种假设。


## 差异编辑



[论文](https://arxiv.org/abs/2210.11427)


[差异编辑](../api/pipelines/diffedit)
 允许对输入图像进行语义编辑
输入提示同时尽可能保留原始输入图像。


## T2I 适配器



[论文](https://arxiv.org/abs/2302.08453)


[T2I-适配器](../api/pipelines/stable_diffusion/适配器)
 是一个添加了额外条件的辅助网络。
有 8 个规范的预训练适配器，经过不同条件的训练，例如边缘检测、草图、
深度图和语义分割。


## 织物



[论文](https://arxiv.org/abs/2307.10159)


[结构](https://github.com/huggingface/diffusers/tree/442017ccc877279bcf24fbe92f92d3d0def191b6/examples/community#stable-diffusion-fabric-pipeline)
 是免培训的
该方法适用于各种流行的扩散模型，该模型利用
最广泛使用的架构中存在的自注意力层来调节
一组反馈图像的扩散过程。