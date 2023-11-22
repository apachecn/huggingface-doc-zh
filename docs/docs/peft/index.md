# 聚四氟乙烯

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/index>
>
> 原始地址：<https://huggingface.co/docs/peft/index>


🤗 PEFT，或参数高效微调（PEFT），是一个库，用于有效地将预训练的语言模型（PLM）适应各种下游应用程序，而无需微调所有模型的参数。
PEFT 方法仅微调少量（额外）模型参数，从而显着降低计算和存储成本，因为微调大规模 PLM 的成本过高。
最近最先进的 PEFT 技术实现了与完全微调相当的性能。


PEFT 与 🤗 Accelerate 无缝集成，可利用 DeepSpeed 和
 [大模型推理](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
 。


[开始吧
 

 如果您是 🤗 PEFT 的新手，请从这里开始，概述该库的主要功能，以及如何使用 PEFT 方法训练模型。](quicktour)
[操作指南
 

 实用指南演示如何在不同类型的任务中应用各种 PEFT 方法，例如图像分类、因果语言建模、自动语音识别等。了解如何将 🤗 PEFT 与 DeepSpeed 和完全分片数据并行脚本结合使用。](./task_guides/image_classification_lora)
[概念指南
 

 更好地从理论上理解 LoRA 和各种软提示方法如何帮助减少可训练参数的数量，从而提高训练效率。](./conceptual_guides/lora)
[参考
 

 🤗 PEFT 类和方法如何工作的技术描述。](./package_reference/config)


## 支持的方法



1.洛拉：
 [LORA：大语言模型的低阶适应](https://arxiv.org/pdf/2106.09685.pdf)
2. 前缀调优：
 [前缀调优：优化生成连续提示](https://aclanthology.org/2021.acl-long.353/)
 ,
 [P-Tuning v2：快速调整可以与跨尺度和任务的通用微调相媲美](https://arxiv.org/pdf/2110.07602.pdf)
3. P 调音：
 [GPT 也能理解](https://arxiv.org/pdf/2103.10385.pdf)
4. 及时调整：
 [参数高效提示调整的规模力量](https://arxiv.org/pdf/2104.08691.pdf)
5.阿达洛拉：
 [参数高效微调的自适应预算分配](https://arxiv.org/abs/2303.10512)
6. [LLaMA-Adapter：零初始注意力的语言模型高效微调](https://github.com/ZrrSkywalker/LLaMA-Adapter)
7.IA3：
 [通过抑制和放大内部激活来注入适配器](https://arxiv.org/abs/2205.05638)


## 支持机型



下表列出了每项任务支持的 PEFT 方法和模型。应用特定的 PEFT 方法
任务，请参阅相应的任务指南。


### 


 因果语言建模


| 	 Model	  | 	 LoRA	  | 	 Prefix Tuning	  | 	 P-Tuning	  | 	 Prompt Tuning	  | 	 IA3	  |
| --- | --- | --- | --- | --- | --- |
| 	 GPT-2	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |
| 	 Bloom	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |
| 	 OPT	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |
| 	 GPT-Neo	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |
| 	 GPT-J	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |
| 	 GPT-NeoX-20B	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |
| 	 LLaMA	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |
| 	 ChatGLM	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |


### 


 条件生成


| 	 Model	  | 	 LoRA	  | 	 Prefix Tuning	  | 	 P-Tuning	  | 	 Prompt Tuning	  | 	 IA3	  |
| --- | --- | --- | --- | --- | --- |
| 	 T5	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |
| 	 BART	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |


### 


 序列分类


| 	 Model	  | 	 LoRA	  | 	 Prefix Tuning	  | 	 P-Tuning	  | 	 Prompt Tuning	  | 	 IA3	  |
| --- | --- | --- | --- | --- | --- |
| 	 BERT	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |
| 	 RoBERTa	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |
| 	 GPT-2	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |  |
| 	 Bloom	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |  |
| 	 OPT	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |  |
| 	 GPT-Neo	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |  |
| 	 GPT-J	  | 	 ✅	  | 	 ✅	  | 	 ✅	  | 	 ✅	  |  |
| 	 Deberta	  | 	 ✅	  |  | 	 ✅	  | 	 ✅	  |  |
| 	 Deberta-v2	  | 	 ✅	  |  | 	 ✅	  | 	 ✅	  |  |


### 


 代币分类


| 	 Model	  | 	 LoRA	  | 	 Prefix Tuning	  | 	 P-Tuning	  | 	 Prompt Tuning	  | 	 IA3	  |
| --- | --- | --- | --- | --- | --- |
| 	 BERT	  | 	 ✅	  | 	 ✅	  |  |  |  |
| 	 RoBERTa	  | 	 ✅	  | 	 ✅	  |  |  |  |
| 	 GPT-2	  | 	 ✅	  | 	 ✅	  |  |  |  |
| 	 Bloom	  | 	 ✅	  | 	 ✅	  |  |  |  |
| 	 OPT	  | 	 ✅	  | 	 ✅	  |  |  |  |
| 	 GPT-Neo	  | 	 ✅	  | 	 ✅	  |  |  |  |
| 	 GPT-J	  | 	 ✅	  | 	 ✅	  |  |  |  |
| 	 Deberta	  | 	 ✅	  |  |  |  |  |
| 	 Deberta-v2	  | 	 ✅	  |  |  |  |  |


### 


 文本到图像的生成


| 	 Model	  | 	 LoRA	  | 	 Prefix Tuning	  | 	 P-Tuning	  | 	 Prompt Tuning	  | 	 IA3	  |
| --- | --- | --- | --- | --- | --- |
| 	 Stable Diffusion	  | 	 ✅	  |  |  |  |  |


### 


 图像分类


| 	 Model	  | 	 LoRA	  | 	 Prefix Tuning	  | 	 P-Tuning	  | 	 Prompt Tuning	  | 	 IA3	  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 	 ViT	  | 	 ✅	  |  |  |  |  |  |
| 	 Swin	  | 	 ✅	  |  |  |  |  |  |


### 


 图像到文本（多模态模型）


我们已经测试了 LoRA
 [ViT](https://huggingface.co/docs/transformers/model_doc/vit)
 和
 [Swin](https://huggingface.co/docs/transformers/model_doc/swin)
 用于图像分类的微调。
然而，LoRA 应该可以用于任何
 [基于 ViT 的模型](https://huggingface.co/models?pipeline_tag=image-classification&sort=downloads&search=vit)
 来自🤗变形金刚。
查看
 [图像分类](/task_guides/image_classification_lora)
 任务指南以了解更多信息。如果您遇到问题，请提出问题。


| 	 Model	  | 	 LoRA	  | 	 Prefix Tuning	  | 	 P-Tuning	  | 	 Prompt Tuning	  | 	 IA3	  |
| --- | --- | --- | --- | --- | --- |
| 	 Blip-2	  | 	 ✅	  |  |  |  |  |


### 


 语义分割


与图像到文本模型一样，您应该能够将 LoRA 应用于任何
 [分割模型](https://huggingface.co/models?pipeline_tag=image-segmentation&sort=downloads)
 。
值得注意的是，我们尚未针对每种架构进行测试。因此，如果您遇到任何问题，请创建问题报告。


| 	 Model	  | 	 LoRA	  | 	 Prefix Tuning	  | 	 P-Tuning	  | 	 Prompt Tuning	  | 	 IA3	  |
| --- | --- | --- | --- | --- | --- |
| 	 SegFormer	  | 	 ✅	  |  |  |  |  |