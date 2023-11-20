# 如何使用 Core ML 运行稳定扩散

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/optimization/coreml>
>
> 原始地址：<https://huggingface.co/docs/diffusers/optimization/coreml>


[核心机器学习](https://developer.apple.com/documentation/coreml)
 是Apple框架支持的模型格式和机器学习库。如果您有兴趣在 macOS 或 iOS/iPadOS 应用程序中运行稳定扩散模型，本指南将向您展示如何将现有 PyTorch 检查点转换为 Core ML 格式，并使用它们通过 Python 或 Swift 进行推理。


Core ML 模型可以利用 Apple 设备中提供的所有计算引擎：CPU、GPU 和 Apple 神经引擎（或 ANE，Apple Silicon Mac 和现代 iPhone/iPad 中提供的张量优化加速器）。根据模型及其运行的设备，Core ML 也可以混合和匹配计算引擎，因此模型的某些部分可能在 CPU 上运行，而其他部分可能在 GPU 上运行。


您还可以运行
 `扩散器`
 Apple Silicon Mac 上的 Python 代码库使用
 `议员`
 PyTorch 内置加速器。这种方法在
 [议员指南](议员)
 ，但它与本机应用程序不兼容。


## 稳定扩散核心 ML 检查点



稳定扩散权重（或检查点）以 PyTorch 格式存储，因此您需要将它们转换为 Core ML 格式，然后才能在本机应用程序中使用它们。


值得庆幸的是，苹果工程师开发了
 [转换工具](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml)
 基于
 `扩散器`
 将 PyTorch 检查点转换为 Core ML。


不过，在转换模型之前，请花点时间探索一下 Hugging Face Hub - 您感兴趣的模型很可能已经以 Core ML 格式提供：


* 这
 [苹果](https://huggingface.co/apple)
 组织包括稳定扩散版本 1.4、1.5、2.0 基础和 2.1 基础
* [coreml](https://huggingface.co/coreml)
 组织包括自定义 DreamBoothed 和微调模型
* 用这个
 [过滤器](https://huggingface.co/models?pipeline_tag=text-to-image&library=coreml&p=2&sort=likes)
 返回所有可用的 Core ML 检查点


如果您找不到您感兴趣的型号，我们建议您按照以下说明进行操作
 [将模型转换为 Core ML](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml)
 由苹果公司。


## 选择要使用的 Core ML 变体



稳定扩散模型可以转换为用于不同目的的不同 Core ML 变体：


* 使用的注意力块类型。注意操作用于“注意”图像表示中不同区域之间的关系，并了解图像和文本表示如何相关。注意力是计算和内存密集型的，因此存在考虑不同设备的硬件特性的不同实现。对于 Core ML 稳定扩散模型，有两种注意力变体：


+ `split_einsum`
（
[苹果公司推出](https://machinelearning.apple.com/research/neural-engine-transformers)
）针对 ANE 设备进行了优化，可在现代 iPhone、iPad 和 M 系列计算机中使用。
+“原始”注意力（用于的基本实现
`扩散器`
) 仅与 CPU/GPU 兼容，与 ANE 不兼容。有可能
*快点*
使用以下命令在 CPU + GPU 上运行模型
`原来的`
关注度高于ANE。看
[此性能基准](https://huggingface.co/blog/fast-mac-diffusers#performance-benchmarks)
以及一些
[社区提供的额外措施](https://github.com/huggingface/swift-coreml-diffusers/issues/31)
了解更多详情。
* 支持的推理框架。


+ `包`
适合Python推理。这可用于在尝试将转换后的 Core ML 模型集成到本机应用程序之前测试它们，或者如果您想探索 Core ML 性能但不需要支持本机应用程序。例如，具有 Web UI 的应用程序可以完美地使用 Python Core ML 后端。
+ `已编译`
Swift 代码需要模型。这
`已编译`
Hub 中的模型将大型 UNet 模型权重拆分为多个文件，以便与 iOS 和 iPadOS 设备兼容。这对应于
[`--chunk-unet`
转换选项](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml)
。如果您想支持本机应用程序，那么您需要选择
`已编译`
变体。


官方 Core ML 稳定扩散
 [模型](https://huggingface.co/apple/coreml-stable-diffusion-v1-4/tree/main)
 包括这些变体，但社区的变体可能有所不同：



```
coreml-stable-diffusion-v1-4
├── README.md
├── original
│   ├── compiled
│   └── packages
└── split_einsum
    ├── compiled
    └── packages
```


您可以下载并使用您需要的变体，如下所示。


## Python 中的核心 ML 推理



安装以下库以在 Python 中运行 Core ML 推理：



```
pip install huggingface_hub
pip install git+https://github.com/apple/ml-stable-diffusion
```


### 


 下载模型检查点


要在 Python 中运行推理，请使用存储在
 `包`
 文件夹，因为
 `已编译`
 仅与 Swift 兼容。您可以选择是否要使用
 `原来的`
 或者
 `split_einsum`
 注意力。


这就是您下载的方式
 `原来的`
 注意从 Hub 到名为的目录的变体
 `模型`
 ：



```
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/packages"

model_path = Path("./models") / (repo_id.split("/")[-1] + "\_" + variant.replace("/", "\_"))
snapshot_download(repo_id, allow_patterns=f"{variant}/\*", local_dir=model_path, local_dir_use_symlinks=False)
print(f"Model downloaded at {model\_path}")
```


### 


 推理


下载模型快照后，您可以使用 Apple 的 Python 脚本对其进行测试。



```
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" -i models/coreml-stable-diffusion-v1-4_original_packages -o </path/to/output/image> --compute-unit CPU_AND_GPU --seed 93
```


`<输出-mlpackages-目录>`
 应指向您在上述步骤中下载的检查点，并且
 `--计算单元`
 指示您想要允许进行推理的硬件。它必须是以下选项之一：
 `全部`
 ,
 `CPU_AND_GPU`
 ,
 `仅CPU`
 ,
 `CPU_AND_NE`
 。您还可以提供可选的输出路径和可重复性的种子。


推理脚本假设您使用的是稳定扩散模型的原始版本，
 `CompVis/stable-diffusion-v1-4`
 。如果您使用其他型号，您
 *有*
 要在推理命令行中指定其集线器 ID，请使用
 `--模型版本`
 选项。这适用于已支持的模型以及您自己训练或微调的自定义模型。


例如，如果您想使用
 [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 ：



```
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" --compute-unit ALL -o output --seed 93 -i models/coreml-stable-diffusion-v1-5_original_packages --model-version runwayml/stable-diffusion-v1-5
```


## Swift 中的核心 ML 推理



在 Swift 中运行推理比在 Python 中运行稍快，因为模型已经在
 `mlmodelc`
 格式。当加载模型时，这在应用程序启动时很明显，但如果您随后运行几代，则应该不会明显。


### 


 下载


要在 Mac 上的 Swift 中运行推理，您需要以下之一
 `已编译`
 检查点版本。我们建议您使用与上一个示例类似的 Python 代码在本地下载它们，但使用其中之一
 `已编译`
 变种：



```
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/compiled"

model_path = Path("./models") / (repo_id.split("/")[-1] + "\_" + variant.replace("/", "\_"))
snapshot_download(repo_id, allow_patterns=f"{variant}/\*", local_dir=model_path, local_dir_use_symlinks=False)
print(f"Model downloaded at {model\_path}")
```


### 


 推理


要运行推理，请克隆 Apple 的存储库：



```
git clone https://github.com/apple/ml-stable-diffusion
cd ml-stable-diffusion
```


然后使用Apple的命令行工具，
 [Swift 包管理器](https://www.swift.org/package-manager/#)
 ：



```
swift run StableDiffusionSample --resource-path models/coreml-stable-diffusion-v1-4_original_compiled --compute-units all "a photo of an astronaut riding a horse on mars"
```


您必须指定
 `--资源路径`
 上一步中下载的检查点之一，因此请确保它包含带有扩展名的已编译 Core ML 包
 `.mlmodelc`
 。这
 `--计算单元`
 必须是以下值之一：
 `全部`
 ,
 `仅 cpu`
 ,
 `cpu和GPU`
 ,
 `cpu和神经引擎`
 。


欲了解更多详情，请参阅
 [Apple 仓库中的说明](https://github.com/apple/ml-stable-diffusion)
 。


## 支持的扩散器功能



Core ML 模型和推理代码不支持 🧨 Diffusers 的许多功能、选项和灵活性。以下是一些需要记住的限制：


* Core ML 模型仅适用于推理。它们不能用于训练或微调。
* 只有两个调度器被移植到 Swift，Stable Diffusion 使用的默认调度器和
 `DPM求解器多步调度器`
 ，我们将其从我们的
 `扩散器`
 执行。我们建议您使用
 `DPM求解器多步调度器`
 ，因为它以大约一半的步骤产生相同的质量。
* 推理代码中提供了否定提示、无分类器指导量表和图像到图像任务。深度引导、ControlNet 和潜在升级器等高级功能尚不可用。


苹果
 [转换和推理仓库](https://github.com/apple/ml-stable-diffusion)
 和我们自己的
 [swift-coreml-diffusers](https://github.com/huggingface/swift-coreml-diffusers)
 存储库旨在作为技术演示器，使其他开发人员能够在此基础上进行构建。


如果您强烈认为缺少任何功能，请随时提出功能请求，或者更好的是，贡献 PR :)


## 本机扩散器 Swift 应用程序



在您自己的 Apple 硬件上运行稳定扩散的一种简单方法是使用
 [我们的开源 Swift 存储库](https://github.com/huggingface/swift-coreml-diffusers)
 ， 基于
 `扩散器`
 以及 Apple 的转换和推理存储库。您可以研究代码，然后编译它
 [Xcode](https://developer.apple.com/xcode/)
 并根据您自己的需要进行调整。为了您的方便，还有一个
 [App Store 中的独立 Mac 应用程序](https://apps.apple.com/app/diffusers/id1666309574)
 ，这样您就可以使用它，而无需处理代码或 IDE。如果您是开发人员并确定 Core ML 是构建 Stable Diffusion 应用程序的最佳解决方案，那么您可以使用本指南的其余部分来开始您的项目。我们迫不及待地想看看您将构建什么:)