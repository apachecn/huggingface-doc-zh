# 安装

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/installation>
>
> 原始地址：<https://huggingface.co/docs/diffusers/installation>


🤗 Diffusers 在 Python 3.8+、PyTorch 1.7.0+ 和 Flax 上进行了测试。对于您正在使用的深度学习库，请按照以下安装说明进行操作：


* [PyTorch](https://pytorch.org/get-started/locally/)
 安装说明
* [亚麻](https://flax.readthedocs.io/en/latest/)
 安装说明


## 使用 pip 安装



您应该将 🤗 扩散器安装在
 [虚拟环境](https://docs.python.org/3/library/venv.html)
 。
如果你不熟悉Python虚拟环境，请看一下这个
 [指南](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
 。
虚拟环境可以更轻松地管理不同的项目并避免依赖项之间的兼容性问题。


首先在项目目录中创建一个虚拟环境：



```
python -m venv .env
```


激活虚拟环境：



```
source .env/bin/activate
```


您还应该安装 🤗 Transformers，因为 🤗 Diffusers 依赖于它的模型：


火炬


隐藏 Pytorch 内容



```
pip install diffusers["torch"] transformers
```


.J{
 中风：#dce0df；
 }
.K {
 笔划线连接：圆形；
 }


贾克斯


隐藏 JAX 内容



```
pip install diffusers["flax"] transformers
```


## 使用 conda 安装



激活虚拟环境后，
 `康达`
 （由社区维护）：



```
conda install -c conda-forge diffusers
```


## 从源安装



在从源代码安装 🤗 Diffusers 之前，请确保已安装 PyTorch 和 🤗 Accelerate。


安装🤗加速：



```
pip install accelerate
```


然后从源安装 🤗 Diffusers：



```
pip install git+https://github.com/huggingface/diffusers
```


该命令安装前沿
 `主要`
 版本而不是最新版本
 `稳定`
 版本。
这
 `主要`
 版本对于了解最新进展很有用。
例如，如果自上次正式版本以来已修复错误，但尚未推出新版本。
然而，这意味着
 `主要`
 版本可能并不总是稳定。
我们努力保持
 `主要`
 版本可操作，大多数问题通常会在几个小时或一天内得到解决。
如果您遇到问题，请打开
 [问题](https://github.com/huggingface/diffusers/issues/new/choose)
 这样我们就可以更快地修复它！


## 可编辑安装



如果您愿意，您将需要可编辑的安装：


* 使用
 `主要`
 源代码的版本。
* 为 🤗 Diffusers 做出贡献，需要测试代码中的更改。


克隆存储库并使用以下命令安装 🤗 Diffusers：



```
git clone https://github.com/huggingface/diffusers.git
cd diffusers
```


火炬


隐藏 Pytorch 内容



```
pip install -e ".[torch]"
```


.J{
 中风：#dce0df；
 }
.K {
 笔划线连接：圆形；
 }


贾克斯


隐藏 JAX 内容



```
pip install -e ".[flax]"
```


这些命令将链接您将存储库克隆到的文件夹和您的 Python 库路径。
除了正常的库路径之外，Python 现在还会查看您克隆到的文件夹内部。
例如，如果您的 Python 包通常安装在
 `~/anaconda3/envs/main/lib/python3.8/site-packages/`
 , Python 也会搜索
 `~/扩散器/`
 您克隆到的文件夹。


您必须保留
 `扩散器`
 如果您想继续使用该库，请保存该文件夹。


现在，您可以使用以下命令轻松地将克隆更新到最新版本的 🤗 Diffusers：



```
cd ~/diffusers/
git pull
```


您的 Python 环境将找到
 `主要`
 下次运行时使用 🤗 扩散器版本。


## 缓存



模型权重和文件从集线器下载到缓存，该缓存通常是您的主目录。您可以通过指定来更改缓存位置
 `HF_HOME`
 或者
 `HUGGINFACE_HUB_CACHE`
 环境变量或配置
 `缓存目录`
 方法中的参数，例如
 [来自\_pretrained()](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained)
 。


缓存文件允许您离线运行 🤗 Diffusers。要防止 🤗 Diffuser 连接到互联网，请设置
 `HF_HUB_OFFLINE`
 环境变量为
 '真实'
 🤗 Diffusers 只会加载缓存中之前下载的文件。



```
export HF_HUB_OFFLINE=True
```


有关管理和清理缓存的更多详细信息，请查看
 [缓存](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
 指导。


## 遥测记录



我们的图书馆在期间收集遥测信息
 [来自\_pretrained()](/docs/diffusers/v0.23.1/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained)
 要求。
收集的数据包括🤗 Diffusers 和 PyTorch/Flax 的版本、请求的模型或管道类，
以及预训练检查点的路径（如果托管在 Hugging Face Hub 上）。
这些使用数据可以帮助我们调试问题并确定新功能的优先级。
仅当从 Hub 加载模型和管道时才会发送遥测数据，
如果您加载本地文件，则不会收集它。


我们了解并非每个人都愿意分享其他信息，并且我们尊重您的隐私。
您可以通过设置来禁用遥测收集
 `DISABLE_TELEMETRY`
 终端中的环境变量：


在 Linux/MacOS 上：



```
export DISABLE_TELEMETRY=YES
```


在 Windows 上：



```
set DISABLE_TELEMETRY=YES
```