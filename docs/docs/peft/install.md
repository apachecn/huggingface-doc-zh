# 安装

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/install>
>
> 原始地址：<https://huggingface.co/docs/peft/install>


在开始之前，您需要设置环境、安装适当的软件包并配置 🤗 PEFT。 🤗 PEFT 已测试
 **Python 3.8+**
 。


🤗 PEFT 可在 PyPI 和 GitHub 上使用：


## 皮伊



要从 PyPI 安装 🤗 PEFT：



```
pip install peft
```


## 来源



每天都会添加尚未发布的新功能，这也意味着可能存在一些错误。要试用它们，请从 GitHub 存储库安装：



```
pip install git+https://github.com/huggingface/peft
```


如果您正在为库做出贡献或希望使用源代码并查看实时情况
当您运行代码时，可以从本地克隆版本安装可编辑版本
存储库：



```
git clone https://github.com/huggingface/peft
cd peft
pip install -e .
```