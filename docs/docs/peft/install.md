# 安装

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/install>
>
> 原始地址：<https://huggingface.co/docs/peft/install>



在开始之前，您需要设置您的环境，安装适当的软件包，并配置 🤗 PEFT。 
🤗 PEFT 在 Python 3.8+ 上进行了测试。  
🤗 PEFT 可通过 PyPI 和 GitHub源码 安装：  


## PyPI
通过 PyPI 安装 🤗 PEFT：

```bash
pip install peft
```


## 源码

每天都会添加尚未发布的新功能，这也意味着可能会存在一些错误。要尝试这些功能，请从 GitHub 存储库安装：

```bash
pip install git+https://github.com/huggingface/peft
```


如果您希望为该库做出贡献，或者想要玩转源代码并在运行代码时看到实时结果，您可以从本地克隆的存储库安装可编辑版本：

```bash
git clone https://github.com/huggingface/peft
cd peft
pip install -e .
```