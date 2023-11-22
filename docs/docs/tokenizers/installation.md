# 安装

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/installation>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/installation>


🤗 Tokenizers 在 Python 3.5+ 上进行了测试。


您应该在中安装 🤗 Tokenizers
 [虚拟环境](https://docs.python.org/3/library/venv.html)
 。如果你是
不熟悉Python虚拟环境，请查看
 [用户
指南](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
 。
使用您要使用的 Python 版本创建虚拟环境
使用并激活它。


## 使用 pip 安装



🤗 Tokenizers 可以使用 pip 安装，如下所示：



```
pip install tokenizers
```


## 从源安装



要使用此方法，您需要安装 Rust 语言。你
可以按照
 [官方
指南](https://www.rust-lang.org/learn/get-started)
 了解更多
信息。


如果您使用的是基于 UNIX 的操作系统，安装应该很简单
运行时：



```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```


或者您可以使用以下命令轻松更新它：



```
rustup update
```


安装 Rust 后，我​​们就可以开始检索 🤗 的源代码
分词器：



```
git clone https://github.com/huggingface/tokenizers
```


然后我们进入python绑定文件夹：



```
cd tokenizers/bindings/python
```


此时你应该有你的
 虚拟环境
 已经
活性。为了编译 🤗 Tokenizers，您需要安装
Python包
 `setuptools_rust`
 ：



```
pip install setuptools_rust
```


然后你就可以在你的虚拟机中编译并安装 🤗 Tokenizers
环境使用以下命令：



```
python setup.py install
```