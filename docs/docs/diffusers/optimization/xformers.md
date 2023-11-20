# xFormers

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/optimization/xformers>
>
> 原始地址：<https://huggingface.co/docs/diffusers/optimization/xformers>


我们推荐
 [xFormers](https://github.com/facebookresearch/xformers)
 用于推理和训练。在我们的测试中，在注意力块中执行的优化可以提高速度并减少内存消耗。


从以下位置安装 xFormers
 `点`
 ：



```
pip install xformers
```


xFormers
 `点`
 软件包需要最新版本的 PyTorch。如果您需要使用以前版本的 PyTorch，那么我们建议
 [从源安装 xFormers](https://github.com/facebookresearch/xformers#installing-xformers)
 。


xFormers安装后，您可以使用
 `enable_xformers_memory_efficient_attention()`
 为了更快的推理和减少内存消耗，如下所示
 [部分](内存#内存效率-注意力)
 。


根据这个
 [问题](https://github.com/huggingface/diffusers/issues/2234#issuecomment-1416931212)
 , xFormers
 `v0.0.16`
 无法用于某些 GPU 中的训练（fine-tune 或 DreamBooth）。如果您发现此问题，请安装问题评论中所示的开发版本。