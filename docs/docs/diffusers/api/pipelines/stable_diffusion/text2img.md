> 翻译任务

* 目前该页面无人翻译，期待你的加入
* 翻译奖励: <https://github.com/orgs/apachecn/discussions/243>
* 任务认领: <https://github.com/apachecn/huggingface-doc-zh/discussions/1>

请参考这个模版来写内容:


# Hugging Face 某某页面

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/stable_diffusion/text2img>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img>

开始写原始页面的翻译内容



注意事项: 

1. 代码参考:

```py
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

2. 公式参考:

1) 无需换行的写法: 

$\sqrt{w^T*w}$

2) 需要换行的写法：

$$
\sqrt{w^T*w}
$$

3. 图片参考(用图片的实际地址就行):

<img src='http://data.apachecn.org/img/logo/logo_green.png' width=20% />

4. **翻译完后请删除上面所有模版内容就行**