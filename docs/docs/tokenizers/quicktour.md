# 快速游览

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/quicktour>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/quicktour>


让我们快速浏览一下 🤗 Tokenizers 库功能。这
库提供了当今最常用的分词器的实现
既易于使用又速度极快。


## 从头开始构建分词器



为了说明 🤗 Tokenizers 库的速度有多快，让我们训练一个新的
分词器开启
 [wikitext-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
 （516M 文本）只需几秒钟。首先，您需要
下载此数据集并使用以下命令解压缩：



```
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
```


### 


 训练分词器


在本次巡演中，我们将构建并训练字节对编码（BPE）
分词器。有关不同类型标记器的更多信息，
看看这个
 [指南](https://huggingface.co/transformers/tokenizer_summary.html)
 在
🤗 变形金刚文档。在这里，训练分词器意味着
将通过以下方式学习合并规则：


* 从训练语料库中存在的所有字符开始
代币。
* 识别最常见的一对标记并将其合并为一个标记。
* 重复直到词汇量（例如，标记的数量）达到
我们想要的尺寸。


该库的主要 API 是
 `类`
`分词器`
 ，这是如何
我们用 BPE 模型实例化一个：



```
from tokenizers import Tokenizer
from tokenizers.models import BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
```


为了在维基文本文件上训练我们的分词器，我们需要
实例化一个 [trainer]{.title-ref}，在本例中是
 `Bpe训练器`



```
from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
```


我们可以设置训练参数，例如
 `词汇大小`
 或者
 `最小频率`
 （这里
保留默认值 30,000 和 0），但最重要的是
部分是给出
 `特殊令牌`
 我们
计划稍后使用（在训练期间根本不使用它们），以便
它们被插入到词汇表中。


编写特殊标记列表的顺序很重要：此处
 `“[UNK]”`
 将得到ID 0，
 `“[CLS]”`
 将得到 ID 1 等等。


我们现在就可以训练我们的分词器，但这并不是最佳的。
如果没有预分词器将我们的输入分割成单词，我们可能会
获取与多个单词重叠的标记：例如我们可以得到一个
 ``“它是”`
 自从这两个词以来的令牌
经常出现在彼此旁边。使用预分词器将确保不会
令牌比预令牌生成器返回的单词大。在这里我们想要
训练子词 BPE 分词器，我们将使用最简单的
通过空格分割可以实现预分词器。



```
from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()
```


现在，我们可以调用
 `Tokenizer.train`
 方法与我们想要使用的任何文件列表：



```
files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
tokenizer.train(files, trainer)
```


这应该只需要几秒钟就可以完整地训练我们的分词器
维基文本数据集！将分词器保存在一个包含所有分词器的文件中
它的配置和词汇，只需使用
 `Tokenizer.save`
 方法：



```
tokenizer.save("data/tokenizer-wiki.json")
```


您可以使用以下命令从该文件重新加载分词器
 `Tokenizer.from_file`
`类方法`
 :



```
tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")
```



### 


 使用分词器


现在我们已经训练了分词器，我们可以在任何我们想要的文本上使用它
与
 `Tokenizer.encode`
 方法：



```
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
```


这将标记器的完整管道应用于文本，返回
一个
 `编码`
 目的。了解更多
有关此管道以及如何应用（或自定义）其部分的信息，请查看
 [本页]（管道）
 。


这
 `编码`
 对象则拥有所有
深度学习模型（或其他模型）所需的属性。这
 `代币`
 属性包含
以标记形式分割文本：



```
print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]
```


同样，
 `id`
 属性将
包含标记器中每个标记的索引
词汇：



```
print(output.ids)
# [27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35]
```



```
print(output.offsets[9])
# (26, 27)
```



```
sentence = "Hello, y'all! How are you 😁 ?"
sentence[26:27]
# "😁"
```



### 



```
tokenizer.token_to_id("[SEP]")
# 2
```



```
from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)
```


