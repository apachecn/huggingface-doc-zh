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


🤗 Tokenizers 库的一个重要功能是它附带
完全对齐跟踪，这意味着您始终可以获得您的部分
对应于给定标记的原始句子。这些存储在
这
 `偏移量`
 我们的属性
 `编码`
 目的。例如，让我们
假设我们想要找到导致该事件的原因
 `“[UNK]”`
 出现的令牌，这是
列表中索引 9 处的 token，我们只需询问该处的偏移量即可
指数：



```
print(output.offsets[9])
# (26, 27)
```


这些是与原始表情符号相对应的索引
句子：



```
sentence = "Hello, y'all! How are you 😁 ?"
sentence[26:27]
# "😁"
```



### 


 后期处理


我们可能希望我们的标记生成器自动添加特殊标记，例如
 `“[CLS]”`
 或者
 `“[九月]”`
 。为此，我们使用后处理器。
 `模板处理`
 是最
常用的，你只需要指定一个模板来处理
单句和句子对，以及特殊标记
以及他们的身份证。


当我们构建标记器时，我们设置
 `“[CLS]”`
 和
 `“[九月]”`
 在位置 1
以及我们的特殊令牌列表中的 2 个，因此这应该是它们的 ID。到
仔细检查，我们可以使用
 `Tokenizer.token_to_id`
 方法：



```
tokenizer.token_to_id("[SEP]")
# 2
```


以下是我们如何设置后处理来为我们提供传统的
BERT 输入：



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


让我们更详细地查看这段代码。首先我们指定
单个句子的模板：那些应该具有以下形式
 `“[CLS] $A [九月]”`
 在哪里
 `$A`
 代表我们的句子。


然后，我们指定句子对的模板，它应该具有
形式
 `“[CLS] $A [九月] $B [九月]”`
 在哪里
 `$A`
 代表第一句话和
 `$B`
 第二个。这
 `:1`
 在模板中添加代表
 `类型 ID`
 我们想要输入的每个部分：它是默认的
一切都为 0（这就是为什么我们没有
 `$A:0`
 ），这里我们将其设置为 1
第二句和最后一句的标记
 `“[九月]”`
 令牌。


最后，我们在我们的代码中指定我们使用的特殊令牌及其 ID
分词器的词汇表。


为了检查它是否正常工作，让我们尝试对其进行编码
句子如前：



```
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
# ["[CLS]", "Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?", "[SEP]"]
```


要检查一对句子的结果，我们只需传递两个
句子
 `Tokenizer.encode`
 :



```
output = tokenizer.encode("Hello, y'all!", "How are you 😁 ?")
print(output.tokens)
# ["[CLS]", "Hello", ",", "y", "'", "all", "!", "[SEP]", "How", "are", "you", "[UNK]", "?", "[SEP]"]
```


然后，您可以检查归因于每个令牌的类型 ID 是否正确



```
print(output.type_ids)
# [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
```


如果你保存你的分词器
 `Tokenizer.save`
 ，后处理器将被一起保存。


### 


 批量编码多个句子


要获得 🤗 Tokenizers 库的全速，最好
使用以下命令批量处理您的文本
 `Tokenizer.encode_batch`
 方法：



```
output = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])
```


然后输出是一个列表
 `编码`
 像我们之前看到的那些物体。您可以一起处理尽可能多的
只要你能记住，就可以随心所欲地编写文本。


要处理一批句子对，请将两个列表传递给
 `Tokenizer.encode_batch`
 方法：将
句子列表 A 和句子列表 B：



```
output = tokenizer.encode_batch(
    [["Hello, y'all!", "How are you 😁 ?"], ["Hello to you too!", "I'm fine, thank you!"]]
)
```


当编码多个句子时，可以自动填充输出
使用当前最长的句子
 `Tokenizer.enable_padding`
 ，与
 `pad_token`
 及其 ID（我们可以
仔细检查填充令牌的 id
 `Tokenizer.token_to_id`
 像以前一样）：



```
tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
```


我们可以设置
 `方向`
 填充物的
（默认为右侧）或给定
 `长度`
 如果我们想将每个样本填充到该特定数字（此处
我们将其保留为未设置以填充到最长文本的大小）。



```
output = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])
print(output[1].tokens)
# ["[CLS]", "How", "are", "you", "[UNK]", "?", "[SEP]", "[PAD]"]
```


在这种情况下，
 ‘注意面具’
 由产生的
分词器考虑填充：



```
print(output[1].attention_mask)
# [1, 1, 1, 1, 1, 1, 1, 0]
```


## 预训练



### 


 使用预训练的分词器


您可以从 Hugging Face Hub 加载任何标记器，只要
 `tokenizer.json`
 文件在存储库中可用。



```
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
```


### 


 从遗留词汇文件导入预训练的分词器


您也可以直接导入预训练的分词器，只要您
有它的词汇文件。例如，以下是如何导入
经典的预训练 BERT 分词器：



```
from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
```


只要您下载了文件
 `bert-base-uncased-vocab.txt`
 和



```
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
```



