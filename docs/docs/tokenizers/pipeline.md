# 标记化管道

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/pipeline>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/pipeline>


打电话时
 `Tokenizer.encode`
 或者
 `Tokenizer.encode_batch`
 ，输入
文本经过以下管道：


* `标准化`
* `预标记化`
*`模型`
* `后处理`


我们将详细了解每个步骤中发生的情况，
以及当你想要的时候
 `解码<解码>`
 一些令牌 ID，以及 🤗 Tokenizers 库如何帮助您
根据您的需要自定义每个步骤。如果你已经
熟悉这些步骤并想通过查看一些代码来学习，请跳转到
 `我们从头开始的 BERT 示例 <example>`
 。


对于需要的示例
 `分词器`
 我们将使用我们在中训练的分词器
 `快速游览`
 ，您可以使用以下命令加载：



```
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")
```


## 正常化



简而言之，标准化是应用于原始数据的一组操作
字符串以使其不那么随机或“更干净”。常见的操作包括
剥离空格、删除重音字符或全部小写
文本。如果您熟悉
 [统一码
标准化](https://unicode.org/reports/tr15)
 ，这也是一个非常
大多数标记器中应用的常见标准化操作。


每个标准化操作都在 🤗 Tokenizers 库中表示
由一个
 `标准化器`
 ，你可以结合
其中几个通过使用
 `标准化器.序列`
 。这是应用 NFD Unicode 标准化的标准化器
并以删除重音符号为例：



```
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
normalizer = normalizers.Sequence([NFD(), StripAccents()])
```


您可以通过将其应用于任何字符串来手动测试该规范化器：



```
normalizer.normalize_str("Héllò hôw are ü?")
# "Hello how are u?"
```


当建造一个
 `分词器`
 ， 你可以
只需更改相应的属性即可自定义其规范化器：



```
tokenizer.normalizer = normalizer
```


当然，如果您更改分词器应用标准化的方式，您
之后可能应该从头开始重新训练它。


## 预标记化



预标记化是将文本分割成更小的对象的行为
给出了你的代币在结束时的上限
训练。思考这个问题的一个好方法是预分词器将
将你的文本分成“单词”，然后，你的最终标记将是部分
这些话。


预标记输入的一种简单方法是按空格分割
标点符号，这是由
 `pre_tokenizers.Whitespace`
 预分词器：



```
from tokenizers.pre_tokenizers import Whitespace
pre_tokenizer = Whitespace()
pre_tokenizer.pre_tokenize_str("Hello! How are you? I'm fine, thank you.")
# [("Hello", (0, 5)), ("!", (5, 6)), ("How", (7, 10)), ("are", (11, 14)), ("you", (15, 18)),
# ("?", (18, 19)), ("I", (20, 21)), ("'", (21, 22)), ('m', (22, 23)), ("fine", (24, 28)),
# (",", (28, 29)), ("thank", (30, 35)), ("you", (36, 39)), (".", (39, 40))]
```


输出是一个元组列表，每个元组包含一个单词和
它在原句子中的跨度（用于确定最终的
 `偏移量`
 我们的
 `编码`
 ）。请注意，分裂
标点符号会分割缩写，例如
 「我是」
 在这个例子中。


您可以将任何组合在一起
 `预分词器`
 一起。例如，这是一个预分词器，它将
按空格、标点符号和数字进行分割，将数字分隔开
个别数字：



```
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Digits
pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
pre_tokenizer.pre_tokenize_str("Call 911!")
# [("Call", (0, 4)), ("9", (5, 6)), ("1", (6, 7)), ("1", (7, 8)), ("!", (8, 9))]
```


正如我们在
 `快速游览`
 ， 你可以
自定义 a 的预分词器
 `分词器`
 只需更改相应的属性即可：



```
tokenizer.pre_tokenizer = pre_tokenizer
```


当然，如果你改变预分词器的方式，你可能应该
之后从头开始重新训练您的分词器。


## 模型



一旦输入文本被标准化和预标记化，
 `分词器`
 将模型应用到
预令牌。这是管道中需要对您进行培训的部分
语料库（或者如果您使用的是预训练的语料库，则已经经过训练）
分词器）。


该模型的作用是将你的“单词”分割成标记，使用
它所学到的规则。它还负责将这些令牌映射到
它们在模型词汇表中对应的 ID。


该模型在初始化时被传递
 `分词器`
 所以你已经知道如何
自定义这部分。目前，🤗 Tokenizers 库支持：


* `模型.BPE`
* `模型.Unigram`
* `models.WordLevel`
* `models.WordPiece`


有关每个模型及其行为的更多详细信息，您可以检查
 [此处]（组件#模型）


## 后期处理



后处理是标记化管道的最后一步，
执行任何额外的转换
 `编码`
 在返回之前，例如
添加潜在的特殊标记。


正如我们在快速浏览中看到的，我们可以自定义后处理器
 `分词器`
 通过设置
相应的属性。例如，这是我们如何进行后处理
使输入适合 BERT 模型：



```
from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)
```


请注意，与预分词器或标准化器相反，您不需要
需要在更改后处理器后重新训练分词器。


## 总而言之：从头开始的 BERT 分词器



让我们将所有这些部分放在一起来构建一个 BERT 分词器。第一的，
BERT依赖于WordPiece，所以我们实例化一个新的
 `分词器`
 使用这个模型：



```
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
```


然后我们知道 BERT 通过删除重音符号来预处理文本
小写。我们还使用 unicode 标准化器：



```
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
```


预分词器只是根据空格和标点符号进行分割：



```
from tokenizers.pre_tokenizers import Whitespace
bert_tokenizer.pre_tokenizer = Whitespace()
```


后期处理使用我们之前看到的模板
部分：



```
from tokenizers.processors import TemplateProcessing
bert_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)
```


我们可以使用这个分词器并在 wikitext 上对其进行训练，就像
 `快速游览`
 :



```
from tokenizers.trainers import WordPieceTrainer
trainer = WordPieceTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
bert_tokenizer.train(files, trainer)
bert_tokenizer.save("data/bert-wiki.json")
```


## 解码



除了对输入文本进行编码之外，
 `分词器`
 还有一个用于解码的API，即转换ID
由您的模型生成回文本。这是通过以下方法完成的
 `Tokenizer.decode`
 （对于一个预测文本）和
 `Tokenizer.decode_batch`
 （对于一批预测）。


这
 `解码器`
 首先将 ID 转换回令牌
（使用标记器的词汇）并删除所有特殊标记，然后
用空格连接这些标记：



```
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.ids)
# [1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2]
tokenizer.decode([1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2])
# "Hello , y ' all ! How are you ?"
```


如果您使用的模型添加了特殊字符来表示子标记
给定的“单词”（例如
 `“##”`
 在
WordPiece）您需要自定义
 `解码器`
 治疗
他们正确地。如果我们把之前的
 `bert_tokenizer`
 例如
默认解码将给出：



```
output = bert_tokenizer.encode("Welcome to the 🤗 Tokenizers library.")
print(output.tokens)
# ["[CLS]", "welcome", "to", "the", "[UNK]", "tok", "##eni", "##zer", "##s", "library", ".", "[SEP]"]
bert_tokenizer.decode(output.ids)
# "welcome to the tok ##eni ##zer ##s library ."
```


但是通过将其更改为适当的解码器，我们得到：



```
from tokenizers import decoders
bert_tokenizer.decoder = decoders.WordPiece()
bert_tokenizer.decode(output.ids)
# "welcome to the tokenizers library."
```



