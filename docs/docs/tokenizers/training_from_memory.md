# 从记忆中训练

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/training_from_memory>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/training_from_memory>


在里面
 [快速游览](快速游览)
 ，我们看到了如何构建和训练
tokenizer 使用文本文件，但我们实际上可以使用任何 Python 迭代器。
在本节中，我们将看到几种不同的训练方法
分词器。


对于下面列出的所有示例，我们将使用相同的
 [Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer)
 和
 「教练」
 ，构建为
下列的：



```
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
tokenizer = Tokenizer(models.Unigram())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.UnigramTrainer(
    vocab_size=20000,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<PAD>", "<BOS>", "<EOS>"],
)
```


该分词器基于
 [Unigram](/docs/tokenizers/v0.13.4.rc2/en/api/models#tokenizers.models.Unigram)
 模型。它
负责使用 NFKC Unicode 标准化对输入进行标准化
方法，并使用
 [ByteLevel](/docs/tokenizers/v0.13.4.rc2/en/api/pre-tokenizers#tokenizers.pre_tokenizers.ByteLevel)
 预分词器和相应的解码器。


有关此处使用的组件的更多信息，您可以查看
 [此处]（组件）
 。


## 最基本的方法



正如您可能已经猜到的，训练我们的分词器的最简单方法
是通过使用
 `列表`
 {.解释文本角色=“obj”}：



```
# First few lines of the "Zen of Python" https://www.python.org/dev/peps/pep-0020/
data = [
    "Beautiful is better than ugly."
    "Explicit is better than implicit."
    "Simple is better than complex."
    "Complex is better than complicated."
    "Flat is better than nested."
    "Sparse is better than dense."
    "Readability counts."
]
tokenizer.train_from_iterator(data, trainer=trainer)
```


容易，对吧？您可以在这里使用任何作为迭代器工作的东西，无论是
 `列表`
 {.解释文本角色=“obj”}，
 `元组`
 {.解释文本
角色=“obj”}，或
 `np.Array`
 {.解释文本角色=“obj”}。任何事物
只要它提供字符串就可以工作。


## 使用 🤗 数据集库



访问现有众多数据集之一的绝佳方法
是通过使用 🤗 数据集库。有关它的更多信息，您
应该检查
 [官方文档
这里](https://huggingface.co/docs/datasets/)
 。


让我们从加载数据集开始：



```
import datasets
dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train+test+validation")
```


下一步是在此数据集上构建迭代器。最简单的方法
为此，可能需要使用生成器：



```
def batch\_iterator(batch\_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]
```


正如您在这里所看到的，为了提高效率，我们实际上可以提供
一批用于训练的示例，而不是逐一迭代它们
一。通过这样做，我们可以预期性能与我们的性能非常相似
直接从文件训练时获得。


迭代器准备就绪后，我们只需要启动训练即可。为了
为了改善进度条的外观，我们可以指定总计
数据集长度：



```
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))
```


就是这样！


## 使用 gzip 文件



由于Python中的gzip文件可以用作迭代器，因此非常方便
对此类文件进行简单的训练：



```
import gzip
with gzip.open("data/my-file.0.gz", "rt") as f:
    tokenizer.train_from_iterator(f, trainer=trainer)
```


现在，如果我们想从多个 gzip 文件进行训练，那就不会太多了
更难：



```
files = ["data/my-file.0.gz", "data/my-file.1.gz", "data/my-file.2.gz"]
def gzip\_iterator():
    for path in files:
        with gzip.open(path, "rt") as f:
            for line in f:
                yield line
tokenizer.train_from_iterator(gzip_iterator(), trainer=trainer)
```


现在你就拥有了！