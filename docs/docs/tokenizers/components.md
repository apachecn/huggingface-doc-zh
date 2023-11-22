# 成分

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/components>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/components>


构建 Tokenizer 时，您可以将各种类型的组件附加到
此 Tokenizer 以便自定义其行为。此页列出了大多数
提供的组件。


## 标准化器



A
 `标准化器`
 负责按顺序预处理输入字符串
将其标准化为与给定用例相关。一些常见的例子
标准化的算法是 Unicode 标准化算法（NFD、NFKD、
NFC 和 NFKC）、小写等……的特殊性
 `标记器`
 就是它
我们在标准化时跟踪对齐情况。这对于
允许从生成的标记映射回输入文本。


这
 `标准化器`
 是可选的。


| 	 Name	  | 	 Description	  | 	 Example	  |
| --- | --- | --- |
| 	 NFD	  | 	 NFD unicode normalization	  |  |
| 	 NFKD	  | 	 NFKD unicode normalization	  |  |
| 	 NFC	  | 	 NFC unicode normalization	  |  |
| 	 NFKC	  | 	 NFKC unicode normalization	  |  |
| 	 Lowercase	  | 	 Replaces all uppercase to lowercase	  | 	 Input:	 `HELLO ὈΔΥΣΣΕΎΣ` 		 Output:	 `hello` 	 ὀδυσσεύς`	  |
| 	 Strip	  | 	 Removes all whitespace characters on the specified sides (left, right or both) of the input	  | 	 Input:	 `"` 	 hi	 `"` 		 Output:	 `"hi"`  |
| 	 StripAccents	  | 	 Removes all accent symbols in unicode (to be used with NFD for consistency)	  | 	 Input:	 `é` 		 Ouput:	 `e`  |
| 	 Replace	  | 	 Replaces a custom string or regexp and changes it with given content	  | `Replace("a", "e")` 	 will behave like this:	 	 Input:	 `"banana"` 		 Ouput:	 `"benene"`  |
| 	 BertNormalizer	  | 	 Provides an implementation of the Normalizer used in the original BERT. Options that can be set are:	 * clean\_text	* handle\_chinese\_chars	* strip\_accents	* lowercase	 |  |
| 	 Sequence	  | 	 Composes multiple normalizers that will run in the provided order	  | `Sequence([NFKC(), Lowercase()])`  |


## 预分词器



这
 `预分词器`
 负责根据一组分割输入
规则。此预处理可让您确保底层
 `模型`
 不会跨多个“分裂”构建代币。例如如果
如果您不想在令牌内有空格，那么您可以使用
 `预分词器`
 在这些空白处分裂。


您可以轻松组合多个
 `预分词器`
 一起使用
 `序列`
 （见下文）。这
 `预分词器`
 也允许修改
字符串，就像一个
 `标准化器`
 做。这是必要的，以允许一些
需要在标准化之前进行分割的复杂算法（例如
字节级）


| 	 Name	  | 	 Description	  | 	 Example	  |
| --- | --- | --- |
| 	 ByteLevel	  | 	 Splits on whitespaces while remapping all the bytes to a set of visible characters. This technique as been introduced by OpenAI with GPT-2 and has some more or less nice properties:	 * Since it maps on bytes, a tokenizer using this only requires	 **256** 	 characters as initial alphabet (the number of values a byte can have), as opposed to the 130,000+ Unicode characters.	* A consequence of the previous point is that it is absolutely unnecessary to have an unknown token using this since we can represent anything with 256 tokens (Youhou!! 🎉🎉)	* For non ascii characters, it gets completely unreadable, but it works nonetheless!	 | 	 Input:	 `"Hello my friend, how are you?"` 		 Ouput:	 `"Hello", "Ġmy", Ġfriend", ",", "Ġhow", "Ġare", "Ġyou", "?"`  |
| 	 Whitespace	  | 	 Splits on word boundaries (using the following regular expression:	 `\w+|[^\w\s]+`  | 	 Input:	 `"Hello there!"` 		 Output:	 `"Hello", "there", "!"`  |
| 	 WhitespaceSplit	  | 	 Splits on any whitespace character	  | 	 Input:	 `"Hello there!"` 		 Output:	 `"Hello", "there!"`  |
| 	 Punctuation	  | 	 Will isolate all punctuation characters	  | 	 Input:	 `"Hello?"` 		 Ouput:	 `"Hello", "?"`  |
| 	 Metaspace	  | 	 Splits on whitespaces and replaces them with a special char “▁” (U+2581)	  | 	 Input:	 `"Hello there"` 		 Ouput:	 `"Hello", "▁there"`  |
| 	 CharDelimiterSplit	  | 	 Splits on a given character	  | 	 Example with	 `x` 	 :	 	 Input:	 `"Helloxthere"` 		 Ouput:	 `"Hello", "there"`  |
| 	 Digits	  | 	 Splits the numbers from any other characters.	  | 	 Input:	 `"Hello123there"` 		 Output:	 `"Hello", "123", "there"`  |
| 	 Split	  | 	 Versatile pre-tokenizer that splits on provided pattern and according to provided behavior. The pattern can be inverted if necessary.	 * pattern should be either a custom string or regexp.	* behavior should be one of:		+ removed		+ isolated		+ merged\_with\_previous		+ merged\_with\_next		+ contiguous	* invert should be a boolean flag.	 | 	 Example with pattern =	 	 , behavior =	 `"isolated"` 	 , invert =	 `False` 	 :	 	 Input:	 `"Hello, how are you?"` 		 Output:	 `"Hello,", " ", "how", " ", "are", " ", "you?"`  |
| 	 Sequence	  | 	 Lets you compose multiple	 `PreTokenizer` 	 that will be run in the given order	  | `Sequence([Punctuation(), WhitespaceSplit()])`  |


## 楷模



模型是用于实际标记化的核心算法，因此，
它们是 Tokenizer 的唯一必需组件。


| 	 Name	  | 	 Description	  |
| --- | --- |
| 	 WordLevel	  | 	 This is the “classic” tokenization algorithm. It let’s you simply map words to IDs without anything fancy. This has the advantage of being really simple to use and understand, but it requires extremely large vocabularies for a good coverage. Using this	 `Model` 	 requires the use of a	 `PreTokenizer` 	. No choice will be made by this model directly, it simply maps input tokens to IDs.	  |
| 	 BPE	  | 	 One of the most popular subword tokenization algorithm. The Byte-Pair-Encoding works by starting with characters, while merging those that are the most frequently seen together, thus creating new tokens. It then works iteratively to build new tokens out of the most frequent pairs it sees in a corpus. BPE is able to build words it has never seen by using multiple subword tokens, and thus requires smaller vocabularies, with less chances of having “unk” (unknown) tokens.	  |
| 	 WordPiece	  | 	 This is a subword tokenization algorithm quite similar to BPE, used mainly by Google in models like BERT. It uses a greedy algorithm, that tries to build long words first, splitting in multiple tokens when entire words don’t exist in the vocabulary. This is different from BPE that starts from characters, building bigger tokens as possible. It uses the famous	 `##` 	 prefix to identify tokens that are part of a word (ie not starting a word).	  |
| 	 Unigram	  | 	 Unigram is also a subword tokenization algorithm, and works by trying to identify the best set of subword tokens to maximize the probability for a given sentence. This is different from BPE in the way that this is not deterministic based on a set of rules applied sequentially. Instead Unigram will be able to compute multiple ways of tokenizing, while choosing the most probable one.	  |


## 后处理器



在整个管道之后，我们有时想插入一些特殊的
标记之前将标记化字符串输入模型，如“[CLS] My
马太棒了[九月]”。这
 `后处理器`
 组件正在做什么
只是。


| 	 Name	  | 	 Description	  | 	 Example	  |
| --- | --- | --- |
| 	 TemplateProcessing	  | 	 Let’s you easily template the post processing, adding special tokens, and specifying the	 `type_id` 	 for each sequence/special token. The template is given two strings representing the single sequence and the pair of sequences, as well as a set of special tokens to use.	  | 	 Example, when specifying a template with these values:	 * single:	 `"[CLS] $A [SEP]"`	* pair:	 `"[CLS] $A [SEP] $B [SEP]"`	* special tokens:		+ `"[CLS]"`		+ `"[SEP]"`


 输入：
 `(“我喜欢这个”，“但不喜欢这个”)`

 输出：
 `“[CLS] 我喜欢这个 [SEP]，但不喜欢这个 [SEP]”` |


## 解码器



解码器知道如何从 Tokenizer 使用的 ID 回到
一段可读的文字。一些
 `标准化器`
 和
 `预分词器`
 使用
例如，需要恢复的特殊字符或标识符。


| 	 Name	  | 	 Description	  |
| --- | --- |
| 	 ByteLevel	  | 	 Reverts the ByteLevel PreTokenizer. This PreTokenizer encodes at the byte-level, using a set of visible Unicode characters to represent each byte, so we need a Decoder to revert this process and get something readable again.	  |
| 	 Metaspace	  | 	 Reverts the Metaspace PreTokenizer. This PreTokenizer uses a special identifer	 `▁` 	 to identify whitespaces, and so this Decoder helps with decoding these.	  |
| 	 WordPiece	  | 	 Reverts the WordPiece Model. This model uses a special identifier	 `##` 	 for continuing subwords, and so this Decoder helps with decoding these.	  |



