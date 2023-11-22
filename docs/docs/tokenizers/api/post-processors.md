# Post-processors

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/api/post-processors>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/api/post-processors>


## BertProcessing




### 




 class
 

 tokenizers.processors.
 

 BertProcessing




 (
 


 sep
 


 cls
 




 )
 


 Parameters
 




* **sep** 
 (
 `Tuple[str, int]` 
 ) —
A tuple with the string representation of the SEP token, and its id
* **cls** 
 (
 `Tuple[str, int]` 
 ) —
A tuple with the string representation of the CLS token, and its id


 This post-processor takes care of adding the special tokens needed by
a Bert model:
 


* a SEP token
* a CLS token


## ByteLevel




### 




 class
 

 tokenizers.processors.
 

 ByteLevel




 (
 


 trim\_offsets
 
 = True
 



 )
 


 Parameters
 




* **trim\_offsets** 
 (
 `bool` 
 ) —
Whether to trim the whitespaces from the produced offsets.


 This post-processor takes care of trimming the offsets.
 



 By default, the ByteLevel BPE might include whitespaces in the produced tokens. If you don’t
want the offsets to include these whitespaces, then this PostProcessor must be used.
 


## RobertaProcessing




### 




 class
 

 tokenizers.processors.
 

 RobertaProcessing




 (
 


 sep
 


 cls
 


 trim\_offsets
 
 = True
 




 add\_prefix\_space
 
 = True
 



 )
 


 Parameters
 




* **sep** 
 (
 `Tuple[str, int]` 
 ) —
A tuple with the string representation of the SEP token, and its id
* **cls** 
 (
 `Tuple[str, int]` 
 ) —
A tuple with the string representation of the CLS token, and its id
* **trim\_offsets** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether to trim the whitespaces from the produced offsets.
* **add\_prefix\_space** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether the add\_prefix\_space option was enabled during pre-tokenization. This
is relevant because it defines the way the offsets are trimmed out.


 This post-processor takes care of adding the special tokens needed by
a Roberta model:
 


* a SEP token
* a CLS token



 It also takes care of trimming the offsets.
By default, the ByteLevel BPE might include whitespaces in the produced tokens. If you don’t
want the offsets to include these whitespaces, then this PostProcessor should be initialized
with
 `trim_offsets=True` 


## TemplateProcessing




### 




 class
 

 tokenizers.processors.
 

 TemplateProcessing




 (
 


 single
 


 pair
 


 special\_tokens
 




 )
 


 Parameters
 




* **single** 
 (
 `Template` 
 ) —
The template used for single sequences
* **pair** 
 (
 `Template` 
 ) —
The template used when both sequences are specified
* **special\_tokens** 
 (
 `Tokens` 
 ) —
The list of special tokens used in each sequences


 Provides a way to specify templates in order to add the special tokens to each
input sequence as relevant.
 



 Let’s take
 `BERT` 
 tokenizer as an example. It uses two special tokens, used to
delimitate each sequence.
 `[CLS]` 
 is always used at the beginning of the first
sequence, and
 `[SEP]` 
 is added at the end of both the first, and the pair
sequences. The final result looks like this:
 


* Single sequence:
 `[CLS] Hello there [SEP]`
* Pair sequences:
 `[CLS] My name is Anthony [SEP] What is my name? [SEP]`


 With the type ids as following:
 



```
[CLS]   ...   [SEP]   ...   [SEP]
0      0      0      1      1
```



 You can achieve such behavior using a TemplateProcessing:
 



```
TemplateProcessing(
    single="[CLS] $0 [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 0)],
)
```




 In this example, each input sequence is identified using a
 `$` 
 construct. This identifier
lets us specify each input sequence, and the type\_id to use. When nothing is specified,
it uses the default values. Here are the different ways to specify it:
 


* Specifying the sequence, with default
 `type_id == 0` 
 :
 `$A` 
 or
 `$B`
* Specifying the
 *type\_id* 
 with default
 `sequence == A` 
 :
 `$0` 
 ,
 `$1` 
 ,
 `$2` 
 , …
* Specifying both:
 `$A:0` 
 ,
 `$B:1` 
 , …



 The same construct is used for special tokens:
 `<identifier>(:<type_id>)?` 
.
 



**Warning** 
 : You must ensure that you are giving the correct tokens/ids as these
will be added to the Encoding without any further check. If the given ids correspond
to something totally different in a
 *Tokenizer* 
 using this
 *PostProcessor* 
 , it
might lead to unexpected results.
 



 Types:
 



 Template (
 `str` 
 or
 `List` 
 ):
 


* If a
 `str` 
 is provided, the whitespace is used as delimiter between tokens
* If a
 `List[str]` 
 is provided, a list of tokens



 Tokens (
 `List[Union[Tuple[int, str], Tuple[str, int], dict]]` 
 ):
 


* A
 `Tuple` 
 with both a token and its associated ID, in any order
* A
 `dict` 
 with the following keys:
 


	+ “id”:
	 `str` 
	 => The special token id, as specified in the Template
	+ “ids”:
	 `List[int]` 
	 => The associated IDs
	+ “tokens”:
	 `List[str]` 
	 => The associated tokens
 The given dict expects the provided
 `ids` 
 and
 `tokens` 
 lists to have
the same length.