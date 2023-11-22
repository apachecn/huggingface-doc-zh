# Input Sequences

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/api/input-sequences>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/api/input-sequences>



 These types represent all the different kinds of sequence that can be used as input of a Tokenizer.
Globally, any sequence can be either a string or a list of strings, according to the operating
mode of the tokenizer:
 `raw text` 
 vs
 `pre-tokenized` 
.
 


## TextInputSequence



`tokenizers.TextInputSequence` 

 A
 `str` 
 that represents an input sequence
 


## PreTokenizedInputSequence



`tokenizers.PreTokenizedInputSequence` 

 A pre-tokenized input sequence. Can be one of:
 


* A
 `List` 
 of
 `str`
* A
 `Tuple` 
 of
 `str`



 alias of
 `Union[List[str], Tuple[str]]` 
.
 


## InputSequence



`tokenizers.InputSequence` 

 Represents all the possible types of input sequences for encoding. Can be:
 


* When
 `is_pretokenized=False` 
 :
 [TextInputSequence](#tokenizers.TextInputSequence)
* When
 `is_pretokenized=True` 
 :
 [PreTokenizedInputSequence](#tokenizers.PreTokenizedInputSequence)



 alias of
 `Union[str, List[str], Tuple[str]]` 
.