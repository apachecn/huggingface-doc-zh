# Encode Inputs

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/api/encode-inputs>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/api/encode-inputs>



 These types represent all the different kinds of input that a
 [Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 
 accepts
when using
 `encode_batch()` 
.
 


## TextEncodeInput



`tokenizers.TextEncodeInput` 

 Represents a textual input for encoding. Can be either:
 


* A single sequence:
 [TextInputSequence](/docs/tokenizers/api/input-sequences#tokenizers.TextInputSequence)
* A pair of sequences:
	+ A Tuple of
	 [TextInputSequence](/docs/tokenizers/api/input-sequences#tokenizers.TextInputSequence)
	+ Or a List of
	 [TextInputSequence](/docs/tokenizers/api/input-sequences#tokenizers.TextInputSequence) 
	 of size 2



 alias of
 `Union[str, Tuple[str, str], List[str]]` 
.
 


## PreTokenizedEncodeInput



`tokenizers.PreTokenizedEncodeInput` 

 Represents a pre-tokenized input for encoding. Can be either:
 


* A single sequence:
 [PreTokenizedInputSequence](/docs/tokenizers/api/input-sequences#tokenizers.PreTokenizedInputSequence)
* A pair of sequences:
	+ A Tuple of
	 [PreTokenizedInputSequence](/docs/tokenizers/api/input-sequences#tokenizers.PreTokenizedInputSequence)
	+ Or a List of
	 [PreTokenizedInputSequence](/docs/tokenizers/api/input-sequences#tokenizers.PreTokenizedInputSequence) 
	 of size 2



 alias of
 `Union[List[str], Tuple[str], Tuple[Union[List[str], Tuple[str]], Union[List[str], Tuple[str]]], List[Union[List[str], Tuple[str]]]]` 
.
 


## EncodeInput



`tokenizers.EncodeInput` 

 Represents all the possible types of input for encoding. Can be:
 


* When
 `is_pretokenized=False` 
 :
 [TextEncodeInput](#tokenizers.TextEncodeInput)
* When
 `is_pretokenized=True` 
 :
 [PreTokenizedEncodeInput](#tokenizers.PreTokenizedEncodeInput)



 alias of
 `Union[str, Tuple[str, str], List[str], Tuple[str], Tuple[Union[List[str], Tuple[str]], Union[List[str], Tuple[str]]], List[Union[List[str], Tuple[str]]]]` 
.