# Decoders

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/api/decoders>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/api/decoders>


## BPEDecoder




### 




 class
 

 tokenizers.decoders.
 

 BPEDecoder




 (
 


 suffix
 
 = '</w>'
 



 )
 


 Parameters
 




* **suffix** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `</w>` 
 ) —
The suffix that was used to caracterize an end-of-word. This suffix will
be replaced by whitespaces during the decoding


 BPEDecoder Decoder
 


## ByteLevel




### 




 class
 

 tokenizers.decoders.
 

 ByteLevel




 (
 

 )
 




 ByteLevel Decoder
 



 This decoder is to be used in tandem with the
 [ByteLevel](/docs/tokenizers/v0.13.4.rc2/en/api/pre-tokenizers#tokenizers.pre_tokenizers.ByteLevel) 
[PreTokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/pre-tokenizers#tokenizers.pre_tokenizers.PreTokenizer) 
.
 


## CTC




### 




 class
 

 tokenizers.decoders.
 

 CTC




 (
 


 pad\_token
 
 = '<pad>'
 




 word\_delimiter\_token
 
 = '|'
 




 cleanup
 
 = True
 



 )
 


 Parameters
 




* **pad\_token** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `<pad>` 
 ) —
The pad token used by CTC to delimit a new token.
* **word\_delimiter\_token** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `|` 
 ) —
The word delimiter token. It will be replaced by a
* **cleanup** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether to cleanup some tokenization artifacts.
Mainly spaces before punctuation, and some abbreviated english forms.


 CTC Decoder
 


## Metaspace




### 




 class
 

 tokenizers.decoders.
 

 Metaspace




 (
 

 )
 


 Parameters
 




* **replacement** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `▁` 
 ) —
The replacement character. Must be exactly one character. By default we
use the
 *▁* 
 (U+2581) meta symbol (Same as in SentencePiece).
* **add\_prefix\_space** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether to add a space to the first word if there isn’t already one. This
lets us treat
 *hello* 
 exactly like
 *say hello* 
.


 Metaspace Decoder
 


## WordPiece




### 




 class
 

 tokenizers.decoders.
 

 WordPiece




 (
 


 prefix
 
 = '##'
 




 cleanup
 
 = True
 



 )
 


 Parameters
 




* **prefix** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `##` 
 ) —
The prefix to use for subwords that are not a beginning-of-word
* **cleanup** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether to cleanup some tokenization artifacts. Mainly spaces before punctuation,
and some abbreviated english forms.


 WordPiece Decoder