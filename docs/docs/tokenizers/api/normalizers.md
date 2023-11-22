# Normalizers

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/api/normalizers>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/api/normalizers>


## BertNormalizer




### 




 class
 

 tokenizers.normalizers.
 

 BertNormalizer




 (
 


 clean\_text
 
 = True
 




 handle\_chinese\_chars
 
 = True
 




 strip\_accents
 
 = None
 




 lowercase
 
 = True
 



 )
 


 Parameters
 




* **clean\_text** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether to clean the text, by removing any control characters
and replacing all whitespaces by the classic one.
* **handle\_chinese\_chars** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether to handle chinese chars by putting spaces around them.
* **strip\_accents** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Whether to strip all accents. If this option is not specified (ie == None),
then it will be determined by the value for
 *lowercase* 
 (as in the original Bert).
* **lowercase** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether to lowercase.


 BertNormalizer
 



 Takes care of normalizing raw text before giving it to a Bert model.
This includes cleaning the text, handling accents, chinese chars and lowercasing
 


## Lowercase




### 




 class
 

 tokenizers.normalizers.
 

 Lowercase




 (
 

 )
 




 Lowercase Normalizer
 


## NFC




### 




 class
 

 tokenizers.normalizers.
 

 NFC




 (
 

 )
 




 NFC Unicode Normalizer
 


## NFD




### 




 class
 

 tokenizers.normalizers.
 

 NFD




 (
 

 )
 




 NFD Unicode Normalizer
 


## NFKC




### 




 class
 

 tokenizers.normalizers.
 

 NFKC




 (
 

 )
 




 NFKC Unicode Normalizer
 


## NFKD




### 




 class
 

 tokenizers.normalizers.
 

 NFKD




 (
 

 )
 




 NFKD Unicode Normalizer
 


## Nmt




### 




 class
 

 tokenizers.normalizers.
 

 Nmt




 (
 

 )
 




 Nmt normalizer
 


## Normalizer




### 




 class
 

 tokenizers.normalizers.
 

 Normalizer




 (
 

 )
 




 Base class for all normalizers
 



 This class is not supposed to be instantiated directly. Instead, any implementation of a
Normalizer will return an instance of this class when instantiated.
 



#### 




 normalize




 (
 


 normalized
 




 )
 


 Parameters
 




* **normalized** 
 (
 `NormalizedString` 
 ) —
The normalized string on which to apply this
 [Normalizer](/docs/tokenizers/v0.13.4.rc2/en/api/normalizers#tokenizers.normalizers.Normalizer)


 Normalize a
 `NormalizedString` 
 in-place
 



 This method allows to modify a
 `NormalizedString` 
 to
keep track of the alignment information. If you just want to see the result
of the normalization on a raw string, you can use
 `normalize_str()` 


#### 




 normalize\_str




 (
 


 sequence
 




 )
 

 →
 



`str` 


 Parameters
 




* **sequence** 
 (
 `str` 
 ) —
A string to normalize




 Returns
 




`str` 




 A string after normalization
 



 Normalize the given string
 



 This method provides a way to visualize the effect of a
 [Normalizer](/docs/tokenizers/v0.13.4.rc2/en/api/normalizers#tokenizers.normalizers.Normalizer) 
 but it does not keep track of the alignment
information. If you need to get/convert offsets, you can use
 `normalize()` 


## Precompiled




### 




 class
 

 tokenizers.normalizers.
 

 Precompiled




 (
 


 precompiled\_charsmap
 




 )
 




 Precompiled normalizer
Don’t use manually it is used for compatiblity for SentencePiece.
 


## Replace




### 




 class
 

 tokenizers.normalizers.
 

 Replace




 (
 


 pattern
 


 content
 




 )
 




 Replace normalizer
 


## Sequence




### 




 class
 

 tokenizers.normalizers.
 

 Sequence




 (
 

 )
 


 Parameters
 




* **normalizers** 
 (
 `List[Normalizer]` 
 ) —
A list of Normalizer to be run as a sequence


 Allows concatenating multiple other Normalizer as a Sequence.
All the normalizers run in sequence in the given order
 


## Strip




### 




 class
 

 tokenizers.normalizers.
 

 Strip




 (
 


 left
 
 = True
 




 right
 
 = True
 



 )
 




 Strip normalizer
 


## StripAccents




### 




 class
 

 tokenizers.normalizers.
 

 StripAccents




 (
 

 )
 




 StripAccents normalizer