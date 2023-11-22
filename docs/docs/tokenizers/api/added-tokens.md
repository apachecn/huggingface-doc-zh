# Added Tokens

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/api/added-tokens>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/api/added-tokens>


## AddedToken




### 




 class
 

 tokenizers.
 

 AddedToken




 (
 


 content
 


 single\_word
 
 = False
 




 lstrip
 
 = False
 




 rstrip
 
 = False
 




 normalized
 
 = True
 



 )
 


 Parameters
 




* **content** 
 (
 `str` 
 ) — The content of the token
* **single\_word** 
 (
 `bool` 
 , defaults to
 `False` 
 ) —
Defines whether this token should only match single words. If
 `True` 
 , this
token will never match inside of a word. For example the token
 `ing` 
 would match
on
 `tokenizing` 
 if this option is
 `False` 
 , but not if it is
 `True` 
.
The notion of ”
 *inside of a word* 
 ” is defined by the word boundaries pattern in
regular expressions (ie. the token should start and end with word boundaries).
* **lstrip** 
 (
 `bool` 
 , defaults to
 `False` 
 ) —
Defines whether this token should strip all potential whitespaces on its left side.
If
 `True` 
 , this token will greedily match any whitespace on its left. For
example if we try to match the token
 `[MASK]` 
 with
 `lstrip=True` 
 , in the text
 `"I saw a [MASK]"` 
 , we would match on
 `" [MASK]"` 
. (Note the space on the left).
* **rstrip** 
 (
 `bool` 
 , defaults to
 `False` 
 ) —
Defines whether this token should strip all potential whitespaces on its right
side. If
 `True` 
 , this token will greedily match any whitespace on its right.
It works just like
 `lstrip` 
 but on the right.
* **normalized** 
 (
 `bool` 
 , defaults to
 `True` 
 with —meth:
 *~tokenizers.Tokenizer.add\_tokens* 
 and
 `False` 
 with
 `add_special_tokens()` 
 ):
Defines whether this token should match against the normalized version of the input
text. For example, with the added token
 `"yesterday"` 
 , and a normalizer in charge of
lowercasing the text, the token could be extract from the input
 `"I saw a lion Yesterday"` 
.


 Represents a token that can be be added to a
 [Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 
.
It can have special options that defines the way it should behave.
 


 property
 

 content
 


 Get the content of this
 `AddedToken` 




 property
 

 lstrip
 


 Get the value of the
 `lstrip` 
 option
 



 property
 

 normalized
 


 Get the value of the
 `normalized` 
 option
 



 property
 

 rstrip
 


 Get the value of the
 `rstrip` 
 option
 



 property
 

 single\_word
 


 Get the value of the
 `single_word` 
 option