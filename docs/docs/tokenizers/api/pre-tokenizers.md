# Pre-tokenizers

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/api/pre-tokenizers>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/api/pre-tokenizers>


## BertPreTokenizer




### 




 class
 

 tokenizers.pre\_tokenizers.
 

 BertPreTokenizer




 (
 

 )
 




 BertPreTokenizer
 



 This pre-tokenizer splits tokens on spaces, and also on punctuation.
Each occurence of a punctuation character will be treated separately.
 


## ByteLevel




### 




 class
 

 tokenizers.pre\_tokenizers.
 

 ByteLevel




 (
 


 add\_prefix\_space
 
 = True
 




 use\_regex
 
 = True
 



 )
 


 Parameters
 




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
* **use\_regex** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Set this to
 `False` 
 to prevent this
 *pre\_tokenizer* 
 from using
the GPT2 specific regexp for spliting on whitespace.


 ByteLevel PreTokenizer
 



 This pre-tokenizer takes care of replacing all bytes of the given string
with a corresponding representation, as well as splitting into words.
 



#### 




 alphabet




 (
 

 )
 

 →
 



`List[str]` 



 Returns
 




`List[str]` 




 A list of characters that compose the alphabet
 



 Returns the alphabet used by this PreTokenizer.
 



 Since the ByteLevel works as its name suggests, at the byte level, it
encodes each byte value to a unique visible character. This means that there is a
total of 256 different characters composing this alphabet.
 


## CharDelimiterSplit




### 




 class
 

 tokenizers.pre\_tokenizers.
 

 CharDelimiterSplit




 (
 

 )
 




 This pre-tokenizer simply splits on the provided char. Works like
 `.split(delimiter)` 


## Digits




### 




 class
 

 tokenizers.pre\_tokenizers.
 

 Digits




 (
 


 individual\_digits
 
 = False
 



 )
 


 Parameters
 




* **individual\_digits** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —


 This pre-tokenizer simply splits using the digits in separate tokens
 


 If set to True, digits will each be separated as follows:
 



```
"Call 123 please" -> "Call ", "1", "2", "3", " please"
```



 If set to False, digits will grouped as follows:
 



```
"Call 123 please" -> "Call ", "123", " please"
```


## Metaspace




### 




 class
 

 tokenizers.pre\_tokenizers.
 

 Metaspace




 (
 


 replacement
 
 = '\_'
 




 add\_prefix\_space
 
 = True
 



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


 Metaspace pre-tokenizer
 



 This pre-tokenizer replaces any whitespace by the provided replacement character.
It then tries to split on these spaces.
 


## PreTokenizer




### 




 class
 

 tokenizers.pre\_tokenizers.
 

 PreTokenizer




 (
 

 )
 




 Base class for all pre-tokenizers
 



 This class is not supposed to be instantiated directly. Instead, any implementation of a
PreTokenizer will return an instance of this class when instantiated.
 



#### 




 pre\_tokenize




 (
 


 pretok
 




 )
 


 Parameters
 




* **pretok** 
 (
 `~tokenizers.PreTokenizedString) -- The pre-tokenized string on which to apply this :class:` 
 ~tokenizers.pre\_tokenizers.PreTokenizer`


 Pre-tokenize a
 `~tokenizers.PyPreTokenizedString` 
 in-place
 



 This method allows to modify a
 `PreTokenizedString` 
 to
keep track of the pre-tokenization, and leverage the capabilities of the
 `PreTokenizedString` 
. If you just want to see the result of
the pre-tokenization of a raw string, you can use
 `pre_tokenize_str()` 


#### 




 pre\_tokenize\_str




 (
 


 sequence
 




 )
 

 →
 



`List[Tuple[str, Offsets]]` 


 Parameters
 




* **sequence** 
 (
 `str` 
 ) —
A string to pre-tokeize




 Returns
 




`List[Tuple[str, Offsets]]` 




 A list of tuple with the pre-tokenized parts and their offsets
 



 Pre tokenize the given string
 



 This method provides a way to visualize the effect of a
 [PreTokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/pre-tokenizers#tokenizers.pre_tokenizers.PreTokenizer) 
 but it does not keep track of the
alignment, nor does it provide all the capabilities of the
 `PreTokenizedString` 
. If you need some of these, you can use
 `pre_tokenize()` 


## Punctuation




### 




 class
 

 tokenizers.pre\_tokenizers.
 

 Punctuation




 (
 


 behavior
 
 = 'isolated'
 



 )
 


 Parameters
 




* **behavior** 
 (
 `SplitDelimiterBehavior` 
 ) —
The behavior to use when splitting.
Choices: “removed”, “isolated” (default), “merged\_with\_previous”, “merged\_with\_next”,
“contiguous”


 This pre-tokenizer simply splits on punctuation as individual characters.
 


## Sequence




### 




 class
 

 tokenizers.pre\_tokenizers.
 

 Sequence




 (
 


 pretokenizers
 




 )
 




 This pre-tokenizer composes other pre\_tokenizers and applies them in sequence
 


## Split




### 




 class
 

 tokenizers.pre\_tokenizers.
 

 Split




 (
 


 pattern
 


 behavior
 


 invert
 
 = False
 



 )
 


 Parameters
 




* **pattern** 
 (
 `str` 
 or
 `Regex` 
 ) —
A pattern used to split the string. Usually a string or a a regex built with
 *tokenizers.Regex*
* **behavior** 
 (
 `SplitDelimiterBehavior` 
 ) —
The behavior to use when splitting.
Choices: “removed”, “isolated”, “merged\_with\_previous”, “merged\_with\_next”,
“contiguous”
* **invert** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether to invert the pattern.


 Split PreTokenizer
 



 This versatile pre-tokenizer splits using the provided pattern and
according to the provided behavior. The pattern can be inverted by
making use of the invert flag.
 


## UnicodeScripts




### 




 class
 

 tokenizers.pre\_tokenizers.
 

 UnicodeScripts




 (
 

 )
 




 This pre-tokenizer splits on characters that belong to different language family
It roughly follows
 <https://github.com/google/sentencepiece/blob/master/data/Scripts.txt>
 Actually Hiragana and Katakana are fused with Han, and 0x30FC is Han too.
This mimicks SentencePiece Unigram implementation.
 


## Whitespace




### 




 class
 

 tokenizers.pre\_tokenizers.
 

 Whitespace




 (
 

 )
 




 This pre-tokenizer simply splits using the following regex:
 `\w+|[^\w\s]+` 


## WhitespaceSplit




### 




 class
 

 tokenizers.pre\_tokenizers.
 

 WhitespaceSplit




 (
 

 )
 




 This pre-tokenizer simply splits on the whitespace. Works like
 `.split()`