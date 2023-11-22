# Tokenizer

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/api/tokenizer>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/api/tokenizer>


## Tokenizer




### 




 class
 

 tokenizers.
 

 Tokenizer




 (
 


 model
 




 )
 


 Parameters
 




* **model** 
 (
 [Model](/docs/tokenizers/v0.13.4.rc2/en/api/models#tokenizers.models.Model) 
 ) —
The core algorithm that this
 `Tokenizer` 
 should be using.


 A
 `Tokenizer` 
 works as a pipeline. It processes some raw text as input
and outputs an
 [Encoding](/docs/tokenizers/v0.13.4.rc2/en/api/encoding#tokenizers.Encoding) 
.
 


 property
 

 decoder
 


 The
 *optional* 
`Decoder` 
 in use by the Tokenizer
 



 property
 

 model
 


 The
 [Model](/docs/tokenizers/v0.13.4.rc2/en/api/models#tokenizers.models.Model) 
 in use by the Tokenizer
 



 property
 

 normalizer
 


 The
 *optional* 
[Normalizer](/docs/tokenizers/v0.13.4.rc2/en/api/normalizers#tokenizers.normalizers.Normalizer) 
 in use by the Tokenizer
 



 property
 

 padding
 




 Returns
 




 (
 `dict` 
 ,
 *optional* 
 )
 



 A dict with the current padding parameters if padding is enabled
 



 Get the current padding parameters
 



*Cannot be set, use* 
`enable_padding()` 
*instead* 




 property
 

 post\_processor
 


 The
 *optional* 
`PostProcessor` 
 in use by the Tokenizer
 



 property
 

 pre\_tokenizer
 


 The
 *optional* 
[PreTokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/pre-tokenizers#tokenizers.pre_tokenizers.PreTokenizer) 
 in use by the Tokenizer
 



 property
 

 truncation
 




 Returns
 




 (
 `dict` 
 ,
 *optional* 
 )
 



 A dict with the current truncation parameters if truncation is enabled
 



 Get the currently set truncation parameters
 



*Cannot set, use* 
`enable_truncation()` 
*instead* 


#### 




 add\_special\_tokens




 (
 


 tokens
 




 )
 

 →
 



`int` 


 Parameters
 




* **tokens** 
 (A
 `List` 
 of
 [AddedToken](/docs/tokenizers/v0.13.4.rc2/en/api/added-tokens#tokenizers.AddedToken) 
 or
 `str` 
 ) —
The list of special tokens we want to add to the vocabulary. Each token can either
be a string or an instance of
 [AddedToken](/docs/tokenizers/v0.13.4.rc2/en/api/added-tokens#tokenizers.AddedToken) 
 for more
customization.




 Returns
 




`int` 




 The number of tokens that were created in the vocabulary
 



 Add the given special tokens to the Tokenizer.
 



 If these tokens are already part of the vocabulary, it just let the Tokenizer know about
them. If they don’t exist, the Tokenizer creates them, giving them a new id.
 



 These special tokens will never be processed by the model (ie won’t be split into
multiple tokens), and they can be removed from the output when decoding.
 




#### 




 add\_tokens




 (
 


 tokens
 




 )
 

 →
 



`int` 


 Parameters
 




* **tokens** 
 (A
 `List` 
 of
 [AddedToken](/docs/tokenizers/v0.13.4.rc2/en/api/added-tokens#tokenizers.AddedToken) 
 or
 `str` 
 ) —
The list of tokens we want to add to the vocabulary. Each token can be either a
string or an instance of
 [AddedToken](/docs/tokenizers/v0.13.4.rc2/en/api/added-tokens#tokenizers.AddedToken) 
 for more customization.




 Returns
 




`int` 




 The number of tokens that were created in the vocabulary
 



 Add the given tokens to the vocabulary
 



 The given tokens are added only if they don’t already exist in the vocabulary.
Each token then gets a new attributed id.
 




#### 




 decode




 (
 


 ids
 


 skip\_special\_tokens
 
 = True
 



 )
 

 →
 



`str` 


 Parameters
 




* **ids** 
 (A
 `List/Tuple` 
 of
 `int` 
 ) —
The list of ids that we want to decode
* **skip\_special\_tokens** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether the special tokens should be removed from the decoded string




 Returns
 




`str` 




 The decoded string
 



 Decode the given list of ids back to a string
 



 This is used to decode anything coming back from a Language Model
 




#### 




 decode\_batch




 (
 


 sequences
 


 skip\_special\_tokens
 
 = True
 



 )
 

 →
 



`List[str]` 


 Parameters
 




* **sequences** 
 (
 `List` 
 of
 `List[int]` 
 ) —
The batch of sequences we want to decode
* **skip\_special\_tokens** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether the special tokens should be removed from the decoded strings




 Returns
 




`List[str]` 




 A list of decoded strings
 



 Decode a batch of ids back to their corresponding string
 




#### 




 enable\_padding




 (
 


 direction
 
 = 'right'
 




 pad\_id
 
 = 0
 




 pad\_type\_id
 
 = 0
 




 pad\_token
 
 = '[PAD]'
 




 length
 
 = None
 




 pad\_to\_multiple\_of
 
 = None
 



 )
 


 Parameters
 




* **direction** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `right` 
 ) —
The direction in which to pad. Can be either
 `right` 
 or
 `left`
* **pad\_to\_multiple\_of** 
 (
 `int` 
 ,
 *optional* 
 ) —
If specified, the padding length should always snap to the next multiple of the
given value. For example if we were going to pad witha length of 250 but
 `pad_to_multiple_of=8` 
 then we will pad to 256.
* **pad\_id** 
 (
 `int` 
 , defaults to 0) —
The id to be used when padding
* **pad\_type\_id** 
 (
 `int` 
 , defaults to 0) —
The type id to be used when padding
* **pad\_token** 
 (
 `str` 
 , defaults to
 `[PAD]` 
 ) —
The pad token to be used when padding
* **length** 
 (
 `int` 
 ,
 *optional* 
 ) —
If specified, the length at which to pad. If not specified we pad using the size of
the longest sequence in a batch.


 Enable the padding
 




#### 




 enable\_truncation




 (
 


 max\_length
 


 stride
 
 = 0
 




 strategy
 
 = 'longest\_first'
 




 direction
 
 = 'right'
 



 )
 


 Parameters
 




* **max\_length** 
 (
 `int` 
 ) —
The max length at which to truncate
* **stride** 
 (
 `int` 
 ,
 *optional* 
 ) —
The length of the previous first sequence to be included in the overflowing
sequence
* **strategy** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `longest_first` 
 ) —
The strategy used to truncation. Can be one of
 `longest_first` 
 ,
 `only_first` 
 or
 `only_second` 
.
* **direction** 
 (
 `str` 
 , defaults to
 `right` 
 ) —
Truncate direction


 Enable truncation
 




#### 




 encode




 (
 


 sequence
 


 pair
 
 = None
 




 is\_pretokenized
 
 = False
 




 add\_special\_tokens
 
 = True
 



 )
 

 →
 



[Encoding](/docs/tokenizers/v0.13.4.rc2/en/api/encoding#tokenizers.Encoding) 


 Parameters
 




* **sequence** 
 (
 `~tokenizers.InputSequence` 
 ) —
The main input sequence we want to encode. This sequence can be either raw
text or pre-tokenized, according to the
 `is_pretokenized` 
 argument:
 
	+ If
	 `is_pretokenized=False` 
	 :
	 `TextInputSequence`
	+ If
	 `is_pretokenized=True` 
	 :
	 `PreTokenizedInputSequence()`
* **pair** 
 (
 `~tokenizers.InputSequence` 
 ,
 *optional* 
 ) —
An optional input sequence. The expected format is the same that for
 `sequence` 
.
* **is\_pretokenized** 
 (
 `bool` 
 , defaults to
 `False` 
 ) —
Whether the input is already pre-tokenized
* **add\_special\_tokens** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to add the special tokens




 Returns
 




[Encoding](/docs/tokenizers/v0.13.4.rc2/en/api/encoding#tokenizers.Encoding) 




 The encoded result
 



 Encode the given sequence and pair. This method can process raw text sequences
as well as already pre-tokenized sequences.
 



 Example:
 


 Here are some examples of the inputs that are accepted:
 



```
encode("A single sequence")*
encode("A sequence", "And its pair")*
encode([ "A", "pre", "tokenized", "sequence" ], is_pretokenized=True)`
encode(
[ "A", "pre", "tokenized", "sequence" ], [ "And", "its", "pair" ],
is_pretokenized=True
)
```


#### 




 encode\_batch




 (
 


 input
 


 is\_pretokenized
 
 = False
 




 add\_special\_tokens
 
 = True
 



 )
 

 →
 



 A
 `List` 
 of [`~tokenizers.Encoding“]
 




 Parameters
 




* **input** 
 (A
 `List` 
 /`
 `Tuple` 
 of
 `~tokenizers.EncodeInput` 
 ) —
A list of single sequences or pair sequences to encode. Each sequence
can be either raw text or pre-tokenized, according to the
 `is_pretokenized` 
 argument:
 
	+ If
	 `is_pretokenized=False` 
	 :
	 `TextEncodeInput()`
	+ If
	 `is_pretokenized=True` 
	 :
	 `PreTokenizedEncodeInput()`
* **is\_pretokenized** 
 (
 `bool` 
 , defaults to
 `False` 
 ) —
Whether the input is already pre-tokenized
* **add\_special\_tokens** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to add the special tokens




 Returns
 




 A
 `List` 
 of [`~tokenizers.Encoding“]
 



 The encoded batch
 



 Encode the given batch of inputs. This method accept both raw text sequences
as well as already pre-tokenized sequences.
 



 Example:
 


 Here are some examples of the inputs that are accepted:
 



```
encode_batch([
"A single sequence",
("A tuple with a sequence", "And its pair"),
[ "A", "pre", "tokenized", "sequence" ],
([ "A", "pre", "tokenized", "sequence" ], "And its pair")
])
```


#### 




 from\_buffer




 (
 


 buffer
 




 )
 

 →
 



[Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 


 Parameters
 




* **buffer** 
 (
 `bytes` 
 ) —
A buffer containing a previously serialized
 [Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer)




 Returns
 




[Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 




 The new tokenizer
 



 Instantiate a new
 [Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 
 from the given buffer.
 




#### 




 from\_file




 (
 


 path
 




 )
 

 →
 



[Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 


 Parameters
 




* **path** 
 (
 `str` 
 ) —
A path to a local JSON file representing a previously serialized
 [Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer)




 Returns
 




[Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 




 The new tokenizer
 



 Instantiate a new
 [Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 
 from the file at the given path.
 




#### 




 from\_pretrained




 (
 


 identifier
 


 revision
 
 = 'main'
 




 auth\_token
 
 = None
 



 )
 

 →
 



[Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 


 Parameters
 




* **identifier** 
 (
 `str` 
 ) —
The identifier of a Model on the Hugging Face Hub, that contains
a tokenizer.json file
* **revision** 
 (
 `str` 
 , defaults to
 *main* 
 ) —
A branch or commit id
* **auth\_token** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 *None* 
 ) —
An optional auth token used to access private repositories on the
Hugging Face Hub




 Returns
 




[Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 




 The new tokenizer
 



 Instantiate a new
 [Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 
 from an existing file on the
Hugging Face Hub.
 




#### 




 from\_str




 (
 


 json
 




 )
 

 →
 



[Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 


 Parameters
 




* **json** 
 (
 `str` 
 ) —
A valid JSON string representing a previously serialized
 [Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer)




 Returns
 




[Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 




 The new tokenizer
 



 Instantiate a new
 [Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 
 from the given JSON string.
 




#### 




 get\_vocab




 (
 


 with\_added\_tokens
 
 = True
 



 )
 

 →
 



`Dict[str, int]` 


 Parameters
 




* **with\_added\_tokens** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to include the added tokens




 Returns
 




`Dict[str, int]` 




 The vocabulary
 



 Get the underlying vocabulary
 




#### 




 get\_vocab\_size




 (
 


 with\_added\_tokens
 
 = True
 



 )
 

 →
 



`int` 


 Parameters
 




* **with\_added\_tokens** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to include the added tokens




 Returns
 




`int` 




 The size of the vocabulary
 



 Get the size of the underlying vocabulary
 




#### 




 id\_to\_token




 (
 


 id
 




 )
 

 →
 



`Optional[str]` 


 Parameters
 




* **id** 
 (
 `int` 
 ) —
The id to convert




 Returns
 




`Optional[str]` 




 An optional token,
 `None` 
 if out of vocabulary
 



 Convert the given id to its corresponding token if it exists
 




#### 




 no\_padding




 (
 

 )
 




 Disable padding
 




#### 




 no\_truncation




 (
 

 )
 




 Disable truncation
 




#### 




 num\_special\_tokens\_to\_add




 (
 


 is\_pair
 




 )
 




 Return the number of special tokens that would be added for single/pair sentences.
:param is\_pair: Boolean indicating if the input would be a single sentence or a pair
:return:
 




#### 




 post\_process




 (
 


 encoding
 


 pair
 
 = None
 




 add\_special\_tokens
 
 = True
 



 )
 

 →
 



[Encoding](/docs/tokenizers/v0.13.4.rc2/en/api/encoding#tokenizers.Encoding) 


 Parameters
 




* **encoding** 
 (
 [Encoding](/docs/tokenizers/v0.13.4.rc2/en/api/encoding#tokenizers.Encoding) 
 ) —
The
 [Encoding](/docs/tokenizers/v0.13.4.rc2/en/api/encoding#tokenizers.Encoding) 
 corresponding to the main sequence.
* **pair** 
 (
 [Encoding](/docs/tokenizers/v0.13.4.rc2/en/api/encoding#tokenizers.Encoding) 
 ,
 *optional* 
 ) —
An optional
 [Encoding](/docs/tokenizers/v0.13.4.rc2/en/api/encoding#tokenizers.Encoding) 
 corresponding to the pair sequence.
* **add\_special\_tokens** 
 (
 `bool` 
 ) —
Whether to add the special tokens




 Returns
 




[Encoding](/docs/tokenizers/v0.13.4.rc2/en/api/encoding#tokenizers.Encoding) 




 The final post-processed encoding
 



 Apply all the post-processing steps to the given encodings.
 



 The various steps are:
 


1. Truncate according to the set truncation params (provided with
 `enable_truncation()` 
 )
2. Apply the
 `PostProcessor`
3. Pad according to the set padding params (provided with
 `enable_padding()` 
 )




#### 




 save




 (
 


 path
 


 pretty
 
 = True
 



 )
 


 Parameters
 




* **path** 
 (
 `str` 
 ) —
A path to a file in which to save the serialized tokenizer.
* **pretty** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether the JSON file should be pretty formatted.


 Save the
 [Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 
 to the file at the given path.
 




#### 




 to\_str




 (
 


 pretty
 
 = False
 



 )
 

 →
 



`str` 


 Parameters
 




* **pretty** 
 (
 `bool` 
 , defaults to
 `False` 
 ) —
Whether the JSON string should be pretty formatted.




 Returns
 




`str` 




 A string representing the serialized Tokenizer
 



 Gets a serialized string representing this
 [Tokenizer](/docs/tokenizers/v0.13.4.rc2/en/api/tokenizer#tokenizers.Tokenizer) 
.
 




#### 




 token\_to\_id




 (
 


 token
 




 )
 

 →
 



`Optional[int]` 


 Parameters
 




* **token** 
 (
 `str` 
 ) —
The token to convert




 Returns
 




`Optional[int]` 




 An optional id,
 `None` 
 if out of vocabulary
 



 Convert the given token to its corresponding id if it exists
 




#### 




 train




 (
 


 files
 


 trainer
 
 = None
 



 )
 


 Parameters
 




* **files** 
 (
 `List[str]` 
 ) —
A list of path to the files that we should use for training
* **trainer** 
 (
 `~tokenizers.trainers.Trainer` 
 ,
 *optional* 
 ) —
An optional trainer that should be used to train our Model


 Train the Tokenizer using the given files.
 



 Reads the files line by line, while keeping all the whitespace, even new lines.
If you want to train from data store in-memory, you can check
 `train_from_iterator()` 


#### 




 train\_from\_iterator




 (
 


 iterator
 


 trainer
 
 = None
 




 length
 
 = None
 



 )
 


 Parameters
 




* **iterator** 
 (
 `Iterator` 
 ) —
Any iterator over strings or list of strings
* **trainer** 
 (
 `~tokenizers.trainers.Trainer` 
 ,
 *optional* 
 ) —
An optional trainer that should be used to train our Model
* **length** 
 (
 `int` 
 ,
 *optional* 
 ) —
The total number of sequences in the iterator. This is used to
provide meaningful progress tracking


 Train the Tokenizer using the provided iterator.
 



 You can provide anything that is a Python Iterator
 


* A list of sequences
 `List[str]`
* A generator that yields
 `str` 
 or
 `List[str]`
* A Numpy array of strings
* …