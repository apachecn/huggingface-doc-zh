# Models

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/api/models>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/api/models>


## BPE




### 




 class
 

 tokenizers.models.
 

 BPE




 (
 


 vocab
 
 = None
 




 merges
 
 = None
 




 cache\_capacity
 
 = None
 




 dropout
 
 = None
 




 unk\_token
 
 = None
 




 continuing\_subword\_prefix
 
 = None
 




 end\_of\_word\_suffix
 
 = None
 




 fuse\_unk
 
 = None
 




 byte\_fallback
 
 = False
 



 )
 


 Parameters
 




* **vocab** 
 (
 `Dict[str, int]` 
 ,
 *optional* 
 ) —
A dictionnary of string keys and their ids
 `{"am": 0,...}`
* **merges** 
 (
 `List[Tuple[str, str]]` 
 ,
 *optional* 
 ) —
A list of pairs of tokens (
 `Tuple[str, str]` 
 )
 `[("a", "b"),...]`
* **cache\_capacity** 
 (
 `int` 
 ,
 *optional* 
 ) —
The number of words that the BPE cache can contain. The cache allows
to speed-up the process by keeping the result of the merge operations
for a number of words.
* **dropout** 
 (
 `float` 
 ,
 *optional* 
 ) —
A float between 0 and 1 that represents the BPE dropout to use.
* **unk\_token** 
 (
 `str` 
 ,
 *optional* 
 ) —
The unknown token to be used by the model.
* **continuing\_subword\_prefix** 
 (
 `str` 
 ,
 *optional* 
 ) —
The prefix to attach to subword units that don’t represent a beginning of word.
* **end\_of\_word\_suffix** 
 (
 `str` 
 ,
 *optional* 
 ) —
The suffix to attach to subword units that represent an end of word.
* **fuse\_unk** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Whether to fuse any subsequent unknown tokens into a single one
* **byte\_fallback** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Whether to use spm byte-fallback trick (defaults to False)


 An implementation of the BPE (Byte-Pair Encoding) algorithm
 



#### 




 from\_file




 (
 


 vocab
 


 merge
 


 \*\*kwargs
 




 )
 

 →
 



[BPE](/docs/tokenizers/v0.13.4.rc2/en/api/models#tokenizers.models.BPE) 


 Parameters
 




* **vocab** 
 (
 `str` 
 ) —
The path to a
 `vocab.json` 
 file
* **merges** 
 (
 `str` 
 ) —
The path to a
 `merges.txt` 
 file




 Returns
 




[BPE](/docs/tokenizers/v0.13.4.rc2/en/api/models#tokenizers.models.BPE) 




 An instance of BPE loaded from these files
 



 Instantiate a BPE model from the given files.
 


 This method is roughly equivalent to doing:
 



```
vocab, merges = BPE.read_file(vocab_filename, merges_filename)
bpe = BPE(vocab, merges)
```




 If you don’t need to keep the
 `vocab, merges` 
 values lying around,
this method is more optimized than manually calling
 `read_file()` 
 to initialize a
 [BPE](/docs/tokenizers/v0.13.4.rc2/en/api/models#tokenizers.models.BPE) 


#### 




 read\_file




 (
 


 vocab
 


 merges
 




 )
 

 →
 



 A
 `Tuple` 
 with the vocab and the merges
 




 Parameters
 




* **vocab** 
 (
 `str` 
 ) —
The path to a
 `vocab.json` 
 file
* **merges** 
 (
 `str` 
 ) —
The path to a
 `merges.txt` 
 file




 Returns
 




 A
 `Tuple` 
 with the vocab and the merges
 



 The vocabulary and merges loaded into memory
 



 Read a
 `vocab.json` 
 and a
 `merges.txt` 
 files
 



 This method provides a way to read and parse the content of these files,
returning the relevant data structures. If you want to instantiate some BPE models
from memory, this method gives you the expected input from the standard files.
 


## Model




### 




 class
 

 tokenizers.models.
 

 Model




 (
 

 )
 




 Base class for all models
 



 The model represents the actual tokenization algorithm. This is the part that
will contain and manage the learned vocabulary.
 



 This class cannot be constructed directly. Please use one of the concrete models.
 



#### 




 get\_trainer




 (
 

 )
 

 →
 



`Trainer` 



 Returns
 




`Trainer` 




 The Trainer used to train this model
 



 Get the associated
 `Trainer` 




 Retrieve the
 `Trainer` 
 associated to this
 [Model](/docs/tokenizers/v0.13.4.rc2/en/api/models#tokenizers.models.Model) 
.
 




#### 




 id\_to\_token




 (
 


 id
 




 )
 

 →
 



`str` 


 Parameters
 




* **id** 
 (
 `int` 
 ) —
An ID to convert to a token




 Returns
 




`str` 




 The token associated to the ID
 



 Get the token associated to an ID
 




#### 




 save




 (
 


 folder
 


 prefix
 




 )
 

 →
 



`List[str]` 


 Parameters
 




* **folder** 
 (
 `str` 
 ) —
The path to the target folder in which to save the various files
* **prefix** 
 (
 `str` 
 ,
 *optional* 
 ) —
An optional prefix, used to prefix each file name




 Returns
 




`List[str]` 




 The list of saved files
 



 Save the current model
 



 Save the current model in the given folder, using the given prefix for the various
files that will get created.
Any file with the same name that already exists in this folder will be overwritten.
 




#### 




 token\_to\_id




 (
 


 tokens
 




 )
 

 →
 



`int` 


 Parameters
 




* **token** 
 (
 `str` 
 ) —
A token to convert to an ID




 Returns
 




`int` 




 The ID associated to the token
 



 Get the ID associated to a token
 




#### 




 tokenize




 (
 


 sequence
 




 )
 

 →
 



 A
 `List` 
 of
 `Token` 


 Parameters
 




* **sequence** 
 (
 `str` 
 ) —
A sequence to tokenize




 Returns
 




 A
 `List` 
 of
 `Token` 




 The generated tokens
 



 Tokenize a sequence
 


## Unigram




### 




 class
 

 tokenizers.models.
 

 Unigram




 (
 


 vocab
 


 unk\_id
 


 byte\_fallback
 




 )
 


 Parameters
 




* **vocab** 
 (
 `List[Tuple[str, float]]` 
 ,
 *optional* 
 ,
 *optional* 
 ) —
A list of vocabulary items and their relative score [(“am”, -0.2442),…]


 An implementation of the Unigram algorithm
 


## WordLevel




### 




 class
 

 tokenizers.models.
 

 WordLevel




 (
 


 vocab
 


 unk\_token
 




 )
 


 Parameters
 




* **vocab** 
 (
 `str` 
 ,
 *optional* 
 ) —
A dictionnary of string keys and their ids
 `{"am": 0,...}`
* **unk\_token** 
 (
 `str` 
 ,
 *optional* 
 ) —
The unknown token to be used by the model.


 An implementation of the WordLevel algorithm
 



 Most simple tokenizer model based on mapping tokens to their corresponding id.
 



#### 




 from\_file




 (
 


 vocab
 


 unk\_token
 




 )
 

 →
 



[WordLevel](/docs/tokenizers/v0.13.4.rc2/en/api/models#tokenizers.models.WordLevel) 


 Parameters
 




* **vocab** 
 (
 `str` 
 ) —
The path to a
 `vocab.json` 
 file




 Returns
 




[WordLevel](/docs/tokenizers/v0.13.4.rc2/en/api/models#tokenizers.models.WordLevel) 




 An instance of WordLevel loaded from file
 



 Instantiate a WordLevel model from the given file
 


 This method is roughly equivalent to doing:
 



```
vocab = WordLevel.read_file(vocab_filename)
wordlevel = WordLevel(vocab)
```




 If you don’t need to keep the
 `vocab` 
 values lying around, this method is
more optimized than manually calling
 `read_file()` 
 to
initialize a
 [WordLevel](/docs/tokenizers/v0.13.4.rc2/en/api/models#tokenizers.models.WordLevel) 


#### 




 read\_file




 (
 


 vocab
 




 )
 

 →
 



`Dict[str, int]` 


 Parameters
 




* **vocab** 
 (
 `str` 
 ) —
The path to a
 `vocab.json` 
 file




 Returns
 




`Dict[str, int]` 




 The vocabulary as a
 `dict` 




 Read a
 `vocab.json` 




 This method provides a way to read and parse the content of a vocabulary file,
returning the relevant data structures. If you want to instantiate some WordLevel models
from memory, this method gives you the expected input from the standard files.
 


## WordPiece




### 




 class
 

 tokenizers.models.
 

 WordPiece




 (
 


 vocab
 


 unk\_token
 


 max\_input\_chars\_per\_word
 




 )
 


 Parameters
 




* **vocab** 
 (
 `Dict[str, int]` 
 ,
 *optional* 
 ) —
A dictionnary of string keys and their ids
 `{"am": 0,...}`
* **unk\_token** 
 (
 `str` 
 ,
 *optional* 
 ) —
The unknown token to be used by the model.
* **max\_input\_chars\_per\_word** 
 (
 `int` 
 ,
 *optional* 
 ) —
The maximum number of characters to authorize in a single word.


 An implementation of the WordPiece algorithm
 



#### 




 from\_file




 (
 


 vocab
 


 \*\*kwargs
 




 )
 

 →
 



[WordPiece](/docs/tokenizers/v0.13.4.rc2/en/api/models#tokenizers.models.WordPiece) 


 Parameters
 




* **vocab** 
 (
 `str` 
 ) —
The path to a
 `vocab.txt` 
 file




 Returns
 




[WordPiece](/docs/tokenizers/v0.13.4.rc2/en/api/models#tokenizers.models.WordPiece) 




 An instance of WordPiece loaded from file
 



 Instantiate a WordPiece model from the given file
 


 This method is roughly equivalent to doing:
 



```
vocab = WordPiece.read_file(vocab_filename)
wordpiece = WordPiece(vocab)
```




 If you don’t need to keep the
 `vocab` 
 values lying around, this method is
more optimized than manually calling
 `read_file()` 
 to
initialize a
 [WordPiece](/docs/tokenizers/v0.13.4.rc2/en/api/models#tokenizers.models.WordPiece) 


#### 




 read\_file




 (
 


 vocab
 




 )
 

 →
 



`Dict[str, int]` 


 Parameters
 




* **vocab** 
 (
 `str` 
 ) —
The path to a
 `vocab.txt` 
 file




 Returns
 




`Dict[str, int]` 




 The vocabulary as a
 `dict` 




 Read a
 `vocab.txt` 
 file
 



 This method provides a way to read and parse the content of a standard
 *vocab.txt* 
 file as used by the WordPiece Model, returning the relevant data structures. If you
want to instantiate some WordPiece models from memory, this method gives you the
expected input from the standard files.