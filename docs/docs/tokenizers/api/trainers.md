# Trainers

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/api/trainers>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/api/trainers>


## BpeTrainer




### 




 class
 

 tokenizers.trainers.
 

 BpeTrainer




 (
 

 )
 


 Parameters
 


# * **vocab\_size** 
 (
 `int` 
 ,
 *optional* 
 ) —
The size of the final vocabulary, including all tokens and alphabet.
* **min\_frequency** 
 (
 `int` 
 ,
 *optional* 
 ) —
The minimum frequency a pair should have in order to be merged.
* **show\_progress** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Whether to show progress bars while training.
* **special\_tokens** 
 (
 `List[Union[str, AddedToken]]` 
 ,
 *optional* 
 ) —
A list of special tokens the model should know of.
* **limit\_alphabet** 
 (
 `int` 
 ,
 *optional* 
 ) —
The maximum different characters to keep in the alphabet.
* **initial\_alphabet** 
 (
 `List[str]` 
 ,
 *optional* 
 ) —
A list of characters to include in the initial alphabet, even
if not seen in the training dataset.
If the strings contain more than one character, only the first one
is kept.
* **continuing\_subword\_prefix** 
 (
 `str` 
 ,
 *optional* 
 ) —
A prefix to be used for every subword that is not a beginning-of-word.
* **end\_of\_word\_suffix** 
 (
 `str` 
 ,
 *optional* 
 ) —
A suffix to be used for every subword that is a end-of-word.
* **max\_token\_length** 
 (
 `int` 
 ,
 *optional* 
 ) —
Prevents creating tokens longer than the specified size.
This can help with reducing polluting your vocabulary with
highly repetitive tokens like
 *======* 
 for wikipedia
> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/tokenizers/api/trainers>
>
> 原始地址：<https://huggingface.co/docs/tokenizers/api/trainers>


 Trainer capable of training a BPE model
 


## UnigramTrainer




### 




 class
 

 tokenizers.trainers.
 

 UnigramTrainer




 (
 


 vocab\_size
 
 = 8000
 




 show\_progress
 
 = True
 




 special\_tokens
 
 = []
 




 shrinking\_factor
 
 = 0.75
 




 unk\_token
 
 = None
 




 max\_piece\_length
 
 = 16
 




 n\_sub\_iterations
 
 = 2
 



 )
 


 Parameters
 




* **vocab\_size** 
 (
 `int` 
 ) —
The size of the final vocabulary, including all tokens and alphabet.
* **show\_progress** 
 (
 `bool` 
 ) —
Whether to show progress bars while training.
* **special\_tokens** 
 (
 `List[Union[str, AddedToken]]` 
 ) —
A list of special tokens the model should know of.
* **initial\_alphabet** 
 (
 `List[str]` 
 ) —
A list of characters to include in the initial alphabet, even
if not seen in the training dataset.
If the strings contain more than one character, only the first one
is kept.
* **shrinking\_factor** 
 (
 `float` 
 ) —
The shrinking factor used at each step of the training to prune the
vocabulary.
* **unk\_token** 
 (
 `str` 
 ) —
The token used for out-of-vocabulary tokens.
* **max\_piece\_length** 
 (
 `int` 
 ) —
The maximum length of a given token.
* **n\_sub\_iterations** 
 (
 `int` 
 ) —
The number of iterations of the EM algorithm to perform before
pruning the vocabulary.


 Trainer capable of training a Unigram model
 


## WordLevelTrainer




### 




 class
 

 tokenizers.trainers.
 

 WordLevelTrainer




 (
 

 )
 


 Parameters
 




* **vocab\_size** 
 (
 `int` 
 ,
 *optional* 
 ) —
The size of the final vocabulary, including all tokens and alphabet.
* **min\_frequency** 
 (
 `int` 
 ,
 *optional* 
 ) —
The minimum frequency a pair should have in order to be merged.
* **show\_progress** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Whether to show progress bars while training.
* **special\_tokens** 
 (
 `List[Union[str, AddedToken]]` 
 ) —
A list of special tokens the model should know of.


 Trainer capable of training a WorldLevel model
 


## WordPieceTrainer




### 




 class
 

 tokenizers.trainers.
 

 WordPieceTrainer




 (
 


 vocab\_size
 
 = 30000
 




 min\_frequency
 
 = 0
 




 show\_progress
 
 = True
 




 special\_tokens
 
 = []
 




 limit\_alphabet
 
 = None
 




 initial\_alphabet
 
 = []
 




 continuing\_subword\_prefix
 
 = '##'
 




 end\_of\_word\_suffix
 
 = None
 



 )
 


 Parameters
 




* **vocab\_size** 
 (
 `int` 
 ,
 *optional* 
 ) —
The size of the final vocabulary, including all tokens and alphabet.
* **min\_frequency** 
 (
 `int` 
 ,
 *optional* 
 ) —
The minimum frequency a pair should have in order to be merged.
* **show\_progress** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Whether to show progress bars while training.
* **special\_tokens** 
 (
 `List[Union[str, AddedToken]]` 
 ,
 *optional* 
 ) —
A list of special tokens the model should know of.
* **limit\_alphabet** 
 (
 `int` 
 ,
 *optional* 
 ) —
The maximum different characters to keep in the alphabet.
* **initial\_alphabet** 
 (
 `List[str]` 
 ,
 *optional* 
 ) —
A list of characters to include in the initial alphabet, even
if not seen in the training dataset.
If the strings contain more than one character, only the first one
is kept.
* **continuing\_subword\_prefix** 
 (
 `str` 
 ,
 *optional* 
 ) —
A prefix to be used for every subword that is not a beginning-of-word.
* **end\_of\_word\_suffix** 
 (
 `str` 
 ,
 *optional* 
 ) —
A suffix to be used for every subword that is a end-of-word.


 Trainer capable of training a WordPiece model