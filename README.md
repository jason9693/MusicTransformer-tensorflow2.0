# Music Transformer: Generating Music with Long-Term Structure

- 2019 ICLR, Cheng-Zhi Anna Huang, Google Brain
- Re-producer : Jason Yang ( Yang-Kichang )
- [paper link](https://arxiv.org/abs/1809.04281) 
- [paper review](https://github.com/SSUHan/PaparReviews/issues/13)

## Abstract

1. This Repository is perfectly cometible with **tensorflow 2.0**

## Contribution

1. Domain: Dramatically reduces the memory footprint, allowing it to scale to musical sequences on the order of minutes.
2. Algorithm: Reduced space complexity of Transformer from O(N^2D) to O(ND).

## Preprocessing

* In this repository using single track method (2nd method in paper.).

* If you want to get implementation of method 1, see [here](https://github.com/COMP6248-Reproducability-Challenge/music-transformer-comp6248) .

* I refered preprocess code from [performaceRNN re-built repository.](https://github.com/djosix/Performance-RNN-PyTorch) 

  * vocab size is smaller than paper.
    1. note on : 21 ~ 109
    2. note off : 21 ~ 109
    3. velocity : 32
    4. time shift : 32
  
  ![](https://user-images.githubusercontent.com/11185336/51083282-cddfc300-175a-11e9-9341-4a9042b17c19.png)



## Trainig

## Hyper Parameter

* learning rate : 0.0001
* head size : 4
* number of layers : 6
* seqence length : 2048
* embedding dim : 256 (dh = 256 / 4 = 64)
* batch size : 3

## Generate Music

* mt.generate() can generate music automatically.

```python
> from models import MusicTransformer
> mt = MusicTransformer(
  	embedding_dim=256, vocab_size=par.vocab_size, 
  	num_layer=6, 
  	max_seq=max_seq,
  	dropout=0.1,
  	debug=False
	)
> mt.generate(prior=[1,3,4,5], length=2048)
```



## TF2.0 Trouble Shooting

### 1. tf.keras

 you can't use `tf.keras` directly in alpha ver. So you should import `from tensorflow.python import keras` ,then use `> keras.{METHODS}` 

* example : 

  ```python
  > from tensorflow.python import keras 
  > dropout = keras.layers.Dropout(0.3)
  ```



### 2. tf.keras.optimizers.Adam() 

tf-2.0alpha currently not supported **keras.optimizers** as **version 2.** so, you can't use **optimizer.apply_gradients()**. So, you should import `from tensorflow.python.keras.optimizer_v2.adam import Adam` first.

* example:

  ```python
  > from tensorflow.python.keras.optimizer_v2.adam import Adam
  > optimizer = Adam(0.0001)
  ```



### 3. Keras Model Subclassing

current tf 2.0(alpha) , subclassed keras model can't use method **save(), summary(), fit()** and **save_weigths() with .h5 format**

