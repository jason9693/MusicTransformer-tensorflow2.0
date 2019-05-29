# Music Transformer: Generating Music with Long-Term Structure

- 2019 ICLR, Cheng-Zhi Anna Huang, Google Brain
- [paper link](https://arxiv.org/abs/1809.04281) 
- [paper review](https://github.com/SSUHan/PaparReviews/issues/13)



## Preprocessing

* In this repository using single track method (2nd method in paper.).

* I refered preprocess code from [performaceRNN re-built repository.](https://github.com/djosix/Performance-RNN-PyTorch) 

  ![](https://user-images.githubusercontent.com/11185336/51083282-cddfc300-175a-11e9-9341-4a9042b17c19.png)



## TF2.0 Trouble Shooting

### 1. tf.keras

 you can't use `tf.keras` directly in alpha ver. So you should import `from tensorflow.python import keras` ,then use `> keras.{methods}` 

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





```python
from tensorflow.python.keras.optimizer_v2.adam import Adam
```