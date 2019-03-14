import tensorflow as tf
import math as m
from tensorflow.python import keras
import numpy as np

def sinusoid(max_seq, embedding_dim):
    return np.array([[
        [
            m.sin(
                pos * m.exp(-m.log(10000) * i / embedding_dim) * m.exp(
                    m.log(10000) / embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
            )
            for i in range(embedding_dim)
        ]
        for pos in range(max_seq)
    ]])


class ExpandDims(keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.expand_dims(inputs, axis=self.axis)

class PositionEmbedding(keras.layers.Layer):
    def __init__(self, max_seq, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        embed_sinusoid_list = np.array( [[
            [
                m.sin(
                    pos * m.exp(-m.log(10000)*i/embedding_dim) * m.exp(m.log(10000)/embedding_dim * (i%2)) + 0.5*m.pi*(i%2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]] )
        self.positional_embedding = tf.constant(embed_sinusoid_list, dtype=tf.float32)

    def _boolean_calculate(self, i):
        return int(i%2)

    def call(self, inputs, **kwargs):
        return tf.add(inputs,self.positional_embedding)


class RelativeGlobalAttention(keras.layers.Layer):
    ''''
    from Music Transformer ( Huang et al, 2018 )
    [paper link](https://arxiv.org/pdf/1809.04281.pdf)
    '''
    def __init__(self ,Dh , D=256, max_seq = 2048,**kwargs):
        super().__init__(**kwargs)
        self.Dh = float(Dh)
        self.D = D
        self.Wq = self.add_variable("Wq", shape=[int(self.D), int(self.D/2)])
        self.Wk = self.add_variable("Wk", shape=[int(self.D), int(self.D/2)])
        self.Wv = self.add_variable("Wv", shape=[int(self.D), int(self.D)])
        self.EmbeddingVar = self.add_variable('emb', shape=[max_seq, int(self.D/2)])
        self.max_seq = max_seq


    def call(self, inputs, **kwargs):
        '''
        :param inputs: a list of tensors. i.e) [Q, K, V]
        :param kwargs:
        :return: final tensor ( output of attention )
        '''

        inputQ = inputs[0]
        inputK = inputs[1]
        inputV = inputs[2]

        Q = tf.tensordot(inputQ, self.Wq, [[2],[0]])
        K = tf.tensordot(inputK, self.Wk, [[2],[0]])
        V = tf.tensordot(inputV, self.Wv, [[2],[0]])

        # mat = [self.max_seq - 1 - i for i in range(inputs[0].shape[1])]
        # mat = tf.constant(mat, dtype=tf.int32)

        E = self.EmbeddingVar
        #tf.nn.embedding_lookup(self.EmbeddingVar, mat)
        E = tf.transpose(E,[1,0])

        QE = tf.tensordot(Q,E, [[2],[0]])
        Srel = self._skewing(QE)
        Kt = tf.transpose(K,[0,2,1])
        QKt = tf.matmul(Q,Kt)

        attention = tf.nn.softmax((QKt + Srel) / tf.sqrt(self.Dh))
        attention = tf.matmul(attention, V)

        return attention

    def _skewing(self, tensor: tf.Tensor):

        pad = tf.zeros_like(tensor[:,:,0])
        pad = tf.expand_dims(pad,2)
        cat = tf.concat([pad, tensor], 2)
        reshaped = tf.reshape(cat, shape=[-1, cat.shape[2], cat.shape[1]])

        Srel = tf.slice(reshaped, [0,1,0], [-1, -1, -1])

        return Srel


class View1D(keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis


    def call(self, inputs, **kwargs):
        return inputs[:,self.axis]

# class SeqLoss(keras.losses.CategoricalCrossentropy):
#     def __init__(self, vocab_size):
#         super(SeqLoss, self).__init__()
#         self.len_word = vocab_size
#         pass
#
#     def call(self, y_true, y_pred):
#         print(y_true)
#         print(y_pred)
#         y_true = np.array(y_true, dtype=np.int)
#         y_true = self.processed_y(y_true)
#         return super().call(y_true, y_pred)
#
#     def processed_y(self, y: np.array):
#         return np.eye(self.len_word)[y]





if __name__ == '__main__':
    pass
    # loss = SeqLoss(240)
    # print(loss.processed_y(np.zeros([10,2048], dtype=np.int)).shape)
    # mock = np.ones([10, 2048], dtype=np.int)
    # print(loss(mock, mock))

    embedding_dim = 512
    max_seq = 2048

    embed_sinusoid_list = np.array([[
        [
            m.sin(
                pos * m.exp(-m.log(10000) * i / embedding_dim) * m.exp(
                    m.log(10000) / embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
            )
            for i in range(embedding_dim)
        ]
        for pos in range(max_seq)
    ]])

    # embed_sinusoid_list = [
    #     [
    #         m.sin(
    #             m.pow(
    #                 (pos * 0.00001), i / embedding_dim
    #             ) - m.pi * 0.5 * ((i + 1) % 2)
    #         )
    #         for i in range(embedding_dim)
    #     ]
    #     for pos in range(max_seq)
    # ]

    import matplotlib.pyplot as plt
    plt.plot(embed_sinusoid_list[0,:,:])
    plt.show()

