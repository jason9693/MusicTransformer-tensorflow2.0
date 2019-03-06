import tensorflow as tf
import math as m

class ExpandDims(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.expand_dims(inputs, axis=self.axis)

class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_seq, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        embed_sinusoid_list = [[
            [
                m.sin(
                    m.pow(
                        (pos * 0.00001), i / embedding_dim
                    ) - m.pi * 0.5 * ((i + 1) % 2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]]
        self.positional_embedding = tf.constant(embed_sinusoid_list, dtype=tf.float32)


    def call(self, inputs, **kwargs):
        return tf.add(inputs,self.positional_embedding)

class RelativeGlobalAttention(tf.keras.layers.Layer):
    ''''
    from Music Transformer ( Huang et al, 2018 )
    [paper link](https://arxiv.org/pdf/1809.04281.pdf)
    '''
    def __init__(self ,Dh , D=256, **kwargs):
        super().__init__(**kwargs)
        self.Dh = float(Dh)
        self.D = D
        self.Wq = self.add_variable("Wq", shape=[int(self.D), int(self.D)])
        self.Wk = self.add_variable("Wk", shape=[int(self.D), int(self.D)])
        self.Wv = self.add_variable("Wv", shape=[int(self.D), int(self.D)])


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

        E = K - Q
        E = tf.transpose(E,[0,2,1])

        QE = tf.matmul(Q,E)
        Srel = self._skewing(QE)
        Kt = tf.transpose(K,[0,2,1])
        QKt = tf.matmul(Q,Kt)

        attention = tf.nn.softmax((QKt + Srel) / tf.sqrt(self.Dh))
        attention = tf.matmul(attention, V)

        return attention

    def _skewing(self, tensor: tf.Tensor):

        pad = tf.zeros_like(tensor)
        pad = pad[:,:,0]
        pad = tf.expand_dims(pad,2)
        cat = tf.concat([pad, tensor], 2)

        reshaped = tf.reshape(cat, shape=[-1, cat.shape[2], cat.shape[1]])

        Srel = tf.slice(reshaped, [0,1,0], [-1, reshaped.shape[2], reshaped.shape[2]])

        return Srel


class View1D(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis


    def call(self, inputs, **kwargs):
        return inputs[:,self.axis]