import tensorflow as tf
import math as m
import tensorflow.contrib.eager as tfe
import numpy as np
tf.enable_eager_execution()


class MusicTransformer(tf.keras.Model):
    def __init__(self, embedding_dim = 256, vocab_size =240, num_layer =6,
                 max_seq = 2048, debug = False):
        super(MusicTransformer, self).__init__()
        self._debug = debug
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embed = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=max_seq)
        self.fc_layer = tf.keras.layers.Dense(self.vocab_size, activation=tf.nn.softmax)
        embed_sinusoid_list = [
            [
                m.sin(
                    m.pow(
                        (pos * 0.00001), i / self.embedding_dim
                    ) - m.pi * 0.5 * ((i + 1) % 2)
                )
                for i in range(self.embedding_dim)
            ]
            for pos in range(max_seq)
        ]
        print(embed_sinusoid_list)
        embed_sinusoid_list = np.array(embed_sinusoid_list)
        self.positional_embedding = tf.constant(embed_sinusoid_list, dtype=tf.float32)
        self.decoder_list = [
            (
                RelativeGlobalAttention(64),
                tf.keras.layers.Add(),
                RelativeGlobalAttention(64),
                tf.keras.layers.Add(),
                tf.keras.layers.Dense(self.embedding_dim, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(self.embedding_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.BatchNormalization()
            )
            for _ in range(self.num_layer)
        ]
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=None, mask=None):
        embedding = tf.add(self.embed(inputs) , tf.expand_dims(self.positional_embedding, 0))
        decoder_input = embedding

        for i in range(self.num_layer):
            decoder = self._decoder(
                decoder_input,
                tf.nn.relu,
                layer=i
            )
            decoder_input = decoder
        return self.fc_layer(self.flatten(decoder_input))

    def _decoder(self, input_tensor, activation=None, layer=0):
        if self._debug:
            print('[DEBUG]:{}'.format('decoder called'))
        dec_layers = self.decoder_list[layer]
        decoder1 = dec_layers[0]([input_tensor, input_tensor, input_tensor])# Assuming Dh = 64
        add_and_norm = dec_layers[1]([decoder1, input_tensor])
        #add_and_norm = dec_layers[6](add_and_norm)

        decoder2 = dec_layers[2]([add_and_norm, add_and_norm, add_and_norm])
        residual = dec_layers[3]([decoder2, add_and_norm])
        #residual = dec_layers[7](residual)

        FFN = dec_layers[4](residual)
        FFN = dec_layers[5](FFN)
        return FFN

    def processed_y(self, y: np.array):
        # print(y)
        # print(np.eye(self.vocab_size)[y])
        return np.eye(self.vocab_size)[y]



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


if __name__ == '__main__':
    mt = MusicTransformer(512)
    out = mt(tf.constant(shape=[10, 2048], value=0.0))
    print(out.shape)
    pass