import tensorflow as tf
import math as m
from tensorflow.python import keras
import numpy as np
import math


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
        embed_sinusoid_list = np.array([[
            [
                m.sin(
                    pos * m.exp(-m.log(10000)*i/embedding_dim) * m.exp(m.log(10000)/embedding_dim * (i%2)) + 0.5*m.pi*(i%2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]])
        self.positional_embedding = tf.constant(embed_sinusoid_list, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return tf.add(inputs,self.positional_embedding)


class DynamicPositionEmbedding(keras.layers.Layer):
    def __init__(self, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.embed_sinusoid_list = np.array([[
            [
                m.sin(
                    pos * m.exp(-m.log(10000) * i / self.embedding_dim) * m.exp(
                        m.log(10000) / self.embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
                )
                for i in range(self.embedding_dim)
            ]
            for pos in range(input_shape[1])
        ]])
        self.positional_embedding = tf.constant(self.embed_sinusoid_list, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return tf.add(inputs, self.positional_embedding)


class RelativeGlobalAttention(keras.layers.Layer):
    """
    from Music Transformer ( Huang et al, 2018 )
    [paper link](https://arxiv.org/pdf/1809.04281.pdf)
    """
    def __init__(self, h=4, d=256, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = keras.layers.Dense(int(self.d/2))
        self.Wk = keras.layers.Dense(int(self.d/2))
        self.Wv = keras.layers.Dense(int(self.d))
        self.fc = keras.layers.Dense(d)

    def build(self, input_shape):
        shape_q = input_shape[0][1]
        shape_k = input_shape[1][1]
        self.len_k = input_shape[1][1]
        self.max_seq = max(input_shape[0][1], input_shape[1][1], input_shape[2][1])
        self.E = self.add_variable('emb', shape=[min(shape_q, shape_k), int(self.dh/2)])
        # self.E = tf.pad(self.E, [[self.max_seq - self.E.shape[0], 0], [0, 0]])
        # print(self.E)

    def call(self, inputs, mask=None, **kwargs):
        """
        :param inputs: a list of tensors. i.e) [Q, K, V]
        :param mask: mask tensor
        :param kwargs:
        :return: final tensor ( output of attention )
        """

        q = inputs[0]
        q = self.Wq(q)
        q = tf.reshape(q, (q.shape[0], q.shape[1], self.h, -1))
        q = tf.transpose(q, (0, 2, 1, 3))  # batch, h, seq, dh

        k = inputs[1]
        k = self.Wk(k)
        k = tf.reshape(k, (k.shape[0], k.shape[1], self.h, -1))
        k = tf.transpose(k, (0, 2, 1, 3))

        v = inputs[2]
        v = self.Wv(v)
        v = tf.reshape(v, (v.shape[0], v.shape[1], self.h, -1))
        v = tf.transpose(v, (0, 2, 1, 3))

        E = self.E
        QE = tf.einsum('bhld,md->bhlm', q, E)

        Srel = self._skewing(QE)
        Kt = tf.transpose(k,[0, 1, 3, 2])
        QKt = tf.matmul(q, Kt)

        if mask is not None:
            QKt += (tf.cast(mask, tf.float32) * -1e9)
            pass

        attention = tf.nn.softmax((QKt + Srel), -1) / math.sqrt(self.dh)
        attention = tf.matmul(attention, v)

        out = tf.transpose(attention, (0, 2, 1, 3))
        out = tf.reshape(out, (-1, self.max_seq, self.d))

        out = self.fc(out)
        return out

    def _skewing(self, tensor: tf.Tensor):
        padded = tf.pad(tensor, [[0, 0], [0,0], [0, 0], [1, 0]])
        reshaped = tf.reshape(padded, shape=[-1, padded.shape[1], padded.shape[-1], padded.shape[-2]])
        Srel = reshaped[:, :, 1:, :]
        return Srel


class View1D(keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return inputs[:,self.axis]


class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, rate=0.1, h=16):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.rga = RelativeGlobalAttention(h=h, d=d_model)

        self.FFN_pre = keras.layers.Dense(512, activation=tf.nn.relu)
        self.FFN_suf = keras.layers.Dense(self.d_model)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, x, mask=None, training=False, **kwargs):

        attn_out = self.rga([x,x,x], mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out+x)

        ffn_out = self.FFN_pre(out1)
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(out1+ffn_out)
        return out2


class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, rate=0.1, h=16):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.rga = RelativeGlobalAttention(d=d_model, h=h)
        self.rga2 = RelativeGlobalAttention(d=d_model, h=h)

        self.FFN_pre = keras.layers.Dense(512, activation=tf.nn.relu)
        self.FFN_suf = keras.layers.Dense(self.d_model)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)

    def call(self, x, encode_out, mask=None, lookup_mask=None, training=False, **kwargs):

        attn_out = self.rga([x, x, x], mask=lookup_mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out+x)

        attn_out2 = self.rga2([out1, encode_out, encode_out], mask=mask)
        attn_out2 = self.dropout2(attn_out2)
        attn_out2 = self.layernorm2(out1+attn_out2)

        ffn_out = self.FFN_pre(out1)
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout3(ffn_out)
        out = self.layernorm3(attn_out2+ffn_out)
        return out


class Encoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, input_vocab_size,
                 rate=0.1, max_len=None):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = keras.layers.Embedding(input_vocab_size, d_model)
        if max_len is not None:
            self.pos_encoding = PositionEmbedding(max_seq=max_len, embedding_dim=self.d_model)
        else:
            self.pos_encoding = DynamicPositionEmbedding(self.d_model)

        self.enc_layers = [EncoderLayer(d_model, rate, h=4)
                           for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, mask=None, training=False):
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training=training)
        return x  # (batch_size, input_seq_len, d_model)


class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, input_vocab_size,
                 rate=0.1, max_len=None):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(input_vocab_size, d_model)
        if max_len is not None:
            self.pos_encoding = PositionEmbedding(max_seq=max_len, embedding_dim=self.d_model)
        else:
            self.pos_encoding = DynamicPositionEmbedding(self.d_model)
        self.dec_layers = [DecoderLayer(d_model, rate, h=4)
                           for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, enc_output, mask, lookup_mask, training):
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, lookup_mask=lookup_mask, mask=mask, training=training)
        return x  # (batch_size, input_seq_len, d_model)


if __name__ == '__main__':
    rga = RelativeGlobalAttention(d=9, h=1)
    result = rga([

        tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32),
        tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32),
        tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32),
        ], mask=tf.sequence_mask(range(3), 3, dtype=tf.int32))

    print(result)

