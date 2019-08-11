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


# TODO : reduce time complexity
class PositionEmbedding(keras.layers.Layer):
    def __init__(self, max_seq, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        embed_sinusoid_list = np.array([[
            [
                m.sin(
                    pos * m.exp(-m.log(10000) * i/embedding_dim) *
                    m.exp(m.log(10000)/embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]])
        self.positional_embedding = tf.constant(embed_sinusoid_list, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return tf.add(inputs, self.positional_embedding)


class PositionEmbeddingV2(keras.layers.Layer):
    def __init__(self, max_seq, embedding_dim, **kwargs):
        super(PositionEmbeddingV2, self).__init__(**kwargs)
        angle_rads = PositionEmbeddingV2.__get_angles(np.arange(max_seq)[:, np.newaxis],
                                                      np.arange(embedding_dim)[np.newaxis, :],
                                                      embedding_dim)

        # apply sin to even indices in the array; 2i
        sines = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)

        pos_encoding = pos_encoding[np.newaxis, ...]

        self.sinusoid = tf.cast(pos_encoding, dtype=tf.float32)

    @staticmethod
    def __get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs, **kwargs):
        return tf.add(inputs, self.sinusoid)


class DynamicPositionEmbedding(keras.layers.Layer):
    def __init__(self, embedding_dim, max_seq=2048, **kwargs):
        super().__init__(**kwargs)
        embed_sinusoid_list = np.array([[
            [
                m.sin(
                    pos * m.exp(-m.log(10000) * i/embedding_dim) *
                    m.exp(m.log(10000)/embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]])
        self.positional_embedding = tf.constant(embed_sinusoid_list, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return tf.add(inputs, self.positional_embedding[:,:inputs.shape[1],:])


class BaselineAttention(keras.layers.Layer):
    def __init__(self, h, d, max_seq=2048, **kwargs):
        super().__init__(**kwargs)
        self.len_k = None
        self.max_seq = None
        self.E = None
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = keras.layers.Dense(int(self.d / 2))
        self.Wk = keras.layers.Dense(int(self.d / 2))
        self.Wv = keras.layers.Dense(int(self.d))
        self.fc = keras.layers.Dense(d)
        self.max_seq = max_seq

    def build(self, input_shape):
        self.len_k = input_shape[1][1]
        # self.max_seq = max(input_shape[0][1], input_shape[1][1], input_shape[2][1])

    def call(self, inputs, mask=None, weight_out=False, **kwargs):
        """
        :param inputs: a list of tensors. i.e) [Q, K, V]
        :param mask: mask tensor
        :param weight_out: decide to get weather weight or not
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

        Kt = tf.transpose(k, [0, 1, 3, 2])
        QKt = tf.matmul(q, Kt)
        logits = QKt
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            logits += (tf.cast(mask, tf.float32) * -1e9)

        attention_weights = tf.nn.softmax(logits, -1)
        attention = tf.matmul(attention_weights, v)

        out = tf.transpose(attention, (0, 2, 1, 3))
        out = tf.reshape(out, (out.shape[0], -1, self.d))

        out = self.fc(out)

        return out, attention_weights


class RelativeGlobalAttention(keras.layers.Layer):
    """
    from Music Transformer ( Huang et al, 2018 )
    [paper link](https://arxiv.org/pdf/1809.04281.pdf)
    """
    def __init__(self, h=4, d=256, add_emb=False, max_seq=2048, **kwargs):
        super().__init__(**kwargs)
        self.len_k = None
        self.max_seq = max_seq
        self.E = None
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = keras.layers.Dense(int(self.d))
        self.Wk = keras.layers.Dense(int(self.d))
        self.Wv = keras.layers.Dense(int(self.d))
        self.fc = keras.layers.Dense(d)
        self.additional = add_emb
        if self.additional:
            self.Radd = None

    def build(self, input_shape):
        self.shape_q = input_shape[0][1]
        self.shape_k = input_shape[1][1]
        # self.max_seq = max(input_shape[0][1], input_shape[1][1], input_shape[2][1])
        self.E = self.add_weight('emb', shape=[self.max_seq, int(self.dh)])

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

        self.len_k = k.shape[2]
        self.len_q = q.shape[2]

        E = self._get_left_embedding(self.len_q, self.len_k)
        QE = tf.einsum('bhld,md->bhlm', q, E)
        QE = self._qe_masking(QE)
        # print(QE.shape)
        Srel = self._skewing(QE)

        Kt = tf.transpose(k,[0, 1, 3, 2])
        QKt = tf.matmul(q, Kt)
        logits = QKt + Srel
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            logits += (tf.cast(mask, tf.float32) * -1e9)

        attention_weights = tf.nn.softmax(logits, -1)
        # tf.print('logit result: \n', logits, output_stream=sys.stdout)

        attention = tf.matmul(attention_weights, v)
        # tf.print('attention result: \n', attention, output_stream=sys.stdout)

        out = tf.transpose(attention, (0, 2, 1, 3))
        out = tf.reshape(out, (out.shape[0], -1, self.d))

        out = self.fc(out)
        return out, attention_weights

    def _get_left_embedding(self, len_q, len_k):
        starting_point = max(0,self.max_seq-len_q)
        e = self.E[starting_point:,:]
        return e

    @staticmethod
    def _qe_masking(qe):
        mask = tf.sequence_mask(
            tf.range(qe.shape[-1] -1, qe.shape[-1] - qe.shape[-2] -1, -1), qe.shape[-1])

        mask = tf.logical_not(mask)
        mask = tf.cast(mask, tf.float32)

        return mask * qe

    def _skewing(self, tensor: tf.Tensor):
        padded = tf.pad(tensor, [[0, 0], [0,0], [0, 0], [1, 0]])
        reshaped = tf.reshape(padded, shape=[-1, padded.shape[1], padded.shape[-1], padded.shape[-2]])
        Srel = reshaped[:, :, 1:, :]
        # print('Sre: {}'.format(Srel))

        if self.len_k > self.len_q:
            Srel = tf.pad(Srel, [[0,0], [0,0], [0,0], [0, self.len_k-self.len_q]])
        elif self.len_k < self.len_q:
            Srel = Srel[:,:,:,:self.len_k]

        return Srel


class View1D(keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return inputs[:,self.axis]


class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.rga = RelativeGlobalAttention(h=h, d=d_model, max_seq=max_seq, add_emb=additional)

        self.FFN_pre = keras.layers.Dense(self.d_model // 2, activation=tf.nn.relu)
        self.FFN_suf = keras.layers.Dense(self.d_model)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, x, mask=None, training=False, **kwargs):
        attn_out, w = self.rga([x,x,x], mask)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(attn_out+x)

        ffn_out = self.FFN_pre(out1)
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = self.layernorm2(out1+ffn_out)
        return out2, w


class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.rga2 = RelativeGlobalAttention(d=d_model, h=h, max_seq=max_seq, add_emb=additional)
        self.rga = RelativeGlobalAttention(d=d_model, h=h, max_seq=max_seq, add_emb=additional)

        self.FFN_pre = keras.layers.Dense(self.d_model // 2, activation=tf.nn.relu)
        self.FFN_suf = keras.layers.Dense(self.d_model)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)

    def call(self, x, encode_out, mask=None, lookup_mask=None, training=False, w_out=False, **kwargs):

        attn_out, aw1 = self.rga([x, x, x], mask=lookup_mask)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(attn_out+x)

        if encode_out is None:
            attn_out2, aw2 = self.rga2([out1, out1, out1], mask=mask)
        else:
            attn_out2, aw2 = self.rga2([out1, encode_out, encode_out], mask=mask)
        attn_out2 = self.dropout2(attn_out2, training=training)
        attn_out2 = self.layernorm2(out1+attn_out2)

        ffn_out = self.FFN_pre(attn_out2)
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout3(ffn_out, training=training)
        out = self.layernorm3(attn_out2+ffn_out)

        if w_out:
            return out, aw1, aw2
        else:
            return out


class Encoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, input_vocab_size, rate=0.1, max_len=None):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = keras.layers.Embedding(input_vocab_size, d_model)
        # if max_len is not None:
        #     self.pos_encoding = PositionEmbedding(max_seq=max_len, embedding_dim=self.d_model)
        if True:
            self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)

        self.enc_layers = [EncoderLayer(d_model, rate, h=self.d_model // 64, additional=False, max_seq=max_len)
                           for i in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, mask=None, training=False):
        weights = []
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, w = self.enc_layers[i](x, mask, training=training)
            weights.append(w)
        return x, weights  # (batch_size, input_seq_len, d_model)


class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, input_vocab_size,
                 rate=0.1, max_len=None):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(input_vocab_size, d_model)

        if True:
            self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)

        self.dec_layers = [DecoderLayer(d_model, rate, h=self.d_model // 64, additional=False, max_seq=max_len)
                           for i in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, mask, lookup_mask, training, enc_output=None):
        weights = []
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, w1, w2 = \
                self.dec_layers[i](x, enc_output, lookup_mask=lookup_mask, mask=mask, training=training, w_out=True)
            weights.append((w1, w2))

        return x, weights  # (batch_size, input_seq_len, d_model)


if __name__ == '__main__':
    rga = RelativeGlobalAttention(d=9, h=1)
    q = tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32)
    k = tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32)

    import utils

    src_mask, trg_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(q.shape[1], tf.argmax(k,-1), tf.argmax(q, -1))
    # print(src_mask.shape, trg_mask.shape, look_ahead_mask.shape)

    result = rga([
        tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32),
        tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32),
        tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32),
        ], mask=trg_mask)

    print(result)

    k = tf.constant([
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
        dtype=tf.float32)
    q = tf.constant([
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
        dtype=tf.float32)
    import utils

    src_mask, trg_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(q.shape[1], tf.argmax(k, -1),
                                                                           tf.argmax(q, -1))
    print(src_mask.shape, trg_mask.shape, look_ahead_mask.shape)
    result = rga([
        tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32),
        tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32),
        tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32),
    ], mask=trg_mask)

    print(result)

    k = tf.constant([
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
        dtype=tf.float32)
    q = tf.constant([
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
        dtype=tf.float32)
    import utils

    src_mask, trg_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(q.shape[1], tf.argmax(k, -1),
                                                                           tf.argmax(q, -1))
    print(src_mask, trg_mask, look_ahead_mask)
    result = rga([
        q,
        k,
        k
    ], mask=look_ahead_mask)

    print(result)

