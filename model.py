import tensorflow as tf
import math as m
from custom_layers import *
import numpy as np
from tensorflow.python import keras
tf.executing_eagerly()
class MusicTransformerV2:
    def __init__(self, embedding_dim = 256, vocab_size =240, num_layer =6,
                 max_seq = 2048,l_r = 0.001, debug = False):
        self._debug = debug
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_seq = max_seq

        # embed_sinusoid_list = [
        #     [
        #         m.sin(
        #             m.pow(
        #                 (pos * 0.00001), i / self.embedding_dim
        #             ) - m.pi * 0.5 * ((i + 1) % 2)
        #         )
        #         for i in range(self.embedding_dim)
        #     ]
        #     for pos in range(max_seq)
        # ]
        # self.embed_sinusoid_list = keras.backend.constant(embed_sinusoid_list)
        self.model = self._build_model()

        optim = keras.optimizers.Adam(l_r)
        self.model.compile(optim, loss='categorical_crossentropy')
        pass

    def _decoder(self, input_tensor, activation=None, layer=0):
        if self._debug:
            print('[DEBUG]:{}'.format('decoder called'))
        decoder1 = RelativeGlobalAttention(64)([input_tensor, input_tensor, input_tensor])# Assuming Dh = 64
        add_and_norm = keras.layers.Add()([decoder1, input_tensor])
        #add_and_norm = keras.layers.BatchNormalization()(add_and_norm)

        decoder2 = RelativeGlobalAttention(64)([add_and_norm, add_and_norm, add_and_norm])
        residual = keras.layers.Add()([decoder2, add_and_norm])
        #residual = keras.layers.BatchNormalization()(residual)

        FFN = keras.layers.Dense(self.embedding_dim, activation=tf.nn.relu)(residual)
        FFN = keras.layers.Dense(self.embedding_dim)(FFN)
        return FFN

    def _build_model(self):
        x = keras.Input([self.max_seq])
        embed = keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_seq)(x)
        embed = PositionEmbedding(self.max_seq, self.embedding_dim)(embed)

        decoder_input = embed

        for i in range(self.num_layer):
            decoder = self._decoder(
                decoder_input,
                tf.nn.relu,
                layer=i
            )
            decoder_input = decoder

        #flatten = keras.layers.Flatten()(decoder_input)
        crop = View1D(-1)(decoder_input)
        fc = keras.layers.Dense(self.vocab_size, activation=tf.nn.softmax)(crop)

        model = keras.Model(x, fc)
        return model

    def train(self, x, y, mode_params = {'batch':10, 'epoch':100}):
        if mode_params is not None:
            return self.model.fit(x, y, batch_size=mode_params['batch'], epochs=mode_params['epoch'])
        else:
            return self.model.train_on_batch(x,y)

    def processed_y(self, y: np.array):
        return np.eye(self.vocab_size)[y]




class MusicTransformer(keras.Model):
    def __init__(self, embedding_dim = 256, vocab_size =240, num_layer =6,
                 max_seq = 2048, debug = False):
        super(MusicTransformer, self).__init__()
        self._debug = debug
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
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
        embed_sinusoid_list = np.array(embed_sinusoid_list)
        self.positional_embedding = tf.constant(embed_sinusoid_list, dtype=tf.float32)
        self.embed = keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=max_seq)
        self.fc_layer = keras.layers.Dense(self.vocab_size, activation=tf.nn.softmax)
        self.decoder_list = [
            (
                RelativeGlobalAttention(64, name='rga_{}_1'.format(i)),
                keras.layers.Add(),
                RelativeGlobalAttention(64, name='rga_{}_2'.format(i)),
                keras.layers.Add(),
                keras.layers.Dense(self.embedding_dim, activation=tf.nn.leaky_relu),
                keras.layers.Dense(self.embedding_dim),
                keras.layers.BatchNormalization(),
                keras.layers.BatchNormalization()
            )
            for i in range(self.num_layer)
        ]
        self.flatten = keras.layers.Flatten()

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
        return np.eye(self.vocab_size)[y]

# class DecoderBlock(keras.layers.Wrapper):
#     def __init__(self, layer, **kwargs):
#         super().__init__(layer, **kwargs)
#         self.rga1 = RelativeGlobalAttention(64, name=self.name+'_1'),
#         self.add1 = tf.keras.layers.Add(),
#         self.rga2 = RelativeGlobalAttention(64, name=self.name+'_2'),
#         self.add2 = tf.keras.layers.Add(),
#         self.dense1 = tf.keras.layers.Dense(self.embedding_dim, activation=tf.nn.leaky_relu),
#         self.dense2 = tf.keras.layers.Dense(self.embedding_dim),
#         self.bn1 = tf.keras.layers.BatchNormalization(),
#         self.bn2 = tf.keras.layers.BatchNormalization()
#         pass








if __name__ == '__main__':
    mt = MusicTransformer(512)
    out = mt(tf.constant(shape=[10, 2048], value=0.0))
    print(out.shape)
    pass