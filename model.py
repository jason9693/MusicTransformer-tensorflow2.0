from custom.layers import *
from custom.callback import *
import numpy as np
import params as par
import utils
from tensorflow.python import keras, enable_eager_execution
tf.executing_eagerly()
enable_eager_execution()


class MusicTransformerV3:
    def __init__(self, embedding_dim=256, vocab_size =240, num_layer =6,
                 max_seq=2048, l_r=0.001, debug = False, dropout = 0.1):
        self._debug = debug
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.dropout = dropout
        self.model = self._build_model()
        optim = keras.optimizers.Adam(lr=l_r, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        # loss_func = TransformerLoss()
        self.model.compile(optim, loss='categorical_crossentropy',  metrics = ['accuracy'])
        pass

    def _encoder_block(self, input_tensor,layer=0):
        if self._debug:
            print('[DEBUG]:{}'.format('encoder was called'))
        encoder1 = RelativeGlobalAttention(64)([input_tensor, input_tensor, input_tensor])
        encoder1 = keras.layers.Dropout(rate=self.dropout)(encoder1)

        residual = keras.layers.Add()([encoder1, input_tensor])
        residual = keras.layers.LayerNormalization()(residual)

        FFN = keras.layers.Conv1D(512, 1, activation=tf.nn.relu)(residual)
        FFN = keras.layers.Conv1D(self.embedding_dim, 1)(FFN)
        FFN = keras.layers.Dropout(rate=self.dropout)(FFN)

        return FFN

    def _decoder_block(self, input_tensor, encoder_input,layer=0):
        if self._debug:
            print('[DEBUG]:{}'.format('decoder was called'))
        decoder1 = RelativeGlobalAttention(64)([input_tensor, input_tensor, input_tensor])# Assuming Dh = 64
        decoder1 = keras.layers.Dropout(rate=self.dropout)(decoder1)

        add_and_norm = keras.layers.Add()([decoder1, input_tensor])
        add_and_norm = keras.layers.LayerNormalization()(add_and_norm)

        decoder2 = RelativeGlobalAttention(64)([encoder_input, encoder_input, add_and_norm])
        decoder2 = keras.layers.Dropout(rate=self.dropout)(decoder2)

        residual = keras.layers.Add()([decoder2, add_and_norm])
        residual = keras.layers.LayerNormalization()(residual)

        FFN = keras.layers.Conv1D(512 ,1, activation=tf.nn.relu)(residual)
        FFN = keras.layers.Conv1D(self.embedding_dim, 1)(FFN)
        FFN = keras.layers.Dropout(rate=self.dropout)(FFN)

        FFN = keras.layers.Add()([FFN, residual])
        FFN = keras.layers.LayerNormalization()(FFN)
        return FFN

    def _build_encoder(self, encoder_length=10):
        x = keras.layers.Input([encoder_length])

        embed = keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=encoder_length)(x)
        embed = DynamicPositionEmbedding(self.embedding_dim)(embed)

        encoder_input = embed

        for i in range(self.num_layer):
            encoder = self._encoder_block(
                encoder_input,
                layer=i
            )
            encoder_input = encoder

        model = keras.Model(x, encoder_input)
        return model

    def _build_decoder(self, encoder_input):
        x = keras.Input([self.max_seq])

        embed = keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_seq)(x)
        embed = PositionEmbedding(self.max_seq, self.embedding_dim)(embed)

        decoder_input = embed

        for i in range(self.num_layer):
            decoder = self._decoder_block(
                decoder_input,
                encoder_input,
                layer=i,
            )
            decoder_input = decoder

        finale = keras.layers.Dense(self.vocab_size, activation=tf.nn.softmax)
        fc = keras.layers.TimeDistributed(finale)(decoder_input)
        model = keras.Model(x, fc)

        return model

    def _build_model(self):
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder(self.encoder.output)

    def _loss(self, x):

        pass

    @tf.function
    def train(self, encoder_x, decoder_x):
        with tf.GradientTape as tape:
            encoder_x = self._fill_with_placeholder(encoder_x, self.max_seq)
            decoder_x = self._fill_with_placeholder(decoder_x, self.max_seq)
            pass
        pass

    def infer(self):
        pass

    def _fill_with_placeholder(self, prev_data, max_len: int, max_val: float = 239):
        placeholder = [max_val for _ in range(max_len - prev_data.shape[1])]
        return tf.concat([prev_data, [placeholder] * prev_data.shape[0]], axis=-1)

    def processed_y(self, y: np.array):
        return np.eye(self.vocab_size)[y]


class MusicTransformerV2:
    def __init__(self, embedding_dim = 256, vocab_size =240, num_layer =6,
                 max_seq = 2048,l_r = 0.001, debug = False, dropout = 0.1):
        self._debug = debug
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.dropout = dropout
        self.model = self._build_model()
        optim = keras.optimizers.Adam(lr=l_r, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        # loss_func = TransformerLoss()
        self.model.compile(optim, loss='categorical_crossentropy',  metrics = ['accuracy'])
        pass

    # def _encoder(self, input_tensor,layer=0):
    #     if self._debug:
    #         print('[DEBUG]:{}'.format('ecoder was called'))
    #
    #     encoder1 = RelativeGlobalAttention(64)([input_tensor, input_tensor, input_tensor])
    #     encoder1 = keras.layers.Dropout(rate=self.dropout)(encoder1)
    #
    #     residual = keras.layers.Add()([encoder1, input_tensor])
    #     residual = keras.layers.LayerNormalization()(residual)
    #
    #     FFN = keras.layers.Conv1D(512, 1, activation=tf.nn.relu)(residual)
    #     FFN = keras.layers.Conv1D(self.embedding_dim, 1)(FFN)
    #     FFN = keras.layers.Dropout(rate=self.dropout)(FFN)
    #
    #     return FFN

    def _decoder(self, input_tensor, layer=0):
        if self._debug:
            print('[DEBUG]:{}'.format('decoder was called'))

        decoder1 = RelativeGlobalAttention(64)([input_tensor, input_tensor, input_tensor])# Assuming Dh = 64
        decoder1 = keras.layers.Dropout(rate=self.dropout)(decoder1)

        add_and_norm = keras.layers.Add()([decoder1, input_tensor])
        add_and_norm = keras.layers.LayerNormalization()(add_and_norm)

        decoder2 = RelativeGlobalAttention(64)([input_tensor, input_tensor, add_and_norm])
        decoder2 = keras.layers.Dropout(rate=self.dropout)(decoder2)

        residual = keras.layers.Add()([decoder2, add_and_norm])
        residual = keras.layers.LayerNormalization()(residual)

        FFN = keras.layers.Conv1D(512 ,1, activation=tf.nn.relu)(residual)
        FFN = keras.layers.Conv1D(self.embedding_dim, 1)(FFN)
        FFN = keras.layers.Dropout(rate=self.dropout)(FFN)

        FFN = keras.layers.Add()([FFN, residual])
        FFN = keras.layers.LayerNormalization()(FFN)
        return FFN

    def _build_model(self):
        x = keras.Input([self.max_seq])

        embed = keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_seq)(x)
        embed = PositionEmbedding(self.max_seq, self.embedding_dim)(embed)

        decoder_input = embed

        for i in range(self.num_layer):
            decoder = self._decoder(
                decoder_input,
                layer=i
            )
            decoder_input = decoder

        finale = keras.layers.Dense(self.vocab_size, activation=tf.nn.softmax)
        fc = keras.layers.TimeDistributed(finale)(decoder_input)
        model = keras.Model(x, fc)

        return model

    def _fill_with_placeholder(self, prev_data, max_len: int, max_val: float = 239):
        placeholder = [max_val for _ in range(max_len - prev_data.shape[1])]
        return tf.concat([prev_data, [placeholder] * prev_data.shape[0]], axis=-1)

    def processed_y(self, y: np.array):
        return np.eye(self.vocab_size)[y]


class MusicTransformer(keras.Model):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, debug=False):
        super(MusicTransformer, self).__init__()
        self._debug = debug
        self.max_seq = max_seq
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.Encoder = Encoder(
            d_model=self.embedding_dim, input_vocab_size=self.vocab_size, num_layers=self.num_layer, max_len=max_seq)
        self.Decoder = Decoder(
            num_layers=self.num_layer, d_model=self.embedding_dim, input_vocab_size=self.vocab_size, max_len=max_seq)
        self.fc = keras.layers.Dense(vocab_size, activation=tf.nn.softmax)
        self._set_metrics()

    def _set_metrics(self):
        accuracy = keras.metrics.SparseCategoricalCrossentropy()
        f1 = None
        self.custom_metrics = [accuracy]

    def call(self, inputs, targets, training=None, src_mask=None, trg_mask=None):
        encoder = self.Encoder(inputs, training=training, mask=src_mask)
        decoder = self.Decoder(targets, enc_output=encoder, training=training, trg_mask=trg_mask, src_mask=src_mask)
        fc = self.fc(decoder)
        if self._debug:
            print(fc)
        return fc

    def train_on_batch(self,
                     x,
                     y=None,
                     sample_weight=None,
                     class_weight=None,
                     reset_metrics=True):
        start_token = tf.ones((y.shape[0], 1), dtype=y.dtype) * par.token_sos
        out_tar = y
        inp_tar = y[:, :-1]
        inp_tar = tf.concat([start_token, inp_tar], -1)

        enc_mask, tar_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, inp_tar)
        predictions = self.__train_step(x, inp_tar, out_tar, enc_mask, tar_mask, True)
        result_metric = []
        loss = self.loss(out_tar, predictions)
        for metric in self.custom_metrics:
            result_metric.append(metric(out_tar, predictions).numpy())

        return [loss.numpy()]+result_metric

    @tf.function
    def __train_step(self, inp, inp_tar, out_tar, enc_mask, tar_mask, training):
        with tf.GradientTape() as tape:
            predictions = self(
                inp,
                targets=inp_tar,
                src_mask=enc_mask,
                trg_mask=tar_mask, training=training)
            loss = self.loss(out_tar, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.grad = gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return predictions

    def predict(self,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False):
        x = utils.pad_with_length(self.max_seq, x, par.pad_token)
        tar_inp = [par.token_sos]
        for seq in range(self.max_seq):
            pass
        pass

    # # TODO: re-define loss function.
    # @staticmethod
    # def loss(real, pred):
    #     mask = tf.math.logical_not(tf.math.equal(real, par.pad_token))
    #     loss_object = keras.losses.SparseCategoricalCrossentropy(
    #         from_logits=True, reduction='none')
    #     _loss = loss_object(real, pred)
    #     _loss *= tf.cast(mask, tf.float32)
    #     return tf.reduce_mean(_loss)


if __name__ == '__main__':
    import utils
    from custom import callback
    print(tf.executing_eagerly())

    src = tf.constant([utils.fill_with_placeholder([1,2,3,4],max_len=2048)])
    trg = tf.constant([utils.fill_with_placeholder([1,2,3,4],max_len=2048)])
    src_mask, trg_mask = utils.get_masked_with_pad_tensor(2048, src,trg)
    # print(src_mask, trg_mask)
    mt = MusicTransformer(debug=True)
    mt.compile(optimizer='adam', loss=callback.TransformerLoss())
    # print(mt.train_step(inp=src, tar=trg))
    print(mt.train_on_batch(x=src, y=trg))
    print(mt.grad)
    # mt.fit(x=src, y=trg)
    # result = mt(
    #     src,
    #     targets=trg,
    #     trg_mask=trg_mask,
    #     src_mask=src_mask,
    # )
    print(mt.summary())
    #mt.model.fit(x=np.ones(shape=[10, 2049]), y=)
    pass