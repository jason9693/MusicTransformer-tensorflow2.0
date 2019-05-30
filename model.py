from custom.layers import *
from custom.callback import *
import numpy as np
import params as par
import utils
from tensorflow.python import keras, enable_eager_execution
tf.executing_eagerly()
enable_eager_execution()

class MusicTransformer(keras.Model):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048,debug=False):
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
        self.fc = keras.layers.Dense(vocab_size, activation=tf.nn.softmax, name='output')
        self._set_metrics()

    def _set_metrics(self):
        accuracy = keras.metrics.SparseCategoricalAccuracy()
        f1 = None
        self.custom_metrics = [accuracy]

    def reset_metrics(self):
        for metric in self.custom_metrics:
            metric.reset_states()
        return

    def call(self, inputs, targets, training=None, src_mask=None, trg_mask=None, lookup_mask=None):
        encoder = self.Encoder(inputs, training=training, mask=src_mask)
        decoder = self.Decoder(targets, enc_output=encoder, training=training, lookup_mask=lookup_mask, mask=trg_mask)
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

        enc_mask, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, inp_tar)
        if self._debug:
            print('enc_mask: {}'.format(enc_mask))
            print('tar_mask: {}'.format(tar_mask))
        predictions = self.__train_step(x, inp_tar, out_tar, enc_mask, tar_mask, look_ahead_mask, True)
        result_metric = []
        loss = self.loss(out_tar, predictions)
        for metric in self.custom_metrics:
            result_metric.append(metric(out_tar, predictions).numpy())

        return [loss.numpy()]+result_metric

    @tf.function
    def __train_step(self, inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, training):
        with tf.GradientTape() as tape:
            predictions = self(
                inp,
                targets=inp_tar,
                src_mask=enc_mask,
                trg_mask=tar_mask, lookup_mask=lookup_mask, training=training)
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
    src_mask, trg_mask, lookup_mask = utils.get_masked_with_pad_tensor(2048, src,trg)
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