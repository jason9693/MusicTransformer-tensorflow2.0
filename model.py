from custom.layers import *
from custom.callback import *
import numpy as np
import params as par
import utils
import sys
from tensorflow.python import keras, enable_eager_execution
tf.executing_eagerly()
enable_eager_execution()


class MusicTransformer(keras.Model):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.1, debug=False, loader_path=None):
        super(MusicTransformer, self).__init__()
        self._debug = debug
        self.max_seq = max_seq
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # if loader_path:
        #     self.__load(loader_path)

        self.Encoder = Encoder(
            d_model=self.embedding_dim, input_vocab_size=self.vocab_size,
            num_layers=self.num_layer, rate=dropout, max_len=max_seq)
        self.Decoder = Decoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        self.drop = keras.layers.Dropout(dropout)
        self.fc = keras.layers.Dense(vocab_size, activation=tf.nn.softmax, name='output')

        self._set_metrics()

    def _set_metrics(self):
        accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.custom_metrics = [accuracy]

    # TODO : loader 작성
    def __load(self, dir_path):
        pass

    def __load_config_from_json(self, config):
        self._debug = config['debug']
        self.max_seq = config['max_seq']
        self.num_layer = config['num_layer']
        self.embedding_dim = config['embedding_dim']
        self.vocab_size = config['vocab_size']

    @staticmethod
    def __prepare_data(x ,y):
        start_token = tf.ones((y.shape[0], 1), dtype=y.dtype) * par.token_sos
        # end_token = tf.ones((y.shape[0], 1), dtype=y.dtype) * par.token_eos

        # # method with eos
        # out_tar = tf.concat([y[:, :-1], end_token], -1)
        # inp_tar = tf.concat([start_token, y[:, :-1]], -1)
        # x = tf.concat([start_token, x[:, 2:], end_token], -1)

        # method without eos
        out_tar = y
        inp_tar = y[:, :-1]
        inp_tar = tf.concat([start_token, inp_tar], -1)
        return x, inp_tar, out_tar

    def reset_metrics(self):
        for metric in self.custom_metrics:
            metric.reset_states()
        return

    def call(self, inputs, targets, training=None, src_mask=None, trg_mask=None, lookup_mask=None):
        encoder = self.Encoder(inputs, training=training, mask=src_mask)
        decoder = self.Decoder(targets, enc_output=encoder, training=training, lookup_mask=lookup_mask, mask=trg_mask)
        # out = self.drop(decoder)
        fc = self.fc(decoder)

        # if self._debug:
        #     tf.print('before fc: \n', decoder, output_stream=sys.stdout)
        #     tf.print('after fc: \n', fc, output_stream=sys.stdout)
        # if self._debug:
        #     tf.print(fc, output_stream=sys.stdout)
        return fc

    def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None, reset_metrics=True):

        if self._debug:
            tf.print('sanity:\n',self.sanity_check(x, y, mode='d'), output_stream=sys.stdout)

        x, dec_input, target = MusicTransformer.__prepare_data(x, y)

        enc_mask, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, dec_input)

        predictions = self.__train_step(x, dec_input, target, enc_mask, tar_mask, look_ahead_mask, True)
        if self._debug:
            print('train step finished')
        result_metric = []
        loss = tf.reduce_mean(self.loss(target, predictions))
        for metric in self.custom_metrics:
            result_metric.append(metric(target, predictions).numpy())

        return [loss.numpy()]+result_metric

    @tf.function
    def __train_step(self, inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, training):
        # if self._debug:
        #     tf.print('train step...', output_stream=sys.stdout)
        with tf.GradientTape() as tape:
            predictions = self.call(
                inp,
                targets=inp_tar,
                src_mask=enc_mask,
                trg_mask=tar_mask, lookup_mask=lookup_mask, training=training)
            loss = self.loss(out_tar, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.grad = gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return predictions

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None,
               max_queue_size=10,
               workers=1,
               use_multiprocessing=False):
        x, inp_tar, out_tar = MusicTransformer.__prepare_data(x, y)

        enc_mask, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, inp_tar)
        predictions = self.call(
                x,
                targets=inp_tar,
                src_mask=enc_mask,
                trg_mask=tar_mask, lookup_mask=look_ahead_mask, training=False)
        loss = tf.reduce_mean(self.loss(out_tar, predictions))
        result_metric = []
        for metric in self.custom_metrics:
            result_metric.append(metric(out_tar, predictions).numpy())
        return [loss.numpy()] + result_metric

    def sanity_check(self, x, y, mode='v'):
        x, inp_tar, out_tar = MusicTransformer.__prepare_data(x, y)

        enc_mask, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, inp_tar)
        predictions = self.call(
            x,
            targets=inp_tar,
            src_mask=enc_mask,
            trg_mask=tar_mask, lookup_mask=look_ahead_mask, training=False)

        if mode == 'v':
            return predictions
        elif mode == 'd':
            dic = {}
            for row in tf.argmax(predictions, -1).numpy():
                for col in row:
                    try:
                        dic[str(col)] += 1
                    except KeyError:
                        dic[str(col)] = 1
            return dic
        else:
            return tf.argmax(predictions, -1)

    def get_config(self):
        config = {}
        config['debug'] = self._debug
        config['max_seq'] = self.max_seq
        config['num_layer'] = self.num_layer
        config['embedding_dim'] = self.embedding_dim
        config['vocab_size'] = self.vocab_size
        return config

    def generate(self, prior: list, mode=None, length = 2048):
        prior = tf.constant([prior])

        decode_array = [par.token_sos]
        decode_array = tf.constant([decode_array])
        print(decode_array)

        # TODO: add beam search
        if mode == 'beam':
            pass
        else:
            for i in range(min(self.max_seq, length)):
                if i % 100 == 0:
                    print('generating... {}% completed'.format((i/min(self.max_seq, length))*100))
                # print(decode_array)
                enc_mask, tar_mask, look_ahead_mask = \
                    utils.get_masked_with_pad_tensor(decode_array.shape[1], prior, decode_array)

                result = self.call(prior,
                    targets=decode_array,
                    src_mask=enc_mask,
                    trg_mask=tar_mask, lookup_mask=look_ahead_mask, training=False)
                result = tf.argmax(result, -1)
                result = tf.cast(result, tf.int32)
                decode_array = tf.concat([decode_array, tf.expand_dims(result[:, -1], 0)], -1)
                del enc_mask
                del tar_mask
                del look_ahead_mask
        return decode_array.numpy()


if __name__ == '__main__':
    import utils
    import json
    from custom import callback
    print(tf.executing_eagerly())

    src = tf.constant([utils.fill_with_placeholder([1,2,3,4],max_len=2048)])
    trg = tf.constant([utils.fill_with_placeholder([1,2,3,4],max_len=2048)])
    src_mask, trg_mask, lookup_mask = utils.get_masked_with_pad_tensor(2048, src,trg)
    print(lookup_mask)
    # print(src_mask, trg_mask)
    mt = MusicTransformer(debug=True, embedding_dim=par.embedding_dim, vocab_size=par.vocab_size)
    mt.save_weights('my_model.h5', save_format='h5')
    mt.load_weights('my_model.h5')
    # print(mt.to_json())

    # print('compile...')
    # mt.compile(optimizer='adam', loss=callback.TransformerLoss(debug=True))
    # # print(mt.train_step(inp=src, tar=trg))
    #
    # print('start training...')
    # for i in range(2):
    #     mt.train_on_batch(x=src, y=trg)
    result = mt.generate([27, 186,  43, 213, 115, 131], length=100)
    print(result)
    import sequence
    sequence.EventSeq.from_array(result[0]).to_note_seq().to_midi_file('result.midi')
    pass