from custom.layers import *
from custom.callback import *
import numpy as np
import params as par
import utils
import sys
from tensorflow.python import keras
import re
import json
import data
tf.executing_eagerly()


class MusicTransformer(keras.Model):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, debug=False, loader_path=None, dist=False):
        super(MusicTransformer, self).__init__()
        self._debug = debug
        self.max_seq = max_seq
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dist = dist

        if loader_path is not None:
            self.load_config_file(loader_path)

        self.Encoder = Encoder(
            d_model=self.embedding_dim, input_vocab_size=self.vocab_size,
            num_layers=self.num_layer, rate=dropout, max_len=max_seq)
        self.Decoder = Decoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        self.fc = keras.layers.Dense(vocab_size, activation=None, name='output')

        self._set_metrics()

        if loader_path is not None:
            self.load_ckpt_file(loader_path)

    def call(self, inputs, targets, training=None, eval=None, src_mask=None, trg_mask=None, lookup_mask=None):
        encoder = self.Encoder(inputs, training=training, mask=src_mask)
        decoder = self.Decoder(targets, enc_output=encoder, training=training, lookup_mask=lookup_mask, mask=trg_mask)
        fc = self.fc(decoder)
        if training or eval:
            return fc
        else:
            return tf.nn.softmax(fc)

    def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None, reset_metrics=True):
        if self._debug:
            tf.print('sanity:\n',self.sanity_check(x, y, mode='d'), output_stream=sys.stdout)

        x, dec_input, target = MusicTransformer.__prepare_train_data(x, y)

        enc_mask, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, dec_input)

        if self.dist:
            predictions = self.__dist_train_step(
                x, dec_input, target, enc_mask, tar_mask, look_ahead_mask, True)
        else:
            predictions = self.__train_step(x, dec_input, target, enc_mask, tar_mask, look_ahead_mask, True)

        if self._debug:
            print('train step finished')
        result_metric = []

        if self.dist:
            loss = self._distribution_strategy.reduce(tf.distribute.ReduceOp.MEAN, self.loss_value, None)
        else:
            loss = tf.reduce_mean(self.loss_value)
        loss = tf.reduce_mean(loss)
        for metric in self.custom_metrics:
            result_metric.append(metric(target, predictions).numpy())

        return [loss.numpy()]+result_metric

    # @tf.function
    def __dist_train_step(self, inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, training):
        return self._distribution_strategy.experimental_run_v2(
            self.__train_step, args=(inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, training))

    # @tf.function
    def __train_step(self, inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, training):
        with tf.GradientTape() as tape:
            predictions = self.call(
                inp, targets=inp_tar, src_mask=enc_mask, trg_mask=tar_mask, lookup_mask=lookup_mask, training=training
            )
            self.loss_value = self.loss(out_tar, predictions)
        gradients = tape.gradient(self.loss_value, self.trainable_variables)
        self.grad = gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return predictions

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None,
                 max_queue_size=10, workers=1, use_multiprocessing=False):

        x, inp_tar, out_tar = MusicTransformer.__prepare_train_data(x, y)
        enc_mask, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, inp_tar)
        predictions = self.call(
                x,
                targets=inp_tar,
                src_mask=enc_mask,
                trg_mask=tar_mask, lookup_mask=look_ahead_mask, training=False, eval=True)
        loss = tf.reduce_mean(self.loss(out_tar, predictions))
        result_metric = []
        for metric in self.custom_metrics:
            result_metric.append(metric(out_tar, tf.nn.softmax(predictions)).numpy())
        return [loss.numpy()] + result_metric

    def save(self, filepath, overwrite=True, include_optimizer=False, save_format=None):
        config_path = filepath+'/'+'config.json'
        ckpt_path = filepath+'/ckpt'

        self.save_weights(ckpt_path, save_format='tf')
        with open(config_path, 'w') as f:
            json.dump(self.get_config(), f)
        return

    def load_config_file(self, filepath):
        config_path = filepath + '/' + 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.__load_config(config)

    def load_ckpt_file(self, filepath, ckpt_name='ckpt'):
        ckpt_path = filepath + '/' + ckpt_name
        try:
            self.load_weights(ckpt_path)
        except FileNotFoundError:
            print("[Warning] model will be initialized...")

    def sanity_check(self, x, y, mode='v'):
        # mode: v -> vector, d -> dict
        x, inp_tar, out_tar = MusicTransformer.__prepare_train_data(x, y)

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
        config['dist'] = self.dist
        return config

    def generate(self, prior: list, beam=None, length=2048):
        prior = tf.constant([prior])

        decode_array = [par.token_sos]
        # TODO: add beam search
        if beam is not None:
            k = beam
            decode_array = tf.constant([decode_array])

            for i in range(min(self.max_seq, length)):
                if i % 100 == 0:
                    print('generating... {}% completed'.format((i/min(self.max_seq, length))*100))
                enc_mask, tar_mask, look_ahead_mask = \
                    utils.get_masked_with_pad_tensor(decode_array.shape[1], prior, decode_array)

                result = self.call(prior, targets=decode_array, src_mask=enc_mask,
                                    trg_mask=tar_mask, lookup_mask=look_ahead_mask, training=False)
                result = result[:,-1,:]
                result = tf.reshape(result, (1, -1))
                result, result_idx = tf.nn.top_k(result, k)
                row = result_idx // par.vocab_size
                col = result_idx % par.vocab_size

                result_array = []
                for r, c in zip(row[0], col[0]):
                    prev_array = decode_array[r.numpy()]
                    result_unit = tf.concat([prev_array, [c.numpy()]], -1)
                    result_array.append(result_unit.numpy())
                    # result_array.append(tf.concat([decode_array[idx], result[:,idx_idx]], -1))
                decode_array = tf.constant(result_array)
                del enc_mask
                del tar_mask
                del look_ahead_mask
            decode_array = decode_array[0]
        else:
            decode_array = tf.constant([decode_array])
            for i in range(min(self.max_seq, length)):
                if i % 100 == 0:
                    print('generating... {}% completed'.format((i/min(self.max_seq, length))*100))
                enc_mask, tar_mask, look_ahead_mask = \
                    utils.get_masked_with_pad_tensor(decode_array.shape[1], prior, decode_array)

                result = self.call(prior, targets=decode_array, src_mask=enc_mask,
                                    trg_mask=tar_mask, lookup_mask=look_ahead_mask, training=False)
                result = tf.argmax(result, -1)
                result = tf.cast(result, tf.int32)
                decode_array = tf.concat([decode_array, tf.expand_dims(result[:, -1], 0)], -1)
                del enc_mask
                del tar_mask
                del look_ahead_mask
        return decode_array.numpy()

    def _set_metrics(self):
        accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.custom_metrics = [accuracy]

    def __load_config(self, config):
        self._debug = config['debug']
        self.max_seq = config['max_seq']
        self.num_layer = config['num_layer']
        self.embedding_dim = config['embedding_dim']
        self.vocab_size = config['vocab_size']
        self.dist = config['dist']

    @staticmethod
    def __prepare_train_data(x, y):
        start_token = tf.ones((y.shape[0], 1), dtype=y.dtype) * par.token_sos
        # end_token = tf.ones((y.shape[0], 1), dtype=y.dtype) * par.token_eos

        # # method with eos
        # out_tar = tf.concat([y[:, :-1], end_token], -1)
        # inp_tar = tf.concat([start_token, y[:, :-1]], -1)
        # x = tf.concat([start_token, x[:, 2:], end_token], -1)

        # method without eos
        out_tar = y
        inp_tar = y[:, :-1]
        # inp_tar = data.add_noise(inp_tar, rate=0)
        inp_tar = tf.concat([start_token, inp_tar], -1)
        return x, inp_tar, out_tar

    def reset_metrics(self):
        for metric in self.custom_metrics:
            metric.reset_states()
        return


class MusicTransformerDecoder(keras.Model):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, debug=False, loader_path=None, dist=False):
        super(MusicTransformerDecoder, self).__init__()
        self._debug = debug
        self.max_seq = max_seq
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dist = dist

        if loader_path is not None:
            self.load_config_file(loader_path)

        self.Encoder = Encoder(
            d_model=self.embedding_dim, input_vocab_size=self.vocab_size,
            num_layers=self.num_layer, rate=dropout, max_len=max_seq)
        self.Decoder = Decoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        self.fc = keras.layers.Dense(vocab_size, activation=None, name='output')

        self._set_metrics()

        if loader_path is not None:
            self.load_ckpt_file(loader_path)

    def call(self, inputs, targets, training=None, eval=None, src_mask=None, trg_mask=None, lookup_mask=None):
        encoder = self.Encoder(inputs, training=training, mask=src_mask)
        decoder = self.Decoder(targets, enc_output=encoder, training=training, lookup_mask=lookup_mask, mask=trg_mask)
        fc = self.fc(decoder)
        if training or eval:
            return fc
        else:
            return tf.nn.softmax(fc)

    def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None, reset_metrics=True):
        if self._debug:
            tf.print('sanity:\n',self.sanity_check(x, y, mode='d'), output_stream=sys.stdout)

        x, dec_input, target = MusicTransformer.__prepare_train_data(x, y)

        enc_mask, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, dec_input)

        if self.dist:
            predictions = self.__dist_train_step(
                x, dec_input, target, enc_mask, tar_mask, look_ahead_mask, True)
        else:
            predictions = self.__train_step(x, dec_input, target, enc_mask, tar_mask, look_ahead_mask, True)

        if self._debug:
            print('train step finished')
        result_metric = []

        if self.dist:
            loss = self._distribution_strategy.reduce(tf.distribute.ReduceOp.MEAN, self.loss_value, None)
        else:
            loss = tf.reduce_mean(self.loss_value)
        loss = tf.reduce_mean(loss)
        for metric in self.custom_metrics:
            result_metric.append(metric(target, predictions).numpy())

        return [loss.numpy()]+result_metric

    # @tf.function
    def __dist_train_step(self, inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, training):
        return self._distribution_strategy.experimental_run_v2(
            self.__train_step, args=(inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, training))

    # @tf.function
    def __train_step(self, inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, training):
        with tf.GradientTape() as tape:
            predictions = self.call(
                inp, targets=inp_tar, src_mask=enc_mask, trg_mask=tar_mask, lookup_mask=lookup_mask, training=training
            )
            self.loss_value = self.loss(out_tar, predictions)
        gradients = tape.gradient(self.loss_value, self.trainable_variables)
        self.grad = gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return predictions

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None,
                 max_queue_size=10, workers=1, use_multiprocessing=False):

        x, inp_tar, out_tar = MusicTransformer.__prepare_train_data(x, y)
        enc_mask, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, inp_tar)
        predictions = self.call(
                x,
                targets=inp_tar,
                src_mask=enc_mask,
                trg_mask=tar_mask, lookup_mask=look_ahead_mask, training=False, eval=True)
        loss = tf.reduce_mean(self.loss(out_tar, predictions))
        result_metric = []
        for metric in self.custom_metrics:
            result_metric.append(metric(out_tar, tf.nn.softmax(predictions)).numpy())
        return [loss.numpy()] + result_metric

    def save(self, filepath, overwrite=True, include_optimizer=False, save_format=None):
        config_path = filepath+'/'+'config.json'
        ckpt_path = filepath+'/ckpt'

        self.save_weights(ckpt_path, save_format='tf')
        with open(config_path, 'w') as f:
            json.dump(self.get_config(), f)
        return

    def load_config_file(self, filepath):
        config_path = filepath + '/' + 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.__load_config(config)

    def load_ckpt_file(self, filepath, ckpt_name='ckpt'):
        ckpt_path = filepath + '/' + ckpt_name
        try:
            self.load_weights(ckpt_path)
        except FileNotFoundError:
            print("[Warning] model will be initialized...")

    def sanity_check(self, x, y, mode='v'):
        # mode: v -> vector, d -> dict
        x, inp_tar, out_tar = MusicTransformer.__prepare_train_data(x, y)

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
        config['dist'] = self.dist
        return config

    def generate(self, prior: list, beam=None, length=2048):
        prior = tf.constant([prior])

        decode_array = [par.token_sos]
        # TODO: add beam search
        if beam is not None:
            k = beam
            decode_array = tf.constant([decode_array])

            for i in range(min(self.max_seq, length)):
                if i % 100 == 0:
                    print('generating... {}% completed'.format((i/min(self.max_seq, length))*100))
                enc_mask, tar_mask, look_ahead_mask = \
                    utils.get_masked_with_pad_tensor(decode_array.shape[1], prior, decode_array)

                result = self.call(prior, targets=decode_array, src_mask=enc_mask,
                                    trg_mask=tar_mask, lookup_mask=look_ahead_mask, training=False)
                result = result[:,-1,:]
                result = tf.reshape(result, (1, -1))
                result, result_idx = tf.nn.top_k(result, k)
                row = result_idx // par.vocab_size
                col = result_idx % par.vocab_size

                result_array = []
                for r, c in zip(row[0], col[0]):
                    prev_array = decode_array[r.numpy()]
                    result_unit = tf.concat([prev_array, [c.numpy()]], -1)
                    result_array.append(result_unit.numpy())
                    # result_array.append(tf.concat([decode_array[idx], result[:,idx_idx]], -1))
                decode_array = tf.constant(result_array)
                del enc_mask
                del tar_mask
                del look_ahead_mask
            decode_array = decode_array[0]
        else:
            decode_array = tf.constant([decode_array])
            for i in range(min(self.max_seq, length)):
                if i % 100 == 0:
                    print('generating... {}% completed'.format((i/min(self.max_seq, length))*100))
                enc_mask, tar_mask, look_ahead_mask = \
                    utils.get_masked_with_pad_tensor(decode_array.shape[1], prior, decode_array)

                result = self.call(prior, targets=decode_array, src_mask=enc_mask,
                                    trg_mask=tar_mask, lookup_mask=look_ahead_mask, training=False)
                result = tf.argmax(result, -1)
                result = tf.cast(result, tf.int32)
                decode_array = tf.concat([decode_array, tf.expand_dims(result[:, -1], 0)], -1)
                del enc_mask
                del tar_mask
                del look_ahead_mask
        return decode_array.numpy()

    def _set_metrics(self):
        accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.custom_metrics = [accuracy]

    def __load_config(self, config):
        self._debug = config['debug']
        self.max_seq = config['max_seq']
        self.num_layer = config['num_layer']
        self.embedding_dim = config['embedding_dim']
        self.vocab_size = config['vocab_size']
        self.dist = config['dist']

    @staticmethod
    def __prepare_train_data(x, y):
        start_token = tf.ones((y.shape[0], 1), dtype=y.dtype) * par.token_sos
        # end_token = tf.ones((y.shape[0], 1), dtype=y.dtype) * par.token_eos

        # # method with eos
        # out_tar = tf.concat([y[:, :-1], end_token], -1)
        # inp_tar = tf.concat([start_token, y[:, :-1]], -1)
        # x = tf.concat([start_token, x[:, 2:], end_token], -1)

        # method without eos
        out_tar = y
        inp_tar = y[:, :-1]
        # inp_tar = data.add_noise(inp_tar, rate=0)
        inp_tar = tf.concat([start_token, inp_tar], -1)
        return x, inp_tar, out_tar

    def reset_metrics(self):
        for metric in self.custom_metrics:
            metric.reset_states()
        return


if __name__ == '__main__':
    import utils
    print(tf.executing_eagerly())

    src = tf.constant([utils.fill_with_placeholder([1,2,3,4],max_len=2048)])
    trg = tf.constant([utils.fill_with_placeholder([1,2,3,4],max_len=2048)])
    src_mask, trg_mask, lookup_mask = utils.get_masked_with_pad_tensor(2048, src,trg)
    print(lookup_mask)
    print(src_mask)
    mt = MusicTransformer(debug=True, embedding_dim=par.embedding_dim, vocab_size=par.vocab_size)
    mt.save_weights('my_model.h5', save_format='h5')
    mt.load_weights('my_model.h5')
    result = mt.generate([27, 186,  43, 213, 115, 131], length=100)
    print(result)
    import sequence
    sequence.EventSeq.from_array(result[0]).to_note_seq().to_midi_file('result.midi')
    pass