from tensorflow.python import keras
import tensorflow as tf
import params as par
import sys
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule


class MTFitCallback(keras.callbacks.Callback):

    def __init__(self, save_path):
        super(MTFitCallback, self).__init__()
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.save_path)


class TransformerLoss(keras.losses.SparseCategoricalCrossentropy):
    def __init__(self, from_logits=False, reduction='none', debug=False,  **kwargs):
        super(TransformerLoss, self).__init__(from_logits, reduction, **kwargs)
        self.debug = debug
        pass

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.math.logical_not(tf.math.equal(y_true, par.pad_token))
        mask = tf.cast(mask, tf.float32)
        _loss = super(TransformerLoss, self).call(y_true, y_pred)
        _loss *= mask
        if self.debug:
            tf.print('loss shape:', _loss.shape, output_stream=sys.stdout)
            tf.print('output:', tf.argmax(y_pred,-1), output_stream=sys.stdout)
            tf.print(mask, output_stream=sys.stdout)
            tf.print(_loss, output_stream=sys.stdout)
        return _loss


def transformer_dist_train_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.math.logical_not(tf.math.equal(y_true, par.pad_token))
    mask = tf.cast(mask, tf.float32)

    y_true_vector = tf.one_hot(y_true, par.vocab_size)

    _loss = tf.nn.softmax_cross_entropy_with_logits(y_true_vector, y_pred)
    # print(_loss.shape)
    #
    # _loss = tf.reduce_mean(_loss, -1)
    _loss *= mask

    return _loss


class CustomSchedule(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def get_config(self):
        super(CustomSchedule, self).get_config()

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


if __name__ == '__main__':
    import numpy as np
    loss = TransformerLoss()(np.array([[1],[0],[0]]), tf.constant([[0.5,0.5],[0.1,0.1],[0.1,0.1]]))
    print(loss)
