from tensorflow.python import keras
import tensorflow as tf
import params as par


class MTFitCallback(keras.callbacks.Callback):

    def __init__(self, save_path):
        super(MTFitCallback, self).__init__()
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.save_path)


class TransformerLoss(keras.losses.SparseCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super(TransformerLoss, self).__init__(**kwargs)
        pass

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.math.logical_not(tf.math.equal(y_true, par.pad_token))
        mask = tf.cast(mask, tf.float32)
        loss_object = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        _loss = loss_object(y_true, y_pred)
        _loss *= mask
        return tf.reduce_mean(_loss)


if __name__ == '__main__':
    import numpy as np
    loss = TransformerLoss()(np.array([[0,1],[1,0],[1,0]]), tf.constant([[0.5,0.5],[0.1,0.1],[0.1,0.1]]))
    print(loss)
    #keras.losses.CategoricalCrossentropy()(np.array([[[0,1],[1,0],[1,0]]]), tf.constant([[[0.5,0.5],[0.5,0.5],[0.5,0.5]]]))