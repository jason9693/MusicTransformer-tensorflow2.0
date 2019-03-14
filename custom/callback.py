from tensorflow.python import keras
import tensorflow as tf

class MTFitCallback(keras.callbacks.Callback):

    def __init__(self, save_path):
        super(MTFitCallback, self).__init__()
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.save_path)



class TransformerLoss(keras.losses.CategoricalCrossentropy):
    def __init__(self, **kwargs):
        super(TransformerLoss, self).__init__(**kwargs)
        pass

    def call(self, y_true, y_pred):
        y_true = y_true[:,-1]
        y_pred = y_pred[:,-1]
        return super().call(y_true=y_true,y_pred=y_pred)


if __name__ == '__main__':
    import numpy as np
    loss = TransformerLoss()(np.array([[0,1],[1,0],[1,0]]), tf.constant([[0.5,0.5],[0.1,0.1],[0.1,0.1]]))
    print(loss)
    #keras.losses.CategoricalCrossentropy()(np.array([[[0,1],[1,0],[1,0]]]), tf.constant([[[0.5,0.5],[0.5,0.5],[0.5,0.5]]]))