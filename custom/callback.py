from tensorflow.python import keras

class MTFitCallback(keras.callbacks.Callback):

    def __init__(self, save_path):
        super(MTFitCallback, self).__init__()
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.save_path)