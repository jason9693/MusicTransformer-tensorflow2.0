from model import MusicTransformer
from custom.layers import *
from custom import callback
from tensorflow.python import keras
import params as par
from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import utils
enable_eager_execution()

tf.executing_eagerly()

if __name__ == '__main__':
    epoch = 100
    batch = 1000

    dataset = Data('dataset/processed/')
    opt = Adam(par.l_r)
    mt = MusicTransformer(
        embedding_dim=256, vocab_size=388 + 2, num_layer=6,
        max_seq=2048, debug=False
    )
    mt.compile(optimizer=opt, loss=callback.TransformerLoss())

    for e in range(epoch):
        for b in range(batch):
            batch_x, batch_y = dataset.seq2seq_batch(3, par.max_seq)
            result_metrics = mt.train_on_batch(batch_x, batch_y)
            print('Loss: {:6.6}, Accuracy: {:3.2}'.format(result_metrics[0], result_metrics[1]))

