from model import MusicTransformer
from custom.layers import *
from custom import callback
from tensorflow.python import keras
import params as par
from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from data import Data
import utils
enable_eager_execution()

tf.executing_eagerly()

if __name__ == '__main__':
    epoch = 100
    batch = 1000

    dataset = Data('dataset/processed/')
    opt = Adam(0.0001)
    # opt = SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
    mt = MusicTransformer(
        embedding_dim=par.embedding_dim, vocab_size=par.vocab_size, num_layer=6,
        max_seq=100, debug=True
    )
    mt.compile(optimizer=opt, loss=callback.TransformerLoss())

    for e in range(epoch):
        for b in range(batch):
            batch_x, batch_y = dataset.seq2seq_batch(2, 100)
            result_metrics = mt.train_on_batch(batch_x, batch_y)
            print('Loss: {:6.6}, Accuracy: {:3.2}'.format(result_metrics[0], result_metrics[1]))

