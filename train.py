from model import MusicTransformerV2, MusicTransformer
import tensorflow as tf
import numpy as np
import params as par
import sequence
import utils
import time
from tensorflow.python import keras
tf.executing_eagerly()

if __name__ == '__main__':
    # print(tf.TensorShape(None).dims)
    # model = MusicTransformer()
    # optimizer = keras.optimizers.Adam(lr=par.l_r)
    # model.compile(optimizer=optimizer, loss=par.loss_type)
    #
    # x= np.zeros(shape=[par.batch_size, par.max_seq])
    # y = np.ones(shape=[par.batch_size], dtype=np.int)
    #
    # # print(model.summary())
    # start_time = time.time()
    # model.train_on_batch(x=x,y=model.processed_y(y))
    # #model.build(input_shape=x.shape[1:])
    # end_time = time.time()
    #
    # print(model.summary())
    # print('update per time: {}'.format(end_time-start_time))
    mt = MusicTransformerV2()
    print(mt.model.summary())