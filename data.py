import utils
import random
import pickle
from tensorflow.python import keras
import numpy as np

class Data:
    def __init__(self, dir_path):
        self.files = list(utils.find_files_by_extensions(dir_path, ['.pickle']))
        self._seq_file_name_idx = 0
        self._seq_idx = 0
        pass

    def __repr__(self):
        return '<class Data has "'+str(len(self.files))+'" files>'

    def batch(self, batch_size, length):
        batch_files = random.sample(self.files, k=batch_size)

        batch_data = [
            self._get_seq(file, length)
            for file in batch_files
        ]
        return batch_data

    def random_sequential_batch(self, batch_size, length):
        batch_files = random.sample(self.files, k=batch_size)
        batch_data = []
        for i in range(batch_size):
            data = self._get_seq(batch_files[i])
            for j in range(len(data) - length):
                batch_data.append(data[j:j+length])
                if len(batch_data) == batch_size:
                    return batch_data

    def sequential_batch(self, batch_size, length):
        batch_data = []
        data = self._get_seq( self.files[self._seq_file_name_idx] )
        # for j in range(len(data) - length):

        while len(batch_data) < batch_size:
            while self._seq_idx < len(data) - length:
                batch_data.append(data[self._seq_idx: self._seq_idx + length])
                self._seq_idx += 1
                if len(batch_data) == batch_size:
                    return batch_data

            self._seq_idx = 0
            self._seq_file_name_idx = self._seq_file_name_idx + 1
            if self._seq_file_name_idx == len(self.files):
                self._seq_file_name_idx = 0
                print('iter intialized')


    def _get_seq(self, fname, max_length=None):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        if max_length is not None:
            if max_length <= len(data):
                start = random.randrange(0,len(data) - max_length)
                data = data[start:start + max_length]


        return data

class PositionalY:
    def __init__(self, data, idx):
        self.data = data
        self.idx = idx

    def position(self):
        return self.idx

    def data(self):
        return self.data

    def __repr__(self):
        return '<Label located in {} position.>'.format(self.idx)

class DataSequence(keras.utils.Sequence):
    def __init__(self, path, batch_size, seq_len):
        self.data = Data(path)
        self.batch_size = batch_size
        self.file_idx = 0
        self.cache = []
        self.seq_len = seq_len
        pass


    def _update_cache(self):
        if self.file_idx < len(self.data.files)-1:
            self.file_idx += 1
        else:
            self.file_idx = 0
        seq = self.data._get_seq(self.data.files[self.file_idx])
        self.cache = self._cut_data(seq)

    # def _cut_data(self, seq):
    #     data = []
    #     size = self.seq_len + 1
    #     for i in range(len(seq)):
    #         data.append([fill_with_placeholder(seq[i : i+size - 1],self.seq_len), PositionalY(seq[i+size], i+size)])
    #
    #     return data


    def __len__(self):
        return len(self.data.files)

    def __getitem__(self, idx):
        data_batch = self.data.batch(self.batch_size,self.seq_len + 1)
        data_batch = np.array(data_batch)
        x = data_batch[:,:-1]
        y = data_batch[:,1:]

        return np.array(x), np.array(y)

def fill_with_placeholder(prev_data: list, max_len: int, max_val: float=239):
    placeholder = [max_val for _ in range(max_len - len(prev_data))]
    return prev_data + placeholder

if __name__ == '__main__':
    data = Data('dataset/processed')
    ds = DataSequence('dataset/processed', 10, 2048)
    print(ds.__getitem__(3)[1].shape)
    #print(data.batch(100, 2048))
    #while True:
    print(data.sequential_batch(10, 2048))