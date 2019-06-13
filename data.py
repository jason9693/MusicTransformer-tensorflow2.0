import utils
import random
import pickle
from tensorflow.python import keras
import numpy as np
import params as par
import sequence


class Data:
    def __init__(self, dir_path):
        self.files = list(utils.find_files_by_extensions(dir_path, ['.pickle']))
        self.file_dict = {
            'train': self.files[:int(len(self.files) * 0.8)],
            'eval': self.files[int(len(self.files) * 0.8): int(len(self.files) * 0.9)],
            'test': self.files[int(len(self.files) * 0.9):],
        }
        self._seq_file_name_idx = 0
        self._seq_idx = 0
        pass

    def __repr__(self):
        return '<class Data has "'+str(len(self.files))+'" files>'

    def batch(self, batch_size, length, mode='train'):

        batch_files = random.sample(self.file_dict[mode], k=batch_size)

        batch_data = [
            self._get_seq(file, length)
            for file in batch_files
        ]
        return np.array(batch_data)  # batch_size, seq_len

    def seq2seq_batch(self, batch_size, length, mode='train'):
        data = self.batch(batch_size, length * 2, mode)
        x = data[:, :length]
        y = data[:, length:]
        return x, y

    def slide_seq2seq_batch(self, batch_size, length, mode='train'):
        data = self.batch(batch_size, length+1, mode)
        x = data[:, :-1]
        y = data[:, 1:]
        return x, y

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
        data = self._get_seq(self.files[self._seq_file_name_idx])

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
            else:
                data = np.append(data, par.token_eos)
                while len(data) < max_length:
                    data = np.append(data, par.pad_token)
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
    def __init__(self, path, batch_size, seq_len, vocab_size=sequence.EventSeq.dim()+2):
        self.data = Data(path)
        self.batch_size = batch_size
        self.file_idx = 0
        self.cache = []
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        pass

    def _update_cache(self):
        if self.file_idx < len(self.data.files)-1:
            self.file_idx += 1
        else:
            self.file_idx = 0
        seq = self.data._get_seq(self.data.files[self.file_idx])
        self.cache = self._cut_data(seq)

    def __len__(self):
        return len(self.data.files)

    def __getitem__(self, idx):
        data_batch = self.data.batch(self.batch_size,self.seq_len + 1)
        data_batch = np.array(data_batch)
        try:
            x = data_batch[:,:-1]
            y = data_batch[:,1:]
        except:
            print('except')
            return self.__getitem__(idx)

        return np.array(x), np.eye(self.vocab_size)[np.array(y)]


if __name__ == '__main__':
    import pprint
    def count_dict(max_length, data):
        cnt_arr = [0] * max_length
        cnt_dict = {}
        # print(cnt_arr)
        for batch in data:
            for index in batch:
                try:
                    cnt_arr[int(index)] += 1

                except:
                    print(index)
                try:
                    cnt_dict['index-'+str(index)] += 1
                except KeyError:
                    cnt_dict['index-'+str(index)] = 1
        return cnt_arr


    print(par.vocab_size)
    data = Data('dataset/processed')
    # ds = DataSequence('dataset/processed', 10, 2048)
    sample = data.seq2seq_batch(1000, 100)[0]
    pprint.pprint(list(sample))
    arr = count_dict(par.vocab_size+3,sample)
    pprint.pprint(
        arr)

    from sequence import EventSeq, Event

    event_cnt = {
        'note_on': 0,
        'note_off': 0,
        'velocity': 0,
        'time_shift': 0
    }
    for event_index in range(len(arr)):
        for event_type, feat_range in EventSeq.feat_ranges().items():

            if feat_range.start <= event_index < feat_range.stop:
                print(event_type+':'+str(arr[event_index])+' event cnt: '+str(event_cnt))
                event_cnt[event_type] += arr[event_index]

                # event_value = event_index - feat_range.start
                # events.append(Event(event_type, time, event_value))
                # if event_type == 'time_shift':
                #     time += EventSeq.time_shift_bins[event_value]
                # break
    print(event_cnt)

    # print(np.max(sample), np.min(sample))
    # print([data._get_seq(file).shape for file in data.files])
    #while True:
    # print(ds.__getitem__(10)[1].argmax(-1))