import utils
import random
import pickle

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



if __name__ == '__main__':
    data = Data('dataset/process_test')
    #print(data.batch(100, 2048))
    #while True:
    print(data.sequential_batch(10, 2048))