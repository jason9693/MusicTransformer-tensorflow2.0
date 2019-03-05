import utils
import random
import pickle

class Data:
    def __init__(self, dir_path):
        self.files = list(utils.find_files_by_extensions(dir_path, ['.pickle']))
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

    def _get_seq(self, fname, max_length=None):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        print(max(data))
        if max_length is not None:
            if max_length <= len(data):
                start = random.randrange(0,len(data) - max_length)
                data = data[start:start + max_length]


        return data



if __name__ == '__main__':
    data = Data('dataset/processed')
    print(data.batch(100, 2048))