import pickle
import os
import re
import sys
import hashlib
from progress.bar import Bar
import tensorflow as tf
import utils
import params as par
from midi_processor.processor import encode_midi, decode_midi
from midi_processor import processor
import config
import random


def preprocess_midi(path):
    return encode_midi(path)
#     note_seq = NoteSeq.from_midi_file(path)
#     note_seq.adjust_time(-note_seq.notes[0].start)
#     event_seq = EventSeq.from_note_seq(note_seq)
#     control_seq = ControlSeq.from_event_seq(event_seq)
#     return event_seq.to_array(), control_seq.to_compressed_array()


def preprocess_midi_files_under(midi_root, save_dir):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'

    for path in Bar('Processing').iter(midi_paths):
        print(' ', end='[{}]'.format(path), flush=True)

        try:
            data = preprocess_midi(path)
        except KeyboardInterrupt:
            print(' Abort')
            return
        except EOFError:
            print('EOF Error')

        with open('{}/{}.pickle'.format(save_dir,path.split('/')[-1]), 'wb') as f:
            pickle.dump(data, f)


# def _augumentation(seq):
#     range_note = range(0, processor.RANGE_NOTE_ON+processor.RANGE_NOTE_OFF)
#     range_time = range(
#         processor.START_IDX['time_shift'],
#         processor.START_IDX['time_shift']+processor.RANGE_TIME_SHIFT
#     )
#     for idx, data in enumerate(seq):
#         if data in range_note:
#


class TFRecordsConverter(object):
    def __init__(self, midi_path, output_dir,
                 num_shards_train=3, num_shards_test=1):
        self.output_dir = output_dir
        self.num_shards_train = num_shards_train
        self.num_shards_test = num_shards_test

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Get lists of ev_seq and ctrl_seq
        self.es_seq_list, self.ctrl_seq_list = self.process_midi_from_dir(midi_path)

        # Counter for total number of images processed.
        self.counter = 0
    pass

    def process_midi_from_dir(self, midi_root):
        """
        :param midi_root: midi 데이터가 저장되어있는 디렉터리 위치.
        :return:
        """

        midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi', '.MID']))
        es_seq_list = []
        ctrl_seq_list = []
        for path in Bar('Processing').iter(midi_paths):
            print(' ', end='[{}]'.format(path), flush=True)

            try:
                data = preprocess_midi(path)
                for es_seq, ctrl_seq in data:
                    max_len = par.max_seq
                    for idx in range(max_len + 1):
                        es_seq_list.append(data[0])
                        ctrl_seq_list.append(data[1])

            except KeyboardInterrupt:
                print(' Abort')
                return
            except:
                print(' Error')
                continue

        return es_seq_list, ctrl_seq_list

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def __write_to_records(self, output_path, indicies):
        writer = tf.io.TFRecordWriter(output_path)
        for i in indicies:
            es_seq = self.es_seq_list[i]
            ctrl_seq = self.ctrl_seq_list[i]

        # example = tf.train.Example(features=tf.train.Features(feature={
        #         'label': TFRecordsConverter._int64_feature(label),
        #         'text': TFRecordsConverter._bytes_feature(bytes(x, encoding='utf-8'))}))


if __name__ == '__main__':
    preprocess_midi_files_under(
            midi_root=sys.argv[1],
            save_dir=sys.argv[2])

