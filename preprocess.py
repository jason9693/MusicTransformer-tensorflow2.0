import os
import re
import sys
import hashlib
from progress.bar import Bar
import pickle

from sequence import NoteSeq, EventSeq, ControlSeq
import utils
import config

def preprocess_midi(path):
    note_seq = NoteSeq.from_midi_file(path)
    note_seq.adjust_time(-note_seq.notes[0].start)
    event_seq = EventSeq.from_note_seq(note_seq)
    control_seq = ControlSeq.from_event_seq(event_seq)
    return event_seq.to_array(), control_seq.to_compressed_array()

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
        except:
            print(' Error')
            continue

        with open('{}/{}.pickle'.format(save_dir,path.split('/')[-1]), 'wb') as f:
            pickle.dump(data[0], f)


        # name = os.path.basename(path)
        # code = hashlib.md5(path.encode()).hexdigest()
        # #save_path = os.path.join(save_dir, out_fmt.format(name, code))
        # #torch.save(data, save_path)

    print('Done')

if __name__ == '__main__':
    # preprocess_midi_files_under(
    #         midi_root=sys.argv[1],
    #         save_dir=sys.argv[2])
    preprocess_midi_files_under(
        midi_root='dataset/midi',
        save_dir='dataset/processed')