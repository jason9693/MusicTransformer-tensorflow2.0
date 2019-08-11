from deprecated.sequence import EventSeq
import numpy as np

rand_array = np.random.random_sample([2048])
rand_array = rand_array * 240
rand_array = rand_array.astype(np.int)

print(rand_array)
es = EventSeq.from_array(rand_array)
es.to_note_seq().to_midi_file('out.midi')


