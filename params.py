import midi_processor.processor as sequence
# max_seq = 2048
max_seq=2048
l_r = 0.001
embedding_dim = 256
num_attention_layer = 6
batch_size = 10
loss_type = 'categorical_crossentropy'
event_dim = sequence.RANGE_NOTE_ON + sequence.RANGE_NOTE_OFF + sequence.RANGE_TIME_SHIFT + sequence.RANGE_VEL
pad_token = event_dim
token_sos = event_dim + 1
token_eos = event_dim + 2
vocab_size = event_dim + 3
