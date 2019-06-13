import sequence
# max_seq = 2048
max_seq=50
l_r = 0.001
embedding_dim = 256
num_attention_layer = 6
batch_size = 10
loss_type = 'categorical_crossentropy'
event_dim = sequence.EventSeq.dim()
pad_token = event_dim
token_sos = event_dim + 1
token_eos = event_dim + 2
vocab_size = event_dim + 3
