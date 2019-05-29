import sequence
max_seq = 2048
l_r = 0.001
embedding_dim = 256
num_attention_layer = 6
batch_size = 10
loss_type = 'categorical_crossentropy'
vocab_size = sequence.EventSeq.dim()
pad_token = vocab_size
token_sos = vocab_size + 1
vocab_size = vocab_size + 2
