import os
import numpy as np
from deprecated.sequence import EventSeq, ControlSeq
import tensorflow as tf
import params as par


def find_files_by_extensions(root, exts=[]):
    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False
    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)


def event_indeces_to_midi_file(event_indeces, midi_file_name, velocity_scale=0.8):
    event_seq = EventSeq.from_array(event_indeces)
    note_seq = event_seq.to_note_seq()
    for note in note_seq.notes:
        note.velocity = int((note.velocity - 64) * velocity_scale + 64)
    note_seq.to_midi_file(midi_file_name)
    return len(note_seq.notes)


def transposition(events, controls, offset=0):
    # events [steps, batch_size, event_dim]
    # return events, controls

    events = np.array(events, dtype=np.int64)
    controls = np.array(controls, dtype=np.float32)
    event_feat_ranges = EventSeq.feat_ranges()

    on = event_feat_ranges['note_on']
    off = event_feat_ranges['note_off']

    if offset > 0:
        indeces0 = (((on.start <= events) & (events < on.stop - offset)) |
                    ((off.start <= events) & (events < off.stop - offset)))
        indeces1 = (((on.stop - offset  <= events) & (events < on.stop)) |
                    ((off.stop - offset <= events) & (events < off.stop)))
        events[indeces0] += offset
        events[indeces1] += offset - 12
    elif offset < 0:
        indeces0 = (((on.start - offset <= events) & (events < on.stop)) |
                    ((off.start - offset <= events) & (events < off.stop)))
        indeces1 = (((on.start <= events) & (events < on.start - offset)) |
                    ((off.start <= events) & (events < off.start - offset)))
        events[indeces0] += offset
        events[indeces1] += offset + 12

    assert ((0 <= events) & (events < EventSeq.dim())).all()
    histr = ControlSeq.feat_ranges()['pitch_histogram']
    controls[:, :, histr.start:histr.stop] = np.roll(
                    controls[:, :, histr.start:histr.stop], offset, -1)

    return events, controls


def dict2params(d, f=','):
    return f.join(f'{k}={v}' for k, v in d.items())


def params2dict(p, f=',', e='='):
    d = {}
    for item in p.split(f):
        item = item.split(e)
        if len(item) < 2:
            continue
        k, *v = item
        d[k] = eval('='.join(v))
    return d


def compute_gradient_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def get_masked_with_pad_tensor(size, src, trg):
    """
    :param size: the size of target input
    :param src: source tensor
    :param trg: target tensor
    :return:
    """
    src = tf.cast(src[:, tf.newaxis, tf.newaxis, :], tf.int32)
    trg = tf.cast(trg[:, tf.newaxis, tf.newaxis, :], tf.int32)
    src_pad_tensor = tf.ones_like(src) * par.pad_token
    src_mask = tf.cast(tf.equal(src, src_pad_tensor), dtype=tf.int32)
    trg_mask = tf.cast(tf.equal(src, src_pad_tensor), dtype=tf.int32)
    if trg is not None:
        trg_pad_tensor = tf.ones_like(trg) * par.pad_token
        dec_trg_mask = tf.cast(tf.equal(trg, trg_pad_tensor), dtype=tf.int32)
        # boolean reversing i.e) True * -1 + 1 = False
        seq_mask = tf.sequence_mask(list(range(1, size+1)), size, dtype=tf.int32) * -1 + 1
        look_ahead_mask = tf.cast(tf.maximum(dec_trg_mask, seq_mask), dtype=tf.int32)
    else:
        trg_mask = None
        look_ahead_mask = None

    return src_mask, trg_mask, look_ahead_mask


def get_mask_tensor(size):
    """
    :param size: max length of token
    :return:
    """
    # boolean reversing i.e) True * -1 + 1 = False
    seq_mask = tf.sequence_mask(range(1, size + 1), size, dtype=tf.int32) * -1 + 1
    return seq_mask


def fill_with_placeholder(prev_data: list, max_len: int, fill_val: float=par.pad_token):
    placeholder = [fill_val for _ in range(max_len - len(prev_data))]
    return prev_data + placeholder


def pad_with_length(max_length: int, seq: list, pad_val: float=par.pad_token):
    """
    :param max_length: max length of token
    :param seq: token list with shape:(length, dim)
    :param pad_val: padding value
    :return:
    """
    pad_length = max(max_length - len(seq), 0)
    pad = [pad_val] * pad_length
    return seq + pad


def append_token(data: tf.Tensor):
    start_token = tf.ones((data.shape[0], 1), dtype=data.dtype) * par.token_sos
    end_token = tf.ones((data.shape[0], 1), dtype=data.dtype) * par.token_eos

    return tf.concat([start_token, data, end_token], -1)


def weights2boards(weights, dir, step): # weights stored weight[layer][w1,w2]
    for weight in weights:
        w1, w2 = weight
        tf.summary.histogram()
    pass


def shape_list(x):
  """Shape list"""
  x_shape = tf.shape(x)
  x_get_shape = x.get_shape().as_list()

  res = []
  for i, d in enumerate(x_get_shape):
    if d is not None:
      res.append(d)
    else:
      res.append(x_shape[i])
  return res


def attention_image_summary(attn, step=0):
  """Compute color image summary.
  Args:
    attn: a Tensor with shape [batch, num_heads, query_length, memory_length]
    image_shapes: optional tuple of integer scalars.
      If the query positions and memory positions represent the
      pixels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, memory_rows, memory_cols).
      If the query positions and memory positions represent the
      pixels x channels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, query_channels,
         memory_rows, memory_cols, memory_channels).
  """
  num_heads = shape_list(attn)[1]
  # [batch, query_length, memory_length, num_heads]
  image = tf.transpose(attn, [0, 2, 3, 1])
  image = tf.math.pow(image, 0.2)  # for high-dynamic-range
  # Each head will correspond to one of RGB.
  # pad the heads to be a multiple of 3
  image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, tf.math.mod(-num_heads, 3)]])
  image = split_last_dimension(image, 3)
  image = tf.reduce_max(image, 4)
  tf.summary.image("attention", image, max_outputs=1, step=step)


def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.
  The first of these two dimensions is n.
  Args:
    x: a Tensor with shape [..., m]
    n: an integer.
  Returns:
    a Tensor with shape [..., n, m/n]
  """
  x_shape = shape_list(x)
  m = x_shape[-1]
  if isinstance(m, int) and isinstance(n, int):
    assert m % n == 0
  return tf.reshape(x, x_shape[:-1] + [n, m // n])

if __name__ == '__main__':

    s = np.array([np.array([1,2]*50),np.array([1,2,3,4]*25)])

    t = np.array([np.array([2,3,4,5,6]*20),np.array([1,2,3,4,5]*20)])
    print(t.shape)

    print(get_masked_with_pad_tensor(100,s,t))

