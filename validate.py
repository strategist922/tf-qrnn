import numpy as np
import tensorflow as tf


def convolution(input_, filter_width, out_fmaps, pool_type, zoneout_, Wz):
    """ Applies 1D convolution along time-dimension (T) assuming input
        tensor of dim (batch_size, T, n) and returns
        (batch_size, T, out_fmaps)
        zoneout: regularization (dropout) of F gate
    """
    in_shape = input_.get_shape()
    in_fmaps = in_shape[-1]
    num_gates = len(pool_type)
    gates = []
    # pad on the left to mask the convolution (make it causal)
    pinput = tf.pad(input_, [[0, 0], [filter_width - 1, 0], [0, 0]])
    with tf.variable_scope('convolutions'):
        # Wz = tf.get_variable('Wz', [filter_width, in_fmaps, out_fmaps],
        #                      initializer=tf.random_uniform_initializer(minval=-.05, maxval=.05))
        z_a = tf.nn.conv1d(pinput, Wz, stride=1, padding='VALID')
        # if self.bias_init_val is not None:
        #     bz = tf.get_variable('bz', [out_fmaps],
        #                          initializer=tf.constant_initializer(0.))
        #     z_a += bz

        z = tf.tanh(z_a)
        # compute gates convolutions
        # for gate_name in pool_type:
        #     Wg = tf.get_variable('W{}'.format(gate_name),
        #                          [filter_width, in_fmaps, out_fmaps],
        #                          initializer=tf.random_uniform_initializer(minval=-.05, maxval=.05))
        #     g_a = tf.nn.conv1d(pinput, Wg, stride=1, padding='VALID')
        #     if self.bias_init_val is not None:
        #         bg = tf.get_variable('b{}'.format(gate_name), [out_fmaps],
        #                              initializer=tf.constant_initializer(0.))
        #         g_a += bg
        #     g = tf.sigmoid(g_a)
        #     if not self.infer and zoneout_ > 0 and gate_name == 'f':
        #         print('Applying zoneout {} to gate F'.format(zoneout_))
        #         # appy zoneout to F
        #         g = zoneout((1. - g), 1. - zoneout_)
        #         # g = 1. - tf.nn.dropout((1. - g), 1. - zoneout)
        #     gates.append(g)
    return z, gates


def _pad_inputs(inputs, conv_size, center_conv):
    # new input dims: [batch x seq x state x in]
    if center_conv:
        num_pads = (conv_size-1) / 2
        padded_inputs = tf.pad(inputs,
                               [[0, 0], [num_pads, num_pads],
                                [0, 0], [0, 0]],
                               "CONSTANT")
    else:
        num_pads = conv_size - 1
        padded_inputs = tf.pad(inputs,
                               [[0, 0], [num_pads, 0],
                                [0, 0], [0, 0]],
                               "CONSTANT")
    # padded_inputs dims: [batch x seq x state x 1]
    return padded_inputs


def conv(inputs, W):
    # assert (state is not None) == self.feed_state
    # input dims: [batch x seq x state x in]
    padded_inputs = _pad_inputs(inputs, 2, False)
    # padded_inputs dims: [batch x seq x state x in]
    conv = tf.nn.conv2d(padded_inputs, W, strides=[1, 1, 1, 1],
                        padding='VALID', name='conv'+str(0))
    # conv += self.b
    # conv dims: [batch x seq x in x num_conv*3]
    Z, F, O = tf.split(conv, 3, 3)
    # Z, F, O dims: [batch x seq x in x num_conv]

    # if self.feed_state:
    #     linear_state = tf.nn.xw_plus_b(state, self.W_v, self.b_v)
    #     # linear_state dims: [batch x state*3]
    #     Z_v, F_v, O_v = tf.split(linear_state, 3, 1)
    #     Z += Z_v
    #     F += F_v
    #     O += O_v

    # apply nonlinearities, turn into lists by seq
    Z = tf.tanh(Z)
    F = tf.sigmoid(F)
    O = tf.sigmoid(O)

    return Z, F, O

batch_size = 5
seq_len = 4
state = 3

x2 = np.random.rand(batch_size, seq_len, state)
x1 = tf.constant(np.expand_dims(x2, 3))
x2 = tf.constant(x2)
sess = tf.Session()

W2 = np.random.rand(2, state, state)  # filter_width, in_fmaps, out_fmaps
W1 = np.zeros((2, state, 1, state*3))
W1[:, :, :, :state] = np.expand_dims(W2, 2)

W1 = tf.constant(W1)
W2 = tf.constant(W2)

Z1, _, _ = conv(x1, W1)
Z2, _ = convolution(x2, 2, state, 'fo', 0.0, W2)

print sess.run(tf.squeeze(Z1))
print
print sess.run(Z2)
