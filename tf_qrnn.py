import tensorflow as tf


class QRNNLayer:
    def __init__(self, input_size, conv_size, hidden_size, layer_id,
                 center_conv=False, feed_state=False, num_in_channels=1):
        self.input_size = input_size
        self.conv_size = conv_size
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        self.center_conv = center_conv
        self.feed_state = feed_state
        self.num_in_channels = num_in_channels
        if center_conv:
            assert conv_size % 2 == 1
        init = tf.random_normal_initializer()
        filter_shape = [conv_size,
                        input_size,
                        num_in_channels,
                        hidden_size*3]

        with tf.variable_scope('QRNN/conv/'+str(layer_id)):
            self.W = tf.get_variable('W', filter_shape,
                                     initializer=init, dtype=tf.float32)
            self.b = tf.get_variable('b', [hidden_size*3],
                                     initializer=init, dtype=tf.float32)
            if feed_state:
                self.W_v = tf.get_variable('W_v', [hidden_size, hidden_size*3])
                self.b_v = tf.get_variable('b_v', [hidden_size*3])

    def conv(self, inputs, state=None):
        assert (state is not None) == self.feed_state
        # input dims: [batch x seq x state x in]
        padded_inputs = self._pad_inputs(inputs, self.conv_size,
                                         self.center_conv)
        # padded_inputs dims: [batch x seq x state x in]
        conv = tf.nn.conv2d(padded_inputs, self.W, strides=[1, 1, 1, 1],
                            padding='VALID', name='conv'+str(self.layer_id))
        conv += self.b
        # conv dims: [batch x seq x in x num_conv*3]
        Z, F, O = tf.split(conv, 3, 3)
        # Z, F, O dims: [batch x seq x in x num_conv]

        if self.feed_state:
            linear_state = tf.nn.xw_plus_b(state, self.W_v, self.b_v)
            # linear_state dims: [batch x state*3]
            Z_v, F_v, O_v = tf.split(linear_state, 3, 1)
            Z += Z_v
            F += F_v
            O += O_v

        # apply nonlinearities, turn into lists by seq
        Z = tf.tanh(Z)
        F = tf.sigmoid(F)
        O = tf.sigmoid(O)

        return Z, F, O

    def _pad_inputs(self, inputs, conv_size, center_conv):
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


class DenseQRNNLayers:
    def __init__(self, input_size, conv_size, hidden_size, layer_ids,
                 center_conv=False, feed_state=False, num_layers=4,
                 pool_functions=None):
        self.layers = []
        self.num_layers = num_layers
        for i in range(num_layers):
            layer = QRNNLayer(input_size, conv_size, hidden_size, layer_ids[i],
                              center_conv, feed_state, num_in_channels=i+1)
            self.layers.append(layer)
        if pool_functions is None:
            pool_functions = [fo_pool] * num_layers-1 + [f_pool]
        self.pool_functions = pool_functions

    def compute(self, inputs, states=None):
        if states is None:
            states = [None] * self.num_layers
        for i in range(self.num_layers):
            Z, F, O = self.layers[i].conv(inputs, states[i])
            outputs = self.pool_functions[i](Z, F, O)
            inputs = tf.concat([inputs, outputs], 3)
        return tf.squeeze(outputs[:, :, :, -1])


def fo_pool(Z, F, O):
    Z = tf.unstack(Z, axis=1)
    F = tf.unstack(F, axis=1)
    O = tf.unstack(O, axis=1)
    C = [tf.fill(tf.shape(Z[0]), 0.0)]
    H = []
    for i in range(len(Z)):
        c = tf.multiply(F[i], C[-1]) + tf.multiply(1-F[i], Z[i])
        h = tf.multiply(O[i], c)
        C.append(c)
        H.append(h)
    H = tf.stack(H, axis=1)
    return tf.transpose(H, perm=[0, 1, 3, 2])


def f_pool(Z, F, O):
    Z = tf.unstack(Z, axis=1)
    F = tf.unstack(F, axis=1)
    O = tf.unstack(O, axis=1)
    H = [tf.fill(tf.shape(Z[0]), 0.0)]
    for i in range(len(Z)):
        h = tf.multiply(F[i], H[-1]) + tf.multiply(1-F[i], Z[i])
        H.append(h)
    H = tf.stack(H[1:], axis=1)
    return tf.transpose(H, perm=[0, 1, 3, 2])
