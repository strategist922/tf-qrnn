import tensorflow as tf


class QRNNLayer:
    def __init__(self, input_size, conv_size, hidden_size, layer_id, pool='fo',
                 zoneout=0.0, center_conv=False, feed_state=False,
                 num_in_channels=1):

        pooling_functions = {
            'f': self.f_pool,
            'fo': self.fo_pool,
            'ifo': self.ifo_pool
        }

        self.input_size = input_size
        self.conv_size = conv_size
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        assert pool in ['f', 'fo', 'ifo']
        self.pool = pool
        self.pool_f = pooling_functions[pool]
        self.zoneout = zoneout
        self.center_conv = center_conv
        self.feed_state = feed_state
        self.num_in_channels = num_in_channels
        if center_conv:
            assert conv_size % 2 == 1
        init = tf.random_normal_initializer()
        filter_shape = [conv_size,
                        input_size,
                        1,  # should this be 1 or num_in_channels?
                        hidden_size*(len(pool)+1)]

        with tf.variable_scope('QRNN/conv/'+str(layer_id)):
            self.W = tf.get_variable('W', filter_shape,
                                     initializer=init, dtype=tf.float32)
            self.b = tf.get_variable('b', [hidden_size*(len(pool)+1)],
                                     initializer=init, dtype=tf.float32)
            if feed_state:
                self.W_v = tf.get_variable('W_v', [hidden_size,
                                                   hidden_size*(len(pool)+1)])
                self.b_v = tf.get_variable('b_v', [hidden_size*(len(pool)+1)])

    def __call__(self, inputs, state=None, train=None):
        inputs = tf.Print(inputs, [tf.reduce_sum(inputs)])
        gates = self.conv(inputs, state)
        gates[0] = tf.Print(gates[0], [tf.reduce_sum(gates)])
        if self.zoneout and self.zoneout > 0.0:
            F = gates[2]
            F = tf.cond(train,
                        lambda: 1-tf.nn.dropout(F, 1-self.zoneout),
                        lambda: F)
            gates[2] = F
        return self.pool_f(gates)

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
        gates = tf.split(conv, (len(self.pool)+1), 3)
        # Z, F, O dims: [batch x seq x in x num_conv]

        if self.feed_state:
            linear_state = tf.nn.xw_plus_b(state, self.W_v, self.b_v)
            # linear_state dims: [batch x state*3]
            state_gates = tf.split(linear_state, (len(self.pool)+1), 1)
            for g, s_g in zip(gates, state_gates):
                g += s_g

        # apply nonlinearities, turn into lists by seq
        gates[0] = tf.tanh(gates[0])
        for g in gates[1:]:
            g = tf.sigmoid(g)

        return gates  # list of gates ex. [Z, F, O]

    def f_pool(self, gates):
        Z, F = gates
        Z = tf.unstack(Z, axis=1)
        F = tf.unstack(F, axis=1)
        H = [tf.fill(tf.shape(Z[0]), 0.0)]
        for i in range(len(Z)):
            h = tf.multiply(F[i], H[-1]) + tf.multiply(1-F[i], Z[i])
            H.append(h)
        H = tf.stack(H[1:], axis=1)
        return tf.transpose(H, perm=[0, 1, 3, 2])

    def fo_pool(self, gates):
        Z, F, O = gates
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

    def ifo_pool(self, gates):
        Z, F, O, I = gates
        Z = tf.unstack(Z, axis=1)
        F = tf.unstack(F, axis=1)
        O = tf.unstack(O, axis=1)
        I = tf.unstack(O, axis=1)
        C = [tf.fill(tf.shape(Z[0]), 0.0)]
        H = []
        for i in range(len(Z)):
            c = tf.multiply(F[i], C[-1]) + tf.multiply(I[i], Z[i])
            h = tf.multiply(O[i], c)
            C.append(c)
            H.append(h)
        H = tf.stack(H, axis=1)
        return tf.transpose(H, perm=[0, 1, 3, 2])

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
                 num_layers, zoneout=0.0, dropout=0.0, center_conv=False,
                 feed_state=False):
        self.layers = []
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        for i in range(num_layers):
            pool = 'fo' if i is not 0 else 'f'
            layer = QRNNLayer(input_size, conv_size, hidden_size, layer_ids[i],
                              pool=pool, center_conv=center_conv,
                              zoneout=zoneout,
                              feed_state=feed_state, num_in_channels=i+1)
            self.layers.append(layer)

    def __call__(self, inputs, states=None, train=None):
        if states is None:
            states = [None] * self.num_layers
        inputs = tf.layers.dense(inputs, self.hidden_size)
        for layer in self.layers:
            outputs = layer(inputs, train=train)
            if self.dropout and self.dropout > 0:
                keep_prob = 1 - self.dropout
                outputs = tf.cond(train,
                                  lambda: tf.nn.dropout(outputs, keep_prob),
                                  lambda: outputs)
            inputs = tf.concat([inputs, outputs], 3)
        return tf.squeeze(outputs[:, :, :, -1])
