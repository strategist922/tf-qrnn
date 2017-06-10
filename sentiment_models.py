import tensorflow as tf
from tf_qrnn import QRNNLayer, DenseQRNNLayers


class SentimentModel:
    def __init__(self, embeddings, BATCH_SIZE, SEQ_LEN, beta=4e-6):
        self.batch_size = BATCH_SIZE
        self.seq_len = SEQ_LEN
        self.embeddings = embeddings

        self.inputs = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LEN],
                                     name="inputs")
        self.masks = tf.placeholder(tf.float32, [BATCH_SIZE, SEQ_LEN],
                                    name="mask")
        self.labels = tf.placeholder(tf.int32, [BATCH_SIZE], name="labels")

        self.train = tf.placeholder(tf.bool, [], name='train')

        x, weights = self.forward()
        loss = self.inference(x)
        # if weights:
        #     for w in weights:
        #         loss += beta*tf.nn.l2_loss(w)
        self.setup_learning(loss)

    def forward(self):
        raise NotImplementedError

    def inference(self, x):
        masks = self.masks
        labels = self.labels

        outputs = tf.layers.dense(x, 1)
        outputs = tf.squeeze(outputs) * masks
        # dims: [batch x seq]
        logits = tf.squeeze(tf.layers.dense(outputs, 2))

        pred = tf.nn.softmax(logits)
        pred = tf.argmax(pred, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=labels)
        self.cost = tf.reduce_sum(loss) / self.batch_size
        correct_prediction = tf.equal(tf.cast(pred, tf.int32), labels)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction,
                                              tf.float32)) / self.batch_size
        self.average_guess = tf.reduce_sum(pred)
        return loss

    def setup_learning(self, loss):
        self.op = tf.train.AdamOptimizer().minimize(loss)

        # set up train vars
        self.epoch = tf.Variable(0,
                                 dtype=tf.int32,
                                 trainable=False,
                                 name='epoch')
        self.best_dev_acc = tf.Variable(0.0,
                                        dtype=tf.float32,
                                        trainable=False,
                                        name='best_dev_acc')

    def _get_embeddings(self, ids):
        # return dims: [batch x seq x state x 1]
        embeddings = tf.nn.embedding_lookup(self.embeddings, ids)
        return tf.expand_dims(embeddings, -1)


class VanillaNNModel(SentimentModel):
    def forward(self):
        inputs = self.inputs

        # set up loss function
        num_layers = 4
        num_convs = 256
        x = tf.squeeze(self._get_embeddings(inputs))
        for i in range(num_layers):
            x = tf.layers.dense(x, num_convs)
            x = tf.sigmoid(x)
            x = tf.cond(self.train,
                        lambda: tf.nn.dropout(x, 0.7),
                        lambda: x)

        x = tf.squeeze(x)  # dims: [batch x seq x state]
        return x, None


class QRNNModel(SentimentModel):
    def forward(self):
        inputs = self.inputs

        num_layers = 4
        input_size = 300
        num_convs = 256
        conv_size = 2
        x = self._get_embeddings(inputs)
        weights = []
        for i in range(num_layers):
            print 'initializing qrnn layer', i
            in_size = input_size if i == 0 else num_convs
            layer = QRNNLayer(in_size, conv_size, num_convs, str(i),
                              zoneout=0.0)
            weights.append(layer.W)
            weights.append(layer.b)
            x = layer(x, train=self.train)
            x = tf.cond(self.train,
                        lambda: tf.nn.dropout(x, 0.7),
                        lambda: x)
        x = tf.squeeze(x)  # dims: [batch x seq x state]
        return x, weights


class DenseQRNNModel(SentimentModel):
    def forward(self):
        inputs = self.inputs

        num_layers = 4
        input_size = 300
        num_convs = 256
        conv_size = 2
        x = self._get_embeddings(inputs)
        qrnn = DenseQRNNLayers(input_size,
                               conv_size,
                               num_convs,
                               range(num_layers),
                               num_layers,
                               dropout=0.3)
        x = qrnn(x, train=self.train)
        weights = [l.W for l in qrnn.layers] + [l.b for l in qrnn.layers]
        return tf.squeeze(x), weights


class LSTMModel(SentimentModel):
    def forward(self):
        inputs = self.inputs

        num_layers = 4
        hidden_size = 256

        x = self._get_embeddings(inputs)
        x = tf.squeeze(x)
        cells = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.BasicLSTMCell(hidden_size)
            for i in range(num_layers)
        ])
        initial_state = cells.zero_state(self.batch_size, dtype=tf.float32)
        x = tf.nn.dynamic_rnn(cells, x, initial_state=initial_state)

        return tf.squeeze(x), None
