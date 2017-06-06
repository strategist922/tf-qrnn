import tensorflow as tf
from tf_qrnn import QRNNLayer, fo_pool


class SentimentModel:
    def __init__():
        pass

    def _get_embeddings(self, ids):
        # return dims: [batch x seq x state x 1]
        embeddings = tf.nn.embedding_lookup(self.embeddings, ids)
        return tf.expand_dims(embeddings, -1)


class VanillaNNModel(SentimentModel):
    def __init__(self, embeddings, BATCH_SIZE, SEQ_LEN):
        self.embeddings = embeddings

        self.inputs = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LEN],
                                     name="inputs")
        self.masks = tf.placeholder(tf.float32, [BATCH_SIZE, SEQ_LEN],
                                    name="mask")
        self.labels = tf.placeholder(tf.int32, [BATCH_SIZE], name="labels")

        inputs = self.inputs
        masks = self.masks
        labels = self.labels

        # set up loss function
        num_layers = 4
        num_convs = 256
        x = tf.squeeze(self._get_embeddings(inputs))
        for i in range(num_layers):
            x = tf.layers.dense(x, num_convs)
            x = tf.nn.dropout(x, keep_prob=0.7)

        x = tf.squeeze(x)  # dims: [batch x seq x state]

        # TODO
        # linear layer to condense state
        # then multiply by mask
        # avg over sequence
        # take softmax predictions
        # etc

        outputs = tf.layers.dense(x, 1)
        outputs = tf.squeeze(outputs) * masks
        # dims: [batch x seq]
        logits = tf.squeeze(tf.layers.dense(outputs, 2))

        pred = tf.nn.softmax(logits)
        pred = tf.argmax(pred, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=labels)
        self.cost = tf.reduce_sum(loss) / BATCH_SIZE
        correct_prediction = tf.equal(tf.cast(pred, tf.int32), labels)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction,
                                              tf.float32)) / BATCH_SIZE
        self.average_guess = tf.reduce_sum(pred)

        # TODO l2 regularization

        # set up optimizer
        self.op = tf.train.AdamOptimizer().minimize(loss)

        # set up train vars
        self.epoch = tf.Variable(0,
                                 dtype=tf.int32,
                                 trainable=False,
                                 name='epoch')
        self.best_dev_loss = tf.Variable(float('inf'),
                                         dtype=tf.float32,
                                         trainable=False,
                                         name='best_dev_loss')


class QRNNModel(SentimentModel):
    def __init__(self, embeddings, BATCH_SIZE, SEQ_LEN):
        self.embeddings = embeddings

        self.inputs = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LEN],
                                     name="inputs")
        self.masks = tf.placeholder(tf.float32, [BATCH_SIZE, SEQ_LEN],
                                    name="mask")
        self.labels = tf.placeholder(tf.int32, [BATCH_SIZE], name="labels")

        inputs = self.inputs
        masks = self.masks
        labels = self.labels

        # set up loss function
        num_layers = 4
        input_size = 300
        num_convs = 256
        conv_size = 3
        x = self._get_embeddings(inputs)
        for i in range(num_layers):
            print 'initializing qrnn layer', i
            if i == 0:
                layer = QRNNLayer(input_size, conv_size, num_convs, str(i))
            else:
                layer = QRNNLayer(num_convs, conv_size, num_convs, str(i))
            Z, F, O = layer.conv(x)
            F = 1 - tf.nn.dropout(F, keep_prob=0.7)
            x = fo_pool(Z, F, O)  # dims: [batch x seq x state x in]
        x = tf.squeeze(x)  # dims: [batch x seq x state]

        # TODO
        # linear layer to condense state
        # then multiply by mask
        # avg over sequence
        # take softmax predictions
        # etc

        outputs = tf.layers.dense(x, 1)
        outputs = tf.squeeze(outputs) * masks
        # dims: [batch x seq]
        logits = tf.squeeze(tf.layers.dense(outputs, 2))

        pred = tf.nn.softmax(logits)
        pred = tf.argmax(pred, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=labels)
        self.cost = tf.reduce_sum(loss) / BATCH_SIZE
        correct_prediction = tf.equal(tf.cast(pred, tf.int32), labels)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction,
                                              tf.float32)) / BATCH_SIZE
        self.average_guess = tf.reduce_sum(pred)

        # TODO l2 regularization

        # set up optimizer
        self.op = tf.train.RMSPropOptimizer(0.001).minimize(loss)

        # set up train vars
        self.epoch = tf.Variable(0,
                                 dtype=tf.int32,
                                 trainable=False,
                                 name='epoch')
        self.best_dev_loss = tf.Variable(float('inf'),
                                         dtype=tf.float32,
                                         trainable=False,
                                         name='best_dev_loss')
