import os
import sys
import tensorflow as tf

import utils

from time import time
import sentiment_models as Models


models = {
    'nn': Models.VanillaNNModel,
    'qrnn': Models.QRNNModel,
    'dense': Models.DenseQRNNModel,
    'lstm': Models.LSTMModel
}


NUM_EPOCHS = 10
BATCH_SIZE = 25
SEQ_LEN = 400
VOCAB_SIZE = 68379
CKPT_PATH = './sentiment_checkpoints'
# TODO
# try LSTMModel
# see if checkpointing works
# try dense
# do penn treebank
# some seq2seq task


def run(model, sess, dataset, train=False):
    j = 0
    prog = utils.Progbar(len(dataset))
    avg_loss = 0.0
    avg_correct = 0.0
    for x, masks, y in dataset:
        input_feed = {
            model.inputs: utils.convert_to_np(x),
            model.masks: utils.convert_to_np(masks),
            model.labels: utils.convert_to_np(y),
            model.train: train
        }
        if train:
            output_feed = [model.cost, model.accuracy,
                           model.average_guess, model.op]
            cost, num_correct, avg_guess, _ = sess.run(output_feed, input_feed)
        else:
            output_feed = [model.cost, model.accuracy, model.average_guess]
            cost, num_correct, avg_guess = sess.run(output_feed, input_feed)
        avg_loss += cost
        avg_correct += num_correct
        j += 1
        prog.update(j,
                    values=[('num_correct', num_correct)],
                    exact=[('cost', cost)])
    avg_loss /= len(dataset)
    avg_correct /= len(dataset)
    return avg_correct, avg_loss


def train(vocab, embeddings, train_data, dev_data, test_data):
    Model = models[sys.argv[1]]
    model = Model(embeddings, BATCH_SIZE, SEQ_LEN)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        utils.check_restore_parameters(sess, saver, CKPT_PATH)
        best_dev_acc = model.best_dev_acc.eval()
        epoch = model.epoch.eval()

        for i in range(epoch, NUM_EPOCHS):
            start = time()

            print 'epoch', i
            sess.run(tf.assign(model.epoch, i))
            # train
            print 'training'
            train_acc, train_loss = run(model, sess, train_data, train=True)
            print
            print 'average train loss', train_loss

            # eval on dev
            print 'evaluating dev'
            dev_acc, dev_loss = run(model, sess, dev_data)
            print
            print 'average dev loss:', dev_loss

            # eval on test
            print 'evaluating test'
            test_acc, test_loss = run(model, sess, test_data)
            print
            print 'average test loss:', test_loss

            if best_dev_acc is None or dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                sess.run(tf.assign(model.best_dev_acc, best_dev_acc))
                saver.save(sess, os.path.join(CKPT_PATH, sys.argv[1]))
                print 'saved new best dev acc'

            print 'epoch', i, 'took', time()-start, 'seconds'


def init_embeddings(embeddings, vocab, dim):
    init = tf.contrib.layers.xavier_initializer()
    embed_list = []
    num_vars = 0
    num_const = 0
    for i in range(3):
        var = tf.Variable(init(shape=[dim]), dtype=tf.float32)
        embed_list.append(var)
    for _id, word in vocab.iteritems():
        if int(_id) in embeddings.keys():
            embed_list.append(tf.constant(embeddings[_id], dtype=tf.float32))
            num_const += 1
        else:
            # var = tf.Variable(init(shape=[dim]))
            # embed_list.append(var)
            embed_list.append(embed_list[2])
            num_vars += 1
    print num_const, num_vars
    return tf.stack(embed_list, axis=0)


if __name__ == '__main__':
    assert sys.argv[1] in models.keys()
    print 'using model', sys.argv[1]

    print 'loading data'
    start = time()
    trainset, dev, test, vocab = utils.get_datasets(batch_size=BATCH_SIZE,
                                                    num_words=VOCAB_SIZE,
                                                    seq_len=SEQ_LEN)
    print 'took', time() - start, 'seconds'
    start = time()
    print 'getting embeddings'
    embeddings = utils.get_embeddings(vocab, './glove.6B/glove.6B.300d.txt')
    print 'took', time() - start, 'seconds'
    print 'initializing embeddings'
    start = time()
    embeddings = init_embeddings(embeddings, vocab, 300)
    print 'took', time() - start, 'seconds'
    print 'begin training'
    train(vocab, embeddings, trainset, dev, test)
