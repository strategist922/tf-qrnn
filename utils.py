import json
import os
import sys
import time
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from torch.utils.data import Dataset, DataLoader


def check_restore_parameters(sess, saver, checkpoint_path):
    try:
        os.mkdir(checkpoint_path)
    except OSError:
        pass

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        print "Loading parameters"
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print "Initializing fresh parameters"


def convert_to_embedding(v):
    return map(float, v)


def get_embeddings(vocab_dict, path):
    vocab_dict = {word: int(_id) for _id, word in vocab_dict.iteritems()}
    embed_id = path.split('.')[-2]
    if embed_id+'_imdb.json' not in os.listdir('.'):
        embeddings = {}
        with open(path) as f:
            for line in f:
                split = line.split()
                word = split[0]
                vec = split[1:]
                if word in vocab_dict.keys():
                    embeddings[vocab_dict[word]] = convert_to_embedding(vec)
        with open(embed_id+'_imdb.json', 'w') as f:
            f.write(json.dumps(embeddings))
    else:
        with open(embed_id+'_imdb.json') as f:
            embeddings = {int(_id): word for _id, word
                          in json.loads(f.read()).iteritems()}
    return embeddings


class imdbDataset(Dataset):
    def __init__(self, dataset, seq_len=100):
        self.x = dataset[0]
        self.pad_inputs(seq_len)
        self.get_lengths(seq_len)
        self.y = dataset[1]

    def pad_inputs(self, seq_len):
        new_xs = []
        for x in self.x:
            if len(x) > seq_len:
                x = x[:seq_len]
            elif len(x) < seq_len:
                x += [0] * (seq_len - len(x))
            assert len(x) == seq_len
            new_xs.append(x)
        self.x = new_xs

    def get_lengths(self, seq_len):
        self.lens = []
        for x in self.x:
            self.lens.append(len(x))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.lens[i], self.y[i]


def get_datasets(batch_size=100, num_words=50000, seq_len=100):
    train, test = imdb.load_data(num_words=num_words)
    vocab = imdb.get_word_index()
    vocab = {int(_id): word.encode('utf-8').lower() for word, _id
             in vocab.iteritems() if _id <= num_words}
    train = imdbDataset(train, seq_len=seq_len)
    n = len(test[0])
    dev = imdbDataset((test[0][0:n/2], test[1][0:n/2]), seq_len=seq_len)
    test = imdbDataset((test[0][n/2:], test[1][n/2:]), seq_len=seq_len)
    # return (DataLoader(train, batch_size, shuffle=True),
    #         DataLoader(dev, batch_size, shuffle=True),
    #         DataLoader(test, batch_size, shuffle=True),
    #         vocab)
    return (DataLoader(train, batch_size),
            DataLoader(dev, batch_size),
            DataLoader(test, batch_size),
            vocab)


def convert_to_np(x):
    if type(x[0]) == int:
        return x.numpy()
    else:
        return np.array([x_i.numpy() for x_i in x]).T


class Progbar(object):
    """
    Progbar class copied from CS224n starter code, which was copied from
    keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] /
                                             max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] /
                                             max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)
