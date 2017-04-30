import tensorflow as tf
import numpy as np
import corpus_parser
import random
import sys

input_seq_length = 10
output_seq_length = input_seq_length
batch_size = 10

data, voc_size = corpus_parser.get_corpus(seq_len=input_seq_length)

input_vocab_size = voc_size
output_vocab_size = voc_size
embedding_dim = 256

sess = tf.InteractiveSession()

encode_input = [tf.placeholder(tf.int32, 
                shape=(None,),
                name = 'ei_%i' %i)
                for i in range(input_seq_length)]

labels = [tf.placeholder(tf.int32,
            shape=(None,),
            name = 'l_%i' %i)
            for i in range(output_seq_length)]

# shift by 1
decode_input = [tf.zeros_like(encode_input[0], dtype=np.int32, name='GO')] + labels[:-1]


keep_prob = tf.placeholder('float')

# stacked to 3 layers
cells = [tf.contrib.rnn.DropoutWrapper(
        tf.contrib.rnn.BasicLSTMCell(embedding_dim), output_keep_prob=keep_prob
    ) for i in range(3)]

stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells)

with tf.variable_scope('decoders') as scope:
    # feed_previous=False, for training
    decode_outputs, decode_state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
        encode_input,
        decode_input,
        stacked_lstm,
        num_encoder_symbols=input_vocab_size,
        num_decoder_symbols=output_vocab_size,
        embedding_size=embedding_dim)
    
    scope.reuse_variables()
    
    decode_outputs_test, decode_state_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
        encode_input,
        decode_input,
        stacked_lstm,
        num_encoder_symbols=input_vocab_size,
        num_decoder_symbols=output_vocab_size,
        embedding_size=embedding_dim,
        feed_previous=True)

loss_weights = [tf.ones_like(l, dtype=tf.float32) for l in labels]
loss = tf.contrib.legacy_seq2seq.sequence_loss(decode_outputs, labels, loss_weights, output_vocab_size)

optimizer = tf.train.AdamOptimizer(0.001)
train_op = optimizer.minimize(loss)

sess.run(tf.global_variables_initializer())


class DataIterator:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        assert len(self.data) % batch_size == 0
        self.iter = self.make_random_iter()

        
    def next_batch(self):
        try:
            idxs = next(self.iter)
        except StopIteration:
            self.iter = self.make_random_iter()
            idxs = next(self.iter)
            
        X, Y = zip(*[self.data[i] for i in idxs])
        X = np.array(X).T
        Y = np.array(Y).T
        return X, Y

    def make_random_iter(self):
        n = len(self.data)
        shuffled_indexes = np.array(range(n))
        random.shuffle(shuffled_indexes)
        batch_indexes = [shuffled_indexes[i:i + self.batch_size] for i in range(0, n, self.batch_size)]
        return iter(batch_indexes)
    

print("#data points: " + str(len(data)))
data_train = data[0:6000]
data_val = data[6000:6400]
data_test = data[6400:6800]
train_iter = DataIterator(data_train, batch_size)
val_iter = DataIterator(data_val, batch_size)
test_iter = DataIterator(data_test, len(data_test))
test_iter_small = DataIterator(data_test, 5)
train_iter_small = DataIterator(data_train, 1)


# with tf.Session() as sess:
def get_feed(X, Y):
    feed_dict = {encode_input[t]: X[t] for t in range(input_seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(output_seq_length)})
    return feed_dict

def train_batch(data_iter):
    X, Y = data_iter.next_batch()
    feed_dict = get_feed(X, Y)
    feed_dict[keep_prob] = 0.6
    _, out = sess.run([train_op, loss], feed_dict)
    return out

def get_eval_batch_data(data_iter):
    X, Y = data_iter.next_batch()
    feed_dict = get_feed(X, Y)
    feed_dict[keep_prob] = 1.
    all_output = sess.run([loss] + decode_outputs_test, feed_dict)
    eval_loss = all_output[0]
    decode_output = np.array(all_output[1:]).transpose([1,0,2])
    return eval_loss, decode_output, X, Y

def eval_batch(data_iter, num_batches):
    losses = []
    predict_loss = []
    for i in range(num_batches):
        eval_loss, output, X, Y = get_eval_batch_data(data_iter)
        losses.append(eval_loss)
        
        for index in range(len(output)):
            real = Y.T[index]
            predict = np.argmax(output, axis = 2)[index]
            predict_loss.append(all(real==predict))
    return np.mean(losses), np.mean(predict_loss)


def print_test_output(output, X, Y):
    Y_predict = np.argmax(output, axis = 2)
    for i in range(len(output)):
        line_x = corpus_parser.vec_to_line(X.T[i])
        line_y_real = corpus_parser.vec_to_line(Y.T[i])
        line_y_predict = corpus_parser.vec_to_line(Y_predict[i])
        print('\t' + line_x + '(' + line_y_real + ") -> " + line_y_predict)
        

batches_per_epoch = int(data_train.shape[0] / batch_size)

for i in range(1000000):
    # print('#batch: ' + str(i))
    try:
        train_batch(train_iter)
        if i % (batches_per_epoch) == 0:
            epoch_num = int(i / (batches_per_epoch))
            print('epoch #' + str(epoch_num))
            val_loss, val_predict = eval_batch(val_iter, 3)
            train_loss, train_predict = eval_batch(train_iter, 3)
            print('val loss   : %f, val predict   = %.1f%%' %(val_loss, val_predict * 100))
            print('train loss : %f, train predict = %.1f%%' %(train_loss, train_predict * 100))

            if epoch_num % 5 == 0:
                _, output, X, Y = get_eval_batch_data(train_iter_small)
                print('train sample')
                print_test_output(output, X, Y)

                _, output, X, Y = get_eval_batch_data(test_iter_small)
                print('test samples')
                print_test_output(output, X, Y)
                print('')
            sys.stdout.flush()

            if epoch_num >= 100:
                # _, output, X, Y = get_eval_batch_data(test_iter)
                # print('test results')
                # print_test_output(output, X, Y)
                sys.stdout.flush()
                break


    except KeyboardInterrupt:
        print('interrupted by user')
        break

