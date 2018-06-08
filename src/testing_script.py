import os
import sys

sys.path.append('../')

import collections
import time
import numpy

numpy.random.seed(1337)
from sklearn import metrics
from keras.models import Sequential, model_from_json
from keras.layers.core import Dropout, Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
from collections import defaultdict
import src.data_processing.data_handler as dh

import argparse

class sarcasm_model():
    _train_file = None
    _test_file = None
    _tweet_file = None
    _output_file = None
    _model_file_path = None
    _word_file_path = None
    _split_word_file_path = None
    _emoji_file_path = None
    _vocab_file_path = None
    _input_weight_file_path = None
    _vocab = None
    _line_maxlen = None

    def __init__(self):
        self._line_maxlen = 30

    def _build_network(self, vocab_size, maxlen, embedding_dimension=256, hidden_units=256, trainable=False):
        print('Build model...')
        model = Sequential()

        model.add(
            Embedding(vocab_size, embedding_dimension, input_length=maxlen, embeddings_initializer='glorot_normal'))

        model.add(Convolution1D(hidden_units, 3, kernel_initializer='he_normal', padding='valid', activation='sigmoid',
                                input_shape=(1, maxlen)))
        # model.add(MaxPooling1D(pool_size=3))
        model.add(Convolution1D(hidden_units, 3, kernel_initializer='he_normal', padding='valid', activation='sigmoid',
                                input_shape=(1, maxlen - 2)))
        # model.add(MaxPooling1D(pool_size=3))

        # model.add(Dropout(0.25))

        model.add(LSTM(hidden_units, kernel_initializer='he_normal', activation='sigmoid', dropout=0.5,
                       return_sequences=True))
        model.add(LSTM(hidden_units, kernel_initializer='he_normal', activation='sigmoid', dropout=0.5))

        model.add(Dense(hidden_units, kernel_initializer='he_normal', activation='sigmoid'))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        adam = Adam(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        print('No of parameter:', model.count_params())

        print(model.summary())
        return model


class test_model(sarcasm_model):
    test = None
    model = None

    def __init__(self, model_file, word_file_path, split_word_path, emoji_file_path, vocab_file_path, output_file,
                 input_weight_file_path=None):
        print('initializing...')
        sarcasm_model.__init__(self)

        self._model_file_path = model_file
        self._word_file_path = word_file_path
        self._split_word_file_path = split_word_path
        self._emoji_file_path = emoji_file_path
        self._vocab_file_path = vocab_file_path
        self._output_file = output_file
        self._input_weight_file_path = input_weight_file_path

        print('test_maxlen', self._line_maxlen)

    def load_trained_model(self, model_file='model.json', weight_file='model.json.hdf5'):
        start = time.time()
        self.__load_model(self._model_file_path + model_file, self._model_file_path + weight_file)
        end = time.time()
        print('model loading time::', (end - start))

    def __load_model(self, model_path, model_weight_path):
        self.model = model_from_json(open(model_path).read())
        print('model loaded from file...')
        self.model.load_weights(model_weight_path)
        print('model weights loaded from file...')

    def load_vocab(self):
        vocab = defaultdict()
        with open(self._vocab_file_path, 'r') as f:
            for line in f.readlines():
                key, value = line.split('\t')
                vocab[key] = value

        return vocab

    def predict(self, test_file, verbose=False):
        try:
            start = time.time()
            self.test = dh.loaddata(test_file, self._word_file_path, self._split_word_file_path, self._emoji_file_path,
                                    normalize_text=True, split_hashtag=True,
                                    ignore_profiles=False)
            end = time.time()
            if (verbose == True):
                print('test resource loading time::', (end - start))

            self._vocab = self.load_vocab()
            print('vocab loaded...')

            start = time.time()
            tX, tY, tD, tC, tA = dh.vectorize_word_dimension(self.test, self._vocab)
            tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)
            end = time.time()
            if (verbose == True):
                print('test resource preparation time::', (end - start))
            # self.__predict_model(tX, self.test)
        except Exception as e:
            print('Error:', e)
            raise

    def __predict_model(self, tX, test):
        y = []
        y_pred = []

        prediction_probability = self.model.predict_proba(tX, batch_size=1, verbose=1)

        try:
            fd = open(self._output_file + '.analysis', 'w')
            for i, (label) in enumerate(prediction_probability):
                gold_label = test[i][1]
                words = test[i][2]
                dimensions = test[i][3]
                context = test[i][4]
                author = test[i][5]

                predicted = numpy.argmax(prediction_probability[i])

                y.append(int(gold_label))
                y_pred.append(predicted)

                fd.write(str(label[0]) + '\t' + str(label[1]) + '\t'
                         + str(gold_label) + '\t'
                         + str(predicted) + '\t'
                         + ' '.join(words))

                fd.write('\n')

            print()

            print('accuracy::', metrics.accuracy_score(y, y_pred))
            print('precision::', metrics.precision_score(y, y_pred, average='weighted'))
            print('recall::', metrics.recall_score(y, y_pred, average='weighted'))
            print('f_score::', metrics.f1_score(y, y_pred, average='weighted'))
            print('f_score::', metrics.classification_report(y, y_pred))
            fd.close()
        except Exception as e:
            print(e)
            raise

    def predict_once(self, sent):
        y = []
        y_pred = []
        self._vocab = self.load_vocab()
        print('vocab loaded...')

        tX = dh.parse_sent(sent, self._word_file_path, self._split_word_file_path, self._emoji_file_path, self._vocab, 
                                    normalize_text=True, split_hashtag=True,
                                    ignore_profiles=False)


        tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)

        prediction_probability = self.model.predict_proba(tX, batch_size=1, verbose=1)
        for i, (label) in enumerate(prediction_probability):
            predicted = numpy.argmax(prediction_probability[i])
            y_pred.append(predicted)

            print(predicted)

if __name__ == "__main__":
    basepath = os.getcwd()[:os.getcwd().rfind('/')]
    train_file = basepath + '/resource/train/Train_v1.txt'
    validation_file = basepath + '/resource/dev/Dev_v1.txt'
    test_file = basepath + '/resource/test/Test_v1.txt'
    word_file_path = basepath + '/resource/word_list_freq.txt'
    split_word_path = basepath + '/resource/word_split.txt'
    emoji_file_path = basepath + '/resource/emoji_unicode_names_final.txt'

    CNN_LSTM_path = '/resource/text_model_CNN_LSTM/' 
    CNN_LSTM_simpler_path = '/resource/text_model_CNN_LSTM_simpler/' 
    CNN_LSTM_word2vec_path = '/resource/text_model_word2vec/'
    CNN_DNN_path = '/resource/text_model_2D/'

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('which_model', metavar='N', type=int, nargs='?', const=1, 
                        help='indicate which model, 1: CNN_LSTM, 2: CNN_LSTM_simpler, 3: CNN_LSTM_word2vec, 4: CNN_DNN')

    args = parser.parse_args()

    which_path = None
    if args.which_model == 1:
        which_path = CNN_LSTM_path
    elif args.which_model == 2:
        which_path = CNN_LSTM_simpler_path
    elif args.which_model == 3:
        which_path = CNN_LSTM_word2vec_path
    elif args.which_model == 4:
        which_path = CNN_DNN_path

    output_file = basepath + which_path + 'TestResults.txt'
    model_file = basepath + which_path + 'weights/'
    vocab_file_path = basepath + which_path + 'vocab_list.txt'

    t = test_model(model_file, word_file_path, split_word_path, emoji_file_path, vocab_file_path, output_file)
    t.load_trained_model(weight_file='model.json.hdf5')
    # t.predict(test_file)
    t.predict_once("If u think u r so perfect , then don't even speak to me .")