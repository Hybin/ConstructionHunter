from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras_contrib.layers import CRF
from keras_contrib import metrics, losses
import pickle


class Sword(object):
    def __init__(self, conf, embedding_dim=200, bi_rnn_units=200, epochs=10):
        self.conf = conf
        self.embedding_dim = embedding_dim
        self.bi_rnn_units = bi_rnn_units
        self.epochs = epochs

    def draw(self, character, construction):
        with open(self.conf.train_dict, "rb") as fp:
            vocabulary = pickle.load(fp)
        fp.close()

        # Model Configuration

        """
        # model.add(Embedding(len(vocabulary.keys()), self.embedding_dim, mask_zero=True))
        model.add(Bidirectional(LSTM(self.bi_rnn_units // 2, return_sequences=True)))

        crf = CRF(len(self.conf.labels), sparse_target=True)
        model.add(crf)

        model.summary()
        model.compile("adam", loss=losses.crf_loss, metrics=[metrics.crf_accuracy])

        return model
        """
