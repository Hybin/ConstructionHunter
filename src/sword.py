from keras import Model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Input, concatenate
from keras.utils import plot_model
import pickle


class Sword(object):
    def __init__(self, conf, embedding_dim=256, bi_rnn_units=128, epochs=10):
        self.conf = conf
        self.embedding_dim = embedding_dim
        self.bi_rnn_units = bi_rnn_units
        self.epochs = epochs

    def draw(self):
        with open(self.conf.train_dict, "rb") as fp:
            vocabulary = pickle.load(fp)
        fp.close()

        # Model Configuration
        input_character = Input(shape=(None, ), name="character")
        feature_character = Embedding(len(vocabulary.keys()) + 1, self.embedding_dim, mask_zero=True)(input_character)
        feature_character = Dropout(0.1)(feature_character)
        feature_character = Bidirectional(LSTM(self.bi_rnn_units // 2, return_sequences=True, recurrent_dropout=0.1))(feature_character)

        input_construction = Input(shape=(None, 4), name="cxn")

        model = concatenate([feature_character, input_construction])
        model = Bidirectional(LSTM(self.bi_rnn_units // 2, return_sequences=True, recurrent_dropout=0.6))(model)
        output = TimeDistributed(Dense(5, activation="softmax"))(model)

        model = Model(inputs=[input_character, input_construction], outputs=output)
        plot_model(model, self.conf.model_image.format("multi_input_and_output_model.png"), show_shapes=True)

        model.compile("rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()

        return model
