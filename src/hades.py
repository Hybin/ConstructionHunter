from keras.preprocessing.sequence import pad_sequences
from sword import Sword
from sklearn.model_selection import train_test_split
from utils import *
import numpy as npy


class Hades(object):
    def __init__(self, conf, processor):
        self.conf = conf
        self.processor = processor
        self.sword = Sword(conf)

    def process_train_data(self):
        """
        Process the training data
        :return: x:np.array, y: np.array
        """
        sequences = self.processor.on()

        word_feature, cxn_feature, labels, max_len = [], [], [], 0
        for sentence, index, sequence, label in sequences:
            if max_len < len(sentence):
                max_len = len(sentence)

            word_feature.append(index)
            cxn_feature.append(sequence)
            labels.append(label)
        max_len += 1

        word_feature = pad_sequences(word_feature, max_len)
        cxn_feature = pad_sequences(cxn_feature, max_len)

        y = pad_sequences(labels, max_len, value=5)
        y = y.reshape(y.shape[0], y.shape[1], 1)

        x = []
        for index, sequence in zip(word_feature, cxn_feature):
            x.append(merge(index, sequence))

        x = npy.array(x)

        return x, y

    def process_test_data(self):
        """
        Process the test data
        :return: x: np.array
        """
        sequences = self.processor.up()

        word_feature, cxn_feature, max_len = [], [], 0
        for sentence, index, sequence in sequences:
            if max_len < len(sentence):
                max_len = len(sentence)

            word_feature.append(index)
            cxn_feature.append(sequence)
        max_len += 1

        word_feature = pad_sequences(word_feature, max_len)
        cxn_feature = pad_sequences(cxn_feature, max_len)
        print(word_feature.shape)
        print(cxn_feature.shape)

        x = []
        for index, sequence in zip(word_feature, cxn_feature):
            x.append(merge(index, sequence))

        x = npy.array(x)

        return x

    def train(self):
        # Model Configuration
        model = self.sword.draw()

        # Load the data
        x, y = self.process_train_data()

        # Create the train data and validate data
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=24)

        # Train the model
        model.fit(train_x, train_y, batch_size=16, epochs=self.sword.epochs, validation_data=[test_x, test_y])
        model.save(self.conf.model_path.format("hades"))

    def predict(self):
        # Model Configuration
        model = self.sword.draw()

        # Load the test data
        x = self.process_test_data()

        # Load the model
        model.load_weights(self.conf.model_path.format("hades"))

        # Make the predictions
        predictions = model.predict(x)
        print(predictions)
