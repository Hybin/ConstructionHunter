from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sword import Sword
from sklearn.model_selection import train_test_split
from processor import Processor
from tqdm import tqdm
from seqeval.metrics import classification_report
from utils import *
import numpy as npy
import pickle
import datetime


class Hades(object):
    def __init__(self, conf, hermes):
        self.conf = conf
        self.processor = Processor(conf)
        self.sword = Sword(conf)
        self.hermes = hermes

    def process_train_data(self):
        """
        Process the training data
        :return: x:np.array, y: np.array
        """
        sequences = self.processor.on(system="hades")

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

        seq_features = []
        for seq in cxn_feature:
            seq_features.append(to_categorical(seq, num_classes=4))

        seq_features = npy.array(seq_features)

        y = pad_sequences(labels, max_len, value=4)
        y = [to_categorical(i, num_classes=5) for i in y]

        x = merge(word_feature, seq_features)

        return x, y

    def process_test_data(self):
        """
        Process the test data
        :return: x: np.array
        """
        sequences, belonging = self.processor.up(system="hades")

        sentences, word_feature, cxn_feature, max_len = [], [], [], 0
        for sentence, index, sequence in sequences:
            if max_len < len(sentence):
                max_len = len(sentence)

            word_feature.append(index)
            cxn_feature.append(sequence)
            sentences.append(sentence)
        max_len += 1

        word_feature = pad_sequences(word_feature, max_len)
        cxn_feature = pad_sequences(cxn_feature, max_len)

        seq_features = []
        for seq in cxn_feature:
            seq_features.append(to_categorical(seq, num_classes=4))

        seq_features = npy.array(seq_features)
        x = merge(word_feature, seq_features)

        return sentences, x, belonging

    def train(self):
        """
        Train the model
        :return:
        """
        # Model Configuration
        model = self.sword.draw()

        # Load the data
        x, y = self.process_train_data()

        # Create the train data and validate data
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=24)
        train_x_kanji, train_x_cxn = resolve(train_x)

        with open(self.conf.validate_set.format("hades"), "wb") as fp:
            pickle.dump((test_x, test_y), fp)
        fp.close()

        # Train the model
        model.fit([train_x_kanji, train_x_cxn], npy.array(train_y), batch_size=16, epochs=self.sword.epochs, validation_split=0.2)
        model.save(self.conf.model_path.format("hades"))

    def evaluate(self):
        """
        Evaluate the model
        :return:
        """
        # Model Configuration
        model = self.sword.draw()

        # Load the test data
        # sentences, x = self.process_test_data()
        # x_kanji, x_cxn = resolve(x)
        with open(self.conf.validate_set.format("hades"), "rb") as fp:
            test_x, test_y = pickle.load(fp)
        fp.close()
        test_x_kanji, test_x_cxn = resolve(test_x)
        test_y = npy.argmax(test_y, axis=-1)

        # Load the model
        self.hermes.info("Load the model.bin.hades...")
        model.load_weights(self.conf.model_path.format("hades"))
        self.hermes.info("Load the model.bin.hades...Finished!")

        # Make the predictions
        predictions = model.predict([test_x_kanji, test_x_cxn])
        predictions = npy.argmax(predictions, axis=-1)

        self.hermes.info("Begin to predict...")

        pred_y, gold_y, count = [], [], 0
        for prediction in tqdm(predictions, desc="Predict"):
            sentence = test_x_kanji[count]
            len_of_sentence = get_length(sentence)
            prediction = prediction[-len_of_sentence:]
            labels = [self.conf.labels[label] for label in test_y[count][-len_of_sentence:]]
            gold_y.append(labels)

            label_index = [self.conf.labels[row] for row in prediction]
            pred_y.append(label_index)

            count += 1

        report = classification_report(gold_y, pred_y)
        print(report)

        self.hermes.info("Begin to predict...Finished!")

    def predict(self):
        """
        Predict the test set with trained model
        :return:
        """
        # Model Configuration
        model = self.sword.draw()

        # Load the test data
        sentences, x, belonging = self.process_test_data()
        x_kanji, x_cxn = resolve(x)

        # Load the model
        self.hermes.info("Load the model.bin.hades...")
        model.load_weights(self.conf.model_path.format("hades"))
        self.hermes.info("Load the model.bin.hades...Finished!")

        # Make the predictions
        predictions = model.predict([x_kanji, x_cxn])
        predictions = npy.argmax(predictions, axis=-1)

        self.hermes.info("Begin to predict...")

        results, count = [], 0
        for prediction in tqdm(predictions, desc="Predict"):
            sentence = sentences[count]
            len_of_sentence = get_length(sentence)
            prediction = prediction[-len_of_sentence:]

            labels = [self.conf.labels[row] for row in prediction]

            result = []
            for i in range(0, len_of_sentence):
                result.append((sentence[i], labels[i]))

            construction = belonging["".join(sentence)]
            results.append((construction, result))

            count += 1

        self.hermes.info("Begin to predict...Finished!")

        return results

    def write(self, results):
        self.hermes.info("Begin to write the annotation...")

        output = {}
        for construction, result in results:
            if construction not in output.keys():
                output[construction] = []

            output[construction].append(txt_to_xml(result))

        # Write the data to output file
        today = datetime.date.today().__format__('%Y_%m_%d')
        for construction, sentences in tqdm(output.items(), desc="Output the predictions"):
            with open(self.conf.output.format("hades", today, construction), "w") as fp:
                fp.write("<?xml version='1.0' encoding='UTF-8'?>" + "\r\n")
                fp.write("<document>" + "\r\n")

                for sentence in sentences:
                    fp.write("\t" + sentence + "\r\n")

                fp.write("</document>")

            fp.close()

        self.hermes.info("Begin to write the annotation...Finished!")


