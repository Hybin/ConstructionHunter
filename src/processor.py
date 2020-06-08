from utils import *
from stanfordcorenlp import StanfordCoreNLP
import pickle
from tqdm import tqdm
import os
import re


class Processor(object):
    def __init__(self, conf):
        self.conf = conf
        # Use StanfordCoreNLP API
        # self.parser = StanfordCoreNLP(self.conf.stanford_core_nlp, lang="zh", quiet=False, memory="8g")
        # Use StanfordCoreNLPServer (Recommend! Faster!)
        # self.parser = StanfordCoreNLP(self.conf.core_nlp_host, port=self.conf.core_nlp_port, lang="zh")

    # ======================================
    # Process the train data of Hades
    # ======================================
    def load_training_data(self, file):
        """ Load the training data
         :param file: string
         :return: training_data: list[tuple]
         """
        with open(self.conf.train_data.format(file), "r", encoding="utf-8") as fp:
            lines = fp.readlines()

            training_data, sentence, labels = [], [], []
            for line in lines:
                line = line.strip()

                if len(line) != 0:
                    sentence.append(line.split()[0])
                    labels.append(line.split()[1])
                else:
                    training_data.append((sentence, labels))
                    sentence, labels = [], []

        fp.close()

        return training_data

    def _create_dictionary(self):
        """
        Create word-index dictionary of training data
        :return:
        """
        frequency = {}

        files = os.listdir("../data/train")
        for file in files:
            data = self.load_training_data(file)
            for sentence, labels in data:
                for word in sentence:
                    if word not in frequency.keys():
                        frequency[word] = 1
                    else:
                        frequency[word] += 1

        vocabulary = [word for word, freq in frequency.items() if freq >= 2]
        word_index = dict((word, index+1) for index, word in enumerate(vocabulary))

        with open(self.conf.train_dict, "wb") as fp:
            pickle.dump(word_index, fp)

        fp.close()

    def get_constituency(self, sentence):
        # tree = self.parser.parse(sentence)
        tree = ""
        return tree

    @staticmethod
    def get_components(file):
        """
         Get the constants and variables from the construction form
         :param file: string
         :return: constants: list[string]
         """
        construction = file.split(".")[0].split("_")[1].split("+")
        cxn_without_comma = [element for element in construction if element != "，"]

        components, constants, previous = [], [], ""
        for component in cxn_without_comma:
            if len(re.search(r'[a-zA-Z0-9]*', component).group()) == 0:
                if previous == "constant":
                    components[-1][1] = components[-1][1] + component
                    continue
                previous = "constant"
            else:
                previous = "variable"
            components.append([previous, component])

        previous = ""
        for component in construction:
            if len(re.search(r'[a-zA-Z0-9，]*', component).group()) == 0:
                if previous == "constant":
                    constants[-1] = constants[-1] + component
                    continue
                constants.append(component)
                previous = "constant"
            else:
                previous = "variable"

        return components, constants

    def cut_sentence(self, sentence, constants):
        segments = Record()

        prev, next = 0, 0
        for constant in constants:
            next = sentence.find(constant, prev)

            if next != -1:
                segments.push(prev, (prev, next))
                segments.push(constant, (next, next + len(constant)))
                next += len(constant)
                prev = next

        if prev != len(sentence) - 1:
            if prev == 0:
                return Record()
            else:
                next_segments = self.cut_sentence(sentence[prev:len(sentence)], constants)

                if next_segments.size() == 0:
                    segments.push(prev, (prev, len(sentence)))
                else:
                    for key, value, visited in next_segments.container:
                        if type(key) == str:
                            segments.push(key, (prev + value[0], prev + value[1]))
                        else:
                            segments.push(prev + key, (prev + value[0], prev + value[1]))

        return segments

    @staticmethod
    def _extract_comma(file, sequence, sentence):
        """
        Extract the feature: comma
        :param file: string
        :param sequence: list[int]
        :param sentence: string
        :return: sequence: list[int]
        """
        if "，" in file:
            commas = [index for index, word in enumerate(sentence) if word == "，"]
            for comma in commas:
                if sequence[comma - 1] != 0 and sequence[comma + 1] != 0:
                    sequence[comma] = 3

        return sequence

    @staticmethod
    def _extract_word(sentence, word_index):
        """
        Extract the feature: word
        :param sentence: string
        :param word_index: dict
        :return: list[int]
        """
        indice = []

        for word in sentence:
            indice.append(word_index.get(word, 0))

        return indice

    def _extract_components(self, components, sentence, segments, sequence):
        """
        Extract the feature: constants and variables
        :param components: list[tuple]
        :param sentence: string
        :param segments: Record
        :param sequence: list[int]
        :return: sequence: list[int]
        """
        current, previous = 0, "context"

        for component in components:
            if component[0] == "constant":
                if segments.get(component[1]) is not None:
                    left, right = segments.get(component[1])
                    replace(sequence, left, right, 2)
                    segments.visit(component[1], (left, right))
                    current = right
                    previous = "constant"
            else:
                left, right = segments.get(current)
                phrases = [phrase for phrase in re.split(r"[，。？！；：]", sentence[left:right]) if len(phrase) > 0]
                if len(phrases) == 0:
                    continue

                if previous == "constant":
                    segment = phrases[0]
                else:
                    segment = phrases[-1]
                current = left

                nodes = tree_to_dict(self.get_constituency(segment))
                target = search(nodes, component[1])
                if target is not None:
                    index = sentence.find(target, left)
                    replace(sequence, index, index + len(target), 1)
                else:
                    index = sentence.find(segment, left)
                    replace(sequence, index, index + len(segment), 1)
                previous = "variable"

        return sequence

    def extract(self, file):
        """
         Extract the feature: words, constants and variables from training data
         :param file: string
         :return: sequences: list[tuple]
         """
        data = self.load_training_data(file)
        components, constants = self.get_components(file)
        with open(self.conf.train_dict, "rb") as fp:
            word_index = pickle.load(fp)

        sequences = []
        for sentence, labels in data:
            indice, sequence = [], [0] * len(sentence)
            text = "".join(sentence)

            # Extract the features of constant and variable
            segments = self.cut_sentence(text, constants)
            if segments.size() > 0:
                sequence = self._extract_components(components, text, segments, sequence)

            # Extract the feature of punctuations
            sequence = self._extract_comma(file, sequence, sentence)

            # Extract the feature of word
            indice = self._extract_word(sentence, word_index)

            sequences.append((sentence, indice, sequence, vectorize_labels(labels, self.conf.labels)))

        fp.close()

        return sequences

    # ======================================
    # Process the test data of Hades
    # ======================================
    def load_test_data(self, file):
        """
        Load the test data
        :param file: string
        :return: test_data: list[list[string]]
        """
        test_data = []

        with open(self.conf.test_data.format(file), "r", encoding="utf-8") as fp:
            lines = fp.readlines()

            for line in lines:
                line = line.strip()

                if len(line) > 0:
                    sentence = []
                    for word in line:
                        sentence.append(word)
                    test_data.append(sentence)

        fp.close()

        return test_data

    def process(self, file):
        """
        Extract the feature: words, constants and variables from test data
        :param file: string
        :return: sequences: list[tuple]
        """
        data = self.load_test_data(file)
        components, constants = self.get_components(file)
        with open(self.conf.train_dict, "rb") as fp:
            word_index = pickle.load(fp)
        fp.close()

        sequences = []
        for sentence in data:
            indice, sequence = [], [0] * len(sentence)
            text = "".join(sentence)

            # Extract the features of constant and variable
            segments = self.cut_sentence(text, constants)
            if segments.size() > 0:
                sequence = self._extract_components(components, text, segments, sequence)

            # Extract the feature of punctuations
            sequence = self._extract_comma(file, sequence, sentence)

            # Extract the feature of word
            indice = self._extract_word(sentence, word_index)

            sequences.append((sentence, indice, sequence))

        return sequences

    # ======================================
    # Process the data of Poseidon
    # ======================================
    def fishing(self, file):
        """
        Process the training data for poseidon
        :param file: string
        :return: sequence: list
        """
        # Load the training data
        data = self.load_training_data(file)

        # Get the construction
        construction = "".join(file.split(".")[0].split("_")[1].split("+"))

        sequences = list()
        for sentence, labels in data:
            sample = get_cxn_sample(sentence, labels)
            sentence = "".join(sentence)
            sequences.append((construction, sentence, sample))
        return sequences

    def surfing(self, file):
        """
        Process the test data for poseidon
        :param file: string
        :return: sequence: list
        """
        # Load the test data
        data = self.load_test_data(file)

        # Get the construction
        construction = "".join(file.split(".")[0].split("_")[1].split("+"))

        sequences = list()
        for record in data:
            sentence = "".join(record)
            sequences.append((construction, sentence))

        return sequences

    # ======================================
    # Deal
    # ======================================
    def on(self, system):
        """
        Process the train data
        :return: sequences: list[tuple]
        """
        sequences = []
        files = os.listdir(self.conf.train_data_dir)
        for file in tqdm(files, desc="Process the training data"):
            if system == "hades":
                sequences += self.extract(file)
            elif system == "poseidon":
                sequences += self.fishing(file)

        return sequences

    def up(self, system):
        """
        Process the test data
        :return: sequences: list[tuple]
        """
        sequences = []
        files = os.listdir(self.conf.test_data_dir)
        for file in tqdm(files, desc="Process the test data"):
            if system == "hades":
                sequences += self.process(file)
            elif system == "poseidon":
                sequences += self.surfing(file)

        return sequences
