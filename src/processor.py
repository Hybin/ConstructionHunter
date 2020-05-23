import re
from utility import *


class Processor(object):
    def __init__(self, conf):
        self.conf = conf

    def load_training_data(self, file):
        """ Load the training data
         :param file: string
         :return: training_data: list[tuple]
         """
        with open(self.conf.train_data.format(file), "r") as fp:
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

        return training_data

    @staticmethod
    def get_constants(file):
        """
         Get the constants from the construction form
         :param file: string
         :return: constants: list[string]
         """
        construction = file.split(".")[0].split("_")[1].split("+")

        constants, previous = [], ""
        for component in construction:
            if len(re.search(r'[a-zA-Z0-9]*', component).group()) == 0:
                if previous != "constant":
                    constants.append(component)
                else:
                    constants[-1] = constants[-1] + component
                previous = "constant"
            else:
                previous = "variable"

        return constants

    def process_on_constants(self, file):
        """
         Extract the feature: constants
         :param file: string
         :return: sequences: list[tuple]
         """
        data = self.load_training_data(file)
        constants = self.get_constants(file)

        sequences = []
        for sentence, labels in data:
            sequence = [0] * len(sentence)

            text, next, prev = "".join(sentence), 0, 0
            for constant in constants:
                next = text.find(constant, prev)
                if next != -1:
                    replace(sequence, next, len(constant), 2)
                    prev = next
                    next += len(constant)

            sequences.append((sentence, sequence, labels))

        return sequences

    # TODO: Process on variables
