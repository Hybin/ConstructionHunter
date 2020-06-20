from hermes import Hermes
from processor import Processor
from features import Features
from thunder import Thunder
from scepter import Scepter
from tqdm import tqdm
import os


class Zeus(object):
    def __init__(self, conf, poseidon, hermes):
        self.conf = conf
        self.poseidon = poseidon
        self.processor = Processor(conf)
        self.hermes = hermes
        self.features = Features()
        self.thunder = Thunder(self.conf, self.features, self.processor, self.poseidon, self.hermes)
        self.scepter = Scepter(conf, self.hermes, self.processor, self.features, self.thunder)

    def process_test_data(self):
        """
        Process the test data

        :return: data: list[tuple]
        """
        self.hermes.info("Load the test files...")
        files = os.listdir(self.conf.test_data_dir)

        # self.hermes.info("Load the Bert...")
        # model, tokenizer = self.thunder.load_bert()

        data = list()
        for file in files:
            samples = self.scepter.cluster(file)
            data.append((file, samples))

        return data

    def predict(self):
        data = self.process_test_data()

        for file, samples in data:
            for sample in samples:
                for word, label in sample:
                    print(word, label)

    def write_to_xml(self):
        """
        Write the data to .xml file

        :return: None
        """
        data = self.process_test_data()

        for file, samples in tqdm(data, desc="Write to the .xml file"):
            self.scepter.annotate(file, samples)

        return
