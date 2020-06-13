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

        data = list()
        for file in files:
            samples = self.scepter.cluster(file)
            data.append((file, samples))

        return data

    def test(self):
        files = os.listdir(self.conf.test_data_dir)

        model, tokenizer = self.thunder.load_bert()
        for file in files[:1]:
            sentences = ["".join(sentence) for sentence in self.processor.load_test_data(file)]
            for sentence in sentences:
                self.scepter.extract_character(sentence, model, tokenizer)

    def write_to_xml(self):
        """
        Write the data to .xml file

        :return: None
        """
        data = self.process_test_data()

        for file, samples in tqdm(data, desc="Write to the .xml file"):
            self.scepter.annotate(file, samples)

        return
