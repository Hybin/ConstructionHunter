from hermes import Hermes
from processor import Processor
from features import Features
from thunder import Thunder
from scepter import Scepter
import os


class Zeus(object):
    def __init__(self, conf):
        self.conf = conf
        self.processor = Processor(conf)
        self.hermes = Hermes("Zeus").on()
        self.features = Features()
        self.thunder = Thunder(self.features, self.processor)
        self.scepter = Scepter(conf, self.hermes, self.processor, self.features, self.thunder)

    def test(self):
        files = os.listdir(self.conf.test_data_dir)
        for file in files:
            print(self.scepter.extract(file))
