from config import Config
from processor import Processor
from hades import Hades
from hermes import Hermes
from utils import *
import argparse
import os

if __name__ == "__main__":
    # Get the path and form of a construction from command-line
    # parser = argparse.ArgumentParser(description="Automatic Recognizer for Chinese sentential construction")
    # parser.add_argument("-f", "--file", help="The path (specifically file name) of the raw material")
    # args = parser.parse_args()

    # Load the logger
    hermes = Hermes("Poseidon").on()

    # Load the configuration
    hermes.info("Load the Config...")
    conf = Config()
    hermes.info("Loading the Config...Finished!")

    # Load the processor
    hermes.info("Loading the Processor...")
    processor = Processor(conf)
    hermes.info("Loading the Processor...Finished!")

    hades = Hades(conf, processor)
    hades.train()
    hades.predict()