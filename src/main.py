from config import Config
from processor import Processor
import argparse
import os

if __name__ == "__main__":
    # Load the configuration
    conf = Config()

    # Get the path and form of a construction from command-line
    # parser = argparse.ArgumentParser(description="Automatic Recognizer for Chinese sentential construction")
    # parser.add_argument("-f", "--file", help="The path (specifically file name) of the raw material")
    # args = parser.parse_args()

    processor = Processor(conf)

    files = os.listdir("../data/train")
    for file in files[:1]:
        print(processor.process_on_constants(file))