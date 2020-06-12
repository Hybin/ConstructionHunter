from config import Config
from processor import Processor
from hades import Hades
from poseidon import Poseidon
from zeus import Zeus
from hermes import Hermes
import argparse

if __name__ == "__main__":
    # Command-Line Options
    # parser = argparse.ArgumentParser(description="Automatic Recognizer for Chinese sentential construction")
    # parser.add_argument("-s", "--system", help="The name of system inside the Automatic Recognizer like zeus, poseidon and hades")
    # args = parser.parse_args()

    # Load the logger


    # Load the configuration

    conf = Config()


    zeus = Zeus(conf)
    zeus.test()