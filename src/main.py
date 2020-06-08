from config import Config
from processor import Processor
from hades import Hades
from poseidon import Poseidon
from hermes import Hermes
import argparse

if __name__ == "__main__":
    # Command-Line Options
    # parser = argparse.ArgumentParser(description="Automatic Recognizer for Chinese sentential construction")
    # parser.add_argument("-s", "--system", help="The name of system inside the Automatic Recognizer like zeus, poseidon and hades")
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

    poseidon = Poseidon(conf, processor)
    poseidon.predict()