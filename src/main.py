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

    # Create the logger
    hermes = Hermes("Zeus").on()

    # Load the configuration
    hermes.info("Load the Config...")
    conf = Config()
    hermes.info("Load the Config...Finished!")

    poseidon = Poseidon(conf, hermes)

    zeus = Zeus(conf, poseidon, hermes)
    zeus.test()