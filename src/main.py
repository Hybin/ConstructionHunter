from config import Config
from processor import Processor
from hades import Hades
from poseidon import Poseidon
from zeus import Zeus
from hermes import Hermes
import argparse

if __name__ == "__main__":
    # Command-Line Options
    parser = argparse.ArgumentParser(description="Automatic Recognizer for Chinese sentential construction")
    parser.add_argument("-s", "--system", help="The name of system inside the Automatic Recognizer like zeus, poseidon and hades")
    parser.add_argument("-m", "--mode", help="The mode of the system, please choose 'train' or 'predict'")
    args = parser.parse_args()

    # Create the logger
    hermes = Hermes(args.syste.capitalize()).on()

    # Load the configuration
    hermes.info("Load the Config...")
    conf = Config()
    hermes.info("Load the Config...Finished!")

    if args.system == "hades":
        hades = Hades(conf, hermes)

        if args.mode == "train":
            hades.train()
        elif args.mode == "predict":
            results = hades.predict()
            hades.write(results)
        else:
            hermes.error("ArgError: Please choose 'predict' or 'train'")

    if args.system == "poseidon":
        poseidon = Poseidon(conf, hermes)

        if args.mode == "train":
            poseidon.train()
        elif args.mode == "predict":
            poseidon.predict()
        else:
            hermes.error("ArgError: Please choose 'predict' or 'train'")

    if args.system == "zeus":
        poseidon = Poseidon(conf, hermes)
        zeus = Zeus(conf, poseidon, hermes)

        if args.mode == "predict":
            zeus.predict()
        else:
            hermes.error("ArgError: Predict mode only for zeus")
