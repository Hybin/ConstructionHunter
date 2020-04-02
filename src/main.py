from config import Config
import argparse

if __name__ == "__main__":
    # Load the configuration
    conf = Config()

    # Get the path and form of a construction from command-line
    parser = argparse.ArgumentParser(description="Automatic Recognizer for Chinese sentential construction")
    parser.add_argument("-f", "--file", help="The path (specifically file name) of the raw material")
    args = parser.parse_args()

    # Run the Recognizer
