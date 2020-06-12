from config import Config


class Features(Config):
    conf_file = "../config/features.json"

    def __init__(self):
        super().__init__()
