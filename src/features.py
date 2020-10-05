"""
features.py
---------------
Transform the features with *.json format into Dict"""
from config import Config


class Features(Config):
    conf_file = "../config/features.json"

    def __init__(self):
        super().__init__()
