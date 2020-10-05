"""
config.py
---------------
Transform the configuration with *.json format into Dict
"""
import json


class Config(object):
    conf_file = "../config/config.json"

    def __init__(self):
        with open(self.conf_file, encoding="utf-8") as cf:
            conf_dict = json.load(cf)

        for key, value in conf_dict.items():
            setattr(self, key, value)
