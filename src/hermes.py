from constants import *
import logging


class Hermes(object):
    def __init__(self, name):
        self.name = name

    def create_logger(self):
        """
        Create a logger
        :return: logger
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        return logger

    @staticmethod
    def create_formatter():
        """
        Create the formatter
        :return: formatter
        """
        fmt = "[%(asctime)-15s] [%(levelname)s] %(filename)s %(message)s"
        date_fmt = "%a %d %b %Y %H:%M:%S"
        formatter = logging.Formatter(fmt, date_fmt)

        return formatter

    def create_file_handler(self):
        """
        Create the file handler
        :return: handler
        """
        path = log_path.format(self.name)
        handler = logging.FileHandler(path)
        handler.setLevel(logging.INFO)

        formatter = self.create_formatter()
        handler.setFormatter(formatter)

        return handler

    def create_console_handler(self):
        """
        Create the console handler
        :return: handler
        """
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        formatter = self.create_formatter()
        handler.setFormatter(formatter)

        return handler

    def on(self):
        """
        Turn on the logger
        :return: logger
        """
        logger = self.create_logger()
        file_handler, console_handler = self.create_file_handler(), self.create_console_handler()

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
