"""
thunder.py
---------------
Model configuration for BERT
"""
from keras_bert import load_trained_model_from_checkpoint, Tokenizer


class Thunder(object):
    def __init__(self, conf, features, processor, poseidon, hermes):
        self.conf = conf
        self.features = features
        self.processor = processor
        self.poseidon = poseidon
        self.hermes = hermes

    def cxn_filter(self, sentence, construction):
        """
        Check if the sentence consists of the whole constants of the construction

        :param sentence: string
        :param construction: list[string]
        :return: score: int, construction: list[string]
        """
        constants = self.processor.get_constants(construction)

        score = 0
        for constant in constants:
            index = sentence.find(constant)
            if index == -1:
                score -= 1
            else:
                sentence = sentence[index + len(constant):]

        return score, construction

    def cxn_optimize(self, sentence, construction):
        """
        Find out the best form of the construction

        :param sentence: string
        :param construction: string
        :return: construction: list[string]
        """
        variants = self.features.constructions[construction]["variants"] + [construction]

        candidates = list()
        for variant in variants:
            variant = variant.split("+")
            score, variant = self.cxn_filter(sentence, variant)
            candidates.append((score, variant))

        candidates.sort()

        return candidates[-1][1]

    def load_bert(self):
        """
        Load the bert model

        :return: model, tokenizer
        """
        self.hermes.info("Load the bert...")
        model = load_trained_model_from_checkpoint(self.conf.bert["config"], self.conf.bert["checkpoint"])

        self.hermes.info("Build the tokenizer...")
        tokenizer = self.poseidon.build_tokenizer()

        return model, tokenizer
