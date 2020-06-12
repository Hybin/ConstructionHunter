class Thunder(object):
    def __init__(self, features, processor):
        self.features = features
        self.processor = processor

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