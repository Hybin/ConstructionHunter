from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils import *
import numpy as npy


class Scepter(object):
    def __init__(self, conf, hermes, processor, features, thunder):
        self.conf = conf
        self.processor = processor
        self.features = features
        self.hermes = hermes
        self.thunder = thunder

    def get_cxn_form(self, sentence, construction):
        """
        Get the best form of the construction

        :param sentence: str
        :param construction: str
        :return: construction: list[str]
        """

        return self.thunder.cxn_optimize(sentence, construction)

    @staticmethod
    def const_match_without_exp(sentence, components, constants):
        """
        Constants Matching without block expansion

        :param sentence: str
        :param components: list[list]
        :param constants: list[str]
        :return: checkpoints: Record
        """
        checkpoints = Record()

        previous = 0
        for constant in constants:
            index = sentence.find(constant)
            checkpoints.push(constant, index + previous)
            sentence = sentence[index + len(constant):]
            previous = index + previous + len(constant)

        return checkpoints, components

    @staticmethod
    def const_match_with_exp(sentence, components, constants):
        """
        Constants Matching with block expansion

        :param sentence: str
        :param components: list[list]
        :param constants: list[str]
        :return: checkpoints: Record
        """
        checkpoints, candidates, elements, machine = Record(), list(), list(), automaton(components)

        # initialize the elements
        # push the variables before the first constant
        for component in components:
            component_type, component_value = component

            if component_type == "constant":
                break
            else:
                elements.append(component)

        # constant matching begins
        times, i = 0, 0
        while i < len(sentence):
            if len(candidates) == 0:
                candidates.append(constants[0])

            hash_table = dict()
            for candidate in candidates:
                segment = sentence[i:]
                hash_table[candidate] = i + segment.find(candidate) if candidate in segment else -1

            if hash_table[candidates[0]] != -1:
                last_match = candidates.pop()
                candidates.insert(0, constants[constants.index(last_match) + 1])
                candidates.append(last_match)

            fronter, target = len(sentence), str()
            for candidate, index in hash_table.items():
                fronter, target = (index, candidate) if index != -1 else (fronter, str())

            if fronter != len(sentence) and target != "":
                checkpoints.push(target, fronter)

                if len(elements) > 0:
                    tail_type, tail_value = elements[-1]
                    if tail_type == "constant":
                        elements.append(["variable", machine[target]["prev"]])

                elements.append(["constant", target])
                if machine[target]["next"] != "end":
                    elements.append(["variable", machine[target]["next"]])

                i = fronter + len(target) - 1

            i += 1
        return checkpoints, elements

    def variables_spot(self, construction, sentence, expand=False):
        """
        Spot the text range of variables

        :param construction: list[str]
        :param sentence: str
        :param expand: bool
        :return: var_spots: list[tuple]
        """
        # Get the components of the construction
        components, constants = self.processor.get_components("", variant=construction)

        # Get the checkpoints of the constants
        checkpoints, elements = self.const_match_without_exp(sentence, components, constants) if not expand else self.const_match_with_exp(sentence, components, constants)

        segments, const_ranges = list(), dict()
        for constant, index, visit in checkpoints.container:
            segments.append((index, index + len(constant)))

            if constant not in const_ranges.keys():
                const_ranges[constant] = list()

            const_ranges[constant].append((index, index + len(constant)))

        # Filling the blanks of the segments
        segments = segments_filling(segments, sentence)

        var_spots, index = list(), 0
        for component in elements:
            component_type, component_value = component

            if component_type == "variable":
                var_spots.append((component_value, segments[index]))
            else:
                current = const_ranges[component_value][0]
                index = segments.index(current) + 1
                const_ranges[component_value].pop(0)

        return var_spots

    def var_match_with_cat(self, sentence, segment, cat):
        """
        Match the variables with syntactic category

        :param sentence: str
        :param segment: tuple[int]
        :param cat: str
        :return: start: int, end: int
        """
        left, right = segment
        nodes = tree_to_dict(self.processor.get_constituency(sentence[left:right]))
        target = search(nodes, cat)
        target = sentence[left:right] if target is None else target

        start = sentence.find(target)
        return start, start + len(target)

    @staticmethod
    def var_match_without_cat(sentence, segment):
        """
        Match the variables without syntactic category

        :param sentence: str
        :param segment: tuple[int]
        :return: start: int, end: int
        """
        left, right = segment
        phrases = [phrase for phrase in re.split(r"[，。？！；：]", sentence[left:right]) if len(phrase) > 0]
        target = phrases[0] if right == len(sentence) else (phrases[-1] if left == 0 else sentence[left:right])

        start = sentence.find(target)
        return start, start + len(target)

    def _extract_constants(self, sentence, construction, expand=False):
        """
        Extract the feature of constants

        :param sentence: str
        :param construction: list[str]
        :param expand: bool
        :return: sequence: list[int]
        """
        # Get the components of the construction
        components, constants = self.processor.get_components("", variant=construction)

        checkpoints, elements = self.const_match_without_exp(sentence, components, constants) if not expand else self.const_match_with_exp(sentence, components, constants)

        sequence = [0] * len(sentence)
        for constant, index, visit in checkpoints.container:
            replace(sequence, index, index + len(constant), 1)

        return sequence

    def _extract_variables(self, sentence, construction, expand=False):
        """
        Extract the feature of variables

        :param sentence: str
        :param construction: list[str]
        :param expand: bool
        :return: sequence: list[int]
        """
        # Get the text ranges of variables
        var_spots = self.variables_spot(construction, sentence, expand)

        sequence = [0] * len(sentence)
        for cat, segment in var_spots:
            if len(re.findall(r"[A-Z]", cat)) > 0:
                start, end = self.var_match_with_cat(sentence, segment, cat)
            else:
                start, end = self.var_match_without_cat(sentence, segment)

            replace(sequence, start, end, 1)

        return sequence

    def _extract_similarity(self, sentence, construction, expand, similar=False):
        """Extract the structural similarity of variables inside the construction

        :param sentence: str
        :param construction: str
        :param expand: bool
        :param similar: bool
        :return: sequence: list[int]
        """
        sequence = [0] * len(sentence)

        if similar:
            # Get the text ranges of variables
            var_spots = self.variables_spot(construction, sentence, expand)

            # Found out the minimal range
            min_len = len(sentence)
            for cat, (left, right) in var_spots:
                if right - left < min_len:
                    min_len = right - left

            # Extract the similarity
            for cat, (left, right) in var_spots:
                replace(sequence, left, left + min_len, 1)

        return sequence

    def _extract_modal(self, sentence, construction, expand):
        """
        Check if the modal particles belongs to the variable
        :param sentence: str
        :param construction: str
        :param expand:  bool
        :return: sequence: list[int]
        """
        sequence = [0] * len(sentence)

        # Get the text ranges of variables
        var_spots = self.variables_spot(construction, sentence, expand)

        for cat, (left, right) in var_spots:
            try:
                text, modal = sentence[left:right][0:-1], sentence[left:right][-1]

                nodes = tree_to_dict(self.processor.get_constituency(modal))

                for node in nodes:
                    for key, value in node.items():
                        if (key == "u" or key == "y") and value == modal:
                            words = tree_to_dict(self.processor.get_constituency(text))

                            target = tuple()
                            for word in words:
                                for c, w in word.items():
                                    if text.rfind(w) == len(text) - len(w):
                                        target = (c, w)

                            if modal not in self.conf.modal_particles or target[0] in ["n", "np"]:
                                continue
                            else:
                                replace(sequence, right - 1, right, 1)
            except IndexError:
                print(var_spots, left, right, sentence)
                exit(0)

        return sequence

    @staticmethod
    def _extract_character(sentence, model, tokenizer):
        # Encode the sentences
        indices, segments = tokenizer.encode(first=sentence, max_len=512)

        # Predict the sentence
        embeddings = model.predict([npy.array([indices]), npy.array([segments])])[0]

        # Dimension Reduce
        embeddings = PCA(n_components=1).fit_transform(embeddings)
        embeddings = [embedding[0] for embedding in embeddings[1:len(sentence) + 1]]

        return embeddings

    def extract(self, file):
        """
        Extract the features of the construction

        :param file: str
        :return: sequences: npy.array
        """
        self.hermes.info("Load the data...")
        sentences = ["".join(sentence) for sentence in self.processor.load_test_data(file)]
        self.hermes.info("Load the data...Finished!")

        construction = file.split(".")[0].split("_")[1]
        expand = self.features.constructions[construction]["block_expansion"]
        similarity = self.features.constructions[construction]["structure_similarity"]
        self.hermes.info("Begin to process the construction: " + construction)
        self.hermes.info("Block expansion of construction: " + str(expand))
        self.hermes.info("Structure similarity of construction: " + str(similarity))

        sequences = list()
        self.hermes.info("Extract the features...")
        for sentence in tqdm(sentences, desc="Extracting"):
            sequence = list()

            construction = file.split(".")[0].split("_")[1]
            construction = self.get_cxn_form(sentence, construction)

            feature_constants = self._extract_constants(sentence, construction, expand)
            feature_variables = self._extract_variables(sentence, construction, expand)
            feature_const_sim = self._extract_similarity(sentence, construction, expand, similarity)
            feature_modal_par = self._extract_modal(sentence, construction, expand)

            feature_variables = xor(feature_variables, feature_modal_par)

            sequence.append(feature_constants)
            sequence.append(feature_variables)
            sequence.append(feature_const_sim)

            sequence = npy.array(sequence).T
            sequences.append(sequence)

        return sentences, npy.array(sequences)

    def cluster(self, file):
        """
        Clustering with extracted features

        :param file: str
        :return: samples: list[list[tuple]]
        """
        sentences, sequences = self.extract(file)

        samples = list()
        for sentence, sequence in zip(sentences, sequences):
            # Clustering the samples
            k_means = KMeans(n_clusters=3, random_state=0).fit(sequence)
            # Predict the labels
            labels = k_means.predict(sequence)
            # Store the sample
            sample = [(word, label) for word, label in zip(sentence, labels)]
            samples.append(sample)

        return samples

    def annotate(self, file, samples):
        pass