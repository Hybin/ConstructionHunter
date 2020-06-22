from constants import *
from keras_bert import Tokenizer
from keras.callbacks import Callback
from tqdm import tqdm
import keras.backend as K
import numpy as npy
import re


# objects
class Stack(object):
    def __init__(self):
        self.container = []

    def push(self, item):
        self.container.append(item)

    def pop(self):
        return self.container.pop()

    def top(self):
        return self.container[-1]


class Queue(object):
    def __init__(self):
        self.container = []

    def push(self, item):
        self.container.append(item)

    def pop(self):
        return self.container.pop(0)

    def top(self):
        return self.container[-1]

    def empty(self):
        if len(self.container) ==  0:
            return True
        else:
            return False


class Record(object):
    def __init__(self):
        self.container = []

    def size(self):
        return len(self.container)

    def push(self, key, value):
        self.container.append((key, value, 0))

    def erase(self, key, value):
        index = -1
        for k, v, a in self.container:
            index += 1
            if k == key and v == value:
                break

        if index != -1:
            return self.container.pop(index)

    def get(self, query, visited=False):
        for key, value, visit in self.container:
            if query == key and visited == visit:
                return value

        return None

    def visit(self, key, value):
        index = -1
        for k, v, a in self.container:
            index += 1
            if k == key and v == value:
                break

        if index != -1:
            self.container[index] = (key, value, 1)


class CustomTokenizer(Tokenizer):
    def _tokenize(self, text):
        characters = list()

        for character in text:
            if character in self._token_dict:
                characters.append(character)
            elif self._is_space(character):
                characters.append("[unused1]")
            else:
                characters.append("[UNK]")

        return characters


class Evaluator(Callback):
    def __init__(self, conf, data, excavate, train_model, pred_model, tokenizer, addition, hermes):
        super().__init__()
        self.accuracy = []
        self.best = 0
        self.passed = 0
        self.conf = conf
        self.data = data
        self.excavate = excavate
        self.train_model = train_model
        self.pred_model = pred_model
        self.tokenizer = tokenizer
        self.addition = addition
        self.hermes = hermes

    def on_batch_begin(self, batch, logs=None):
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.0) / self.params['steps'] * LEARNING_RATE
            K.set_value(self.train_model.optimizer.lr, lr)
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (LEARNING_RATE - MIN_LEARNING_RATE)
            lr += MIN_LEARNING_RATE
            K.set_value(self.train_model.optimizer.lr, lr)
        self.passed += 1

    def evaluate(self):
        correct = 0

        for record in tqdm(self.data, "Predict"):
            sample = self.excavate(record[0], record[1], self.tokenizer, self.pred_model, self.addition)
            if sample == record[2]:
                correct += 1

        return correct / len(self.data)

    def on_epoch_end(self, epoch, logs=None):
        accuracy = self.evaluate()
        self.accuracy.append(accuracy)

        if accuracy > self.best:
            self.best = accuracy
            self.train_model.save_weights(self.conf.model_path.format("poseidon"))

        self.hermes.info("accuracy: {}, best accuracy: {}".format(accuracy, self.best))


# functions
def replace(array, start, end, value):
    """
    Replace the values of a list from start to (start + length)
    :param array: list
    :param start: int
    :param end: int
    :param value: int
    :return: array: list
    """
    for i in range(start, end):
        array[i] = value

    return array


def tree_to_dict(tree):
    """
    Convert the tree to dict

    :param tree: string
    :return: nodes: list[dict]
    """
    stack, node, nodes = Stack(), "", list()
    for character in tree:
        if character == "(":
            continue
        elif character == " " or character == "\n":
            if len(node):
                stack.push(("cat", node))
                node = ""
        elif character == ")":
            if len(node):
                stack.push(("tok", node))
                node = ""

            value = ""
            while stack.top()[0] == "tok":
                value = stack.pop()[1] + value

            key = stack.pop()[1]
            if key in categories.keys():
                nodes.append({categories[key]: value})
            else:
                nodes.append({key.lower(): value})
            stack.push(("tok", value))
        else:
            node += character

    return nodes


def search(array, key):
    """
    Search for dict-item according to the given key

    :param array: list[dict]
    :param key: string
    :return: value: string
    """
    key = re.split(r"\d", key)[0]
    max_len, result = 0, ""
    for item in array:
        if key in item.keys() and len(item[key]) > max_len:
            max_len = len(item[key])
            result = item[key]

    if len(result) > 0:
        return result
    else:
        return None


def vectorize_labels(labels, encoding):
    """
    Convert the labels to vector

    :param labels: list
    :param encoding: list
    :return: vector[list]
    """
    vector = []

    for label in labels:
        vector.append(encoding.index(label))

    return vector


def merge(series1, series2):
    """
    Merge two sequence

    :param series1: list[int]
    :param series2: list[int]
    :return: series: list[list]
    """
    series = []

    for i in range(0, len(series1)):
        item = [series1[i], series2[i]]
        series.append(item)

    return series


def resolve(series):
    """
    Resolve the integrated series

    :param series: series: list[list]
    :return: series1: np.array, series2: np.array
    """
    series1, series2 = [], []
    for item in series:
        series1.append(item[0])
        series2.append(item[1])

    return npy.array(series1), npy.array(series2)


def get_length(array):
    """
    Get the real length of sentence which is padded

    :param array: np.array
    :return: length: int
    """
    lst = [item for item in array]
    sentence = []

    for item in lst:
        if item == 0:
            continue
        else:
            sentence = lst[lst.index(item):]
            break

    return len(sentence)


def get_cxn_sample(sentence, labels):
    """
    Get the sample of construction inside the sentence

    :param sentence: list[str]
    :param labels: list[str]
    :return: sample: str
    """
    sample = str()
    for i in range(len(sentence)):
        if labels[i] != "O":
            sample += sentence[i]

    return sample


def list_search(source, target):
    """
    Find out the position of the target in the source

    :param source: list[str]
    :param target: list[str]
    :return: position: int
    """
    length = len(target)
    for i in range(len(source)):
        if source[i:i+length] == target:
            return i

    return -1


def train_test_split(data):
    """
    Split the data

    :param data: list[Any]
    :return: train_data, test_data
    """
    seed = list(range(len(data)))
    npy.random.shuffle(seed)

    train, test = [], []
    for i, j in enumerate(seed):
        if i % 9 == 0:
            test.append(data[j])
        else:
            train.append(data[j])

    return train, test


def softmax(x):
    """
    Compute the softmax

    :param x: np.array
    :return: float32
    """
    x = x - npy.max(x)
    x = npy.exp(x)

    return x / npy.sum(x)


def additional(data):
    """
    Get the additional characters

    :param data: list[Any]
    :return: addition: Set
    """
    addition = set()

    for record in data:
        addition.update(re.findall(r"[^\u4e00-\u9fa5a-zA-Z0-9、“”，]", record[0]))

    return addition


def sequence_padding(X, padding=0):
    """
    Sequences Padding

    :param X: npy.array
    :param padding: int
    :return: X
    """
    max_len = max([len(x) for x in X])

    return npy.array([
        npy.concatenate([x, [padding] * (max_len - len(x))]) if len(x) < max_len else x for x in X
    ])


def segments_filling(segments, sentence):
    """
    Filling the blanks of segments

    :param segments: list[tuple]
    :param sentence: string
    :return:
    """
    ranges, size = list(), len(segments)

    for i in range(size):
        left, right = segments[i]

        # Blanks at first
        if i == 0 and left != 0:
            ranges.append((0, left))

        # Blanks at middle
        if i > 0:
            tail, head = segments[i - 1][1], segments[i][0]

            if tail != head:
                ranges.append((tail, head))

        ranges.append((left, right))

        # Blanks at last
        if i == size - 1 and right != len(sentence):
            ranges.append((right, len(sentence)))

    return ranges


def automaton(construction):
    """ Convert the construction to automaton

    :param construction: str
    :return: machine: dict
    """
    nodes, machine = dict(), dict()

    for i in range(len(construction)):
        if i not in nodes.keys():
            nodes[i] = dict()

        nodes[i]["value"] = construction[i][1]
        nodes[i]["prev"] = "start" if i == 0 else construction[i - 1][1]
        nodes[i]["next"] = "end" if i == len(construction) - 1 else construction[i + 1][1]

    for node, value in nodes.items():
        if value["value"] not in machine.keys():
            machine[value["value"]] = dict()

        machine[value["value"]]["next"] = value["next"]
        machine[value["value"]]["prev"] = value["prev"]

    return machine


def xor(a, b):
    """ XOR of two list

    :param a: list[int]
    :param b: list[int]
    :return: sequence: list[int]
    """
    sequence = [0] * len(a)

    for i in range(len(a)):
        sequence[i] = a[i] ^ b[i]

    return sequence


def judge(sequence, construction):
    """
    Judge the property of the cluster label

    :param sequence: list[tuple]
    :param construction: str
    :return: labels: dict
    """
    labels = dict()

    candidates = dict()
    for word, label in sequence:
        if word in construction:
            if label not in candidates.keys():
                candidates[label] = 0
            candidates[label] += 1

        if word in ["。", "...", "？", "！"]:
            labels[label] = "O"

    constant, higher = 0, 0
    for label, score in candidates.items():
        higher, constant = (score, label) if score > higher else (higher, constant)

    labels[constant] = "C"

    for label in [0, 1, 2]:
        if label not in labels.keys():
            labels[label] = "V"

    return labels
