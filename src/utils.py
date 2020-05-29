from constants import *
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
        vector.append(encoding.index(label) + 1)

    return vector


def merge(series1, series2):
    """
    Merge two sequence
    :param series1: list[int]
    :param series2: list[int]
    :return: series: list[list[int]]
    """
    series = []

    for i in range(0, len(series1)):
        item = [series1[i], series2[i]]
        series.append(item)

    return series
