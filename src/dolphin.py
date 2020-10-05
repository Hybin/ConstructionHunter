"""
dolphin.py
----------------
Preprocessing for BERT
"""
from keras.preprocessing.sequence import pad_sequences
from math import ceil
from constants import MAX_SEQUENCE_LENGTH
from utils import list_search, sequence_padding
from tqdm import tqdm
import numpy as npy


class Dolphin(object):
    def __init__(self, data, tokenizer, batch_size=32):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.steps = ceil(len(self.data) / self.batch_size)

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            indice = list(range(len(self.data)))
            npy.random.shuffle(indice)

            input_tokens, input_segments, input_head_edges, input_tail_edges = [], [], [], []
            for index in tqdm(indice, desc="Training"):
                record = self.data[index]

                # Process the input data
                construction, sentence = record[0], record[1][:MAX_SEQUENCE_LENGTH]
                input_text = "___{}___{}".format(construction, sentence)
                tokens_of_sentence = self.tokenizer.tokenize(input_text)

                # Process the output data
                sample = record[2]
                tokens_of_sample = self.tokenizer.tokenize(sample)[1:-1]

                len_of_sentence = len(tokens_of_sentence)
                input_head_edge, input_tail_edge = npy.zeros(len_of_sentence), npy.zeros(len_of_sentence)

                head = list_search(tokens_of_sentence, tokens_of_sample)
                if head != -1:
                    tail = head + len(tokens_of_sample) - 1
                    input_head_edge[head] = 1
                    input_tail_edge[tail] = 1
                    input_token, input_segment = self.tokenizer.encode(first=input_text)

                    input_tokens.append(input_token)
                    input_segments.append(input_segment)
                    input_head_edges.append(input_head_edge)
                    input_tail_edges.append(input_tail_edge)

                    if len(input_tokens) == self.batch_size or index == indice[-1]:
                        input_tokens = pad_sequences(input_tokens, MAX_SEQUENCE_LENGTH)
                        input_segments = pad_sequences(input_segments, MAX_SEQUENCE_LENGTH)
                        input_head_edges = pad_sequences(input_head_edges, MAX_SEQUENCE_LENGTH)
                        input_tail_edges = pad_sequences(input_tail_edges, MAX_SEQUENCE_LENGTH)

                        yield [input_tokens, input_segments, input_head_edges, input_tail_edges], None
                        input_tokens, input_segments, input_head_edges, input_tail_edges = [], [], [], []
