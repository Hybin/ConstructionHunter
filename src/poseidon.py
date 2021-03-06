from trident import Trident
from dolphin import Dolphin
from processor import Processor
from tqdm import tqdm
from utils import *
import re
import datetime


class Poseidon(object):
    def __init__(self, conf, hermes):
        self.conf = conf
        self.processor = Processor(conf)
        self.trident = Trident(conf)
        self.hermes = hermes

    def build_tokenizer(self):
        """
        Build the tokenizer
        :return: tokenizer
        """
        token_dict = dict()

        with open(self.conf.bert["dict"], "r", encoding="utf-8") as reader:
            lines = reader.readlines()

            for line in tqdm(lines, desc="Build the tokenizer"):
                token = line.strip()
                token_dict[token] = len(token_dict.keys())
        reader.close()

        tokenizer = CustomTokenizer(token_dict)

        return tokenizer

    @staticmethod
    def excavate(sentence, construction, tokenizer, model, addition):
        # Process the input text
        input_text = "___{}___{}".format(construction, sentence)
        input_text = input_text[:MAX_SEQUENCE_LENGTH]
        tokens = tokenizer.tokenize(input_text)
        input_tokens, input_segments = tokenizer.encode(first=input_text)
        input_tokens, input_segments = npy.array([input_tokens]), npy.array([input_segments])

        # Predict
        pred_head_edge, pred_tail_edge = model.predict([input_tokens, input_segments])
        pred_head_edge, pred_tail_edge = softmax(pred_head_edge[0]), softmax(pred_tail_edge[0])

        for index, token in enumerate(tokens):
            if len(token) == 1 and re.findall(r"[^\u4e00-\u9fa5a-zA-Z0-9、“”，]", token) and token not in addition:
                pred_head_edge[index] -= 10
        head = npy.argmax(pred_head_edge)

        for tail in range(head, len(tokens)):
            token = tokens[tail]
            if len(token) == 1 and re.findall(r"[^\u4e00-\u9fa5a-zA-Z0-9、“”，]", token) and token not in addition:
                break

        tail = npy.argmax(pred_tail_edge[head:tail+1]) + head
        sample = input_text[head-1:tail]
        return sample

    def train(self):
        """
        Process the train data
        :return:
        """
        # Load the train data
        data = self.processor.on(system="poseidon")

        # Activate the additional characters
        addition = additional(data)

        # Split the train data and dev data
        train, dev = train_test_split(data)

        # Load the tokenizer
        tokenizer = self.build_tokenizer()

        # Build the train data
        train_data = Dolphin(train, tokenizer)

        # Build the train model
        pred_model, train_model = self.trident.draw()

        # Load the evaluator
        evaluator = Evaluator(self.conf, dev, self.excavate, train_model, pred_model, tokenizer, addition, self.hermes)

        # Train
        train_model.fit_generator(train_data.__iter__(),
                                  steps_per_epoch=len(train_data),
                                  epochs=10,
                                  callbacks=[evaluator])

    def predict(self):
        # Load the tokenizer
        tokenizer = self.build_tokenizer()

        # Build the train model
        pred_model, train_model = self.trident.draw()

        # Load the model
        self.hermes.info("Load the model...")
        pred_model.load_weights(self.conf.model_path.format("poseidon"))
        self.hermes.info("Load the model...finished!")

        # Load the test data
        sequences, belonging = self.processor.up(system="poseidon")

        # Activate the additional characters
        addition = additional(sequences)

        data = {}
        for construction, sentence in tqdm(sequences, desc="Predicting"):
            sample = self.excavate(sentence, construction, tokenizer, pred_model, addition)
            if construction not in data.keys():
                data[construction] = []

            data[construction].append(mining(sample, sentence, construction))

        self.hermes.info("Begin to write the annotation...Finished!")
        today = datetime.date.today().__format__('%Y_%m_%d')
        for construction, sentences in tqdm(data.items(), desc="Output the predictions"):
            with open(self.conf.output.format("poseidon", today, construction), "w") as fp:
                fp.write("<?xml version='1.0' encoding='UTF-8'?>" + "\r\n")
                fp.write("<document>" + "\r\n")

                for sentence in sentences:
                    fp.write("\t" + sentence + "\r\n")

                fp.write("</document>")
            fp.close()

        self.hermes.info("Begin to write the annotation...Finished!")
