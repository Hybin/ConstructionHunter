from keras import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint
from constants import *


class Trident(object):
    def __init__(self, conf):
        self.conf = conf

    def load_bert(self):
        bert = load_trained_model_from_checkpoint(
            self.conf.bert["config"],
            self.conf.bert["checkpoint"],
            seq_len=MAX_SEQUENCE_LENGTH)

        for layer in bert.layers:
            layer.trainable = True

        return bert

    def draw(self):
        bert = self.load_bert()

        # Model Configuration
        input_tokens = Input(shape=(None,))
        input_segments = Input(shape=(None,))
        input_head_edge = Input(shape=(None,))
        input_tail_edge = Input(shape=(None,))

        # Set the mask layer
        mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(input_tokens)

        # Add the layers
        embedding = bert([input_tokens, input_segments])
        pred_head = Dense(1, use_bias=False)(embedding)
        pred_head = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([pred_head, mask])
        pred_tail = Dense(1, use_bias=False)(embedding)
        pred_tail = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([pred_tail, mask])

        # Build the model for predictions
        pred_model = Model(inputs=[input_tokens, input_segments], outputs=[pred_head, pred_tail])

        # Build the model for training
        train_model = Model(inputs=[input_tokens, input_segments, input_head_edge, input_tail_edge],
                            outputs=[pred_head, pred_tail])

        # Compute the loss
        loss_head = K.mean(K.categorical_crossentropy(input_head_edge, pred_head, from_logits=True))
        pred_tail -= (1 - K.cumsum(input_head_edge, 1)) * 1e10
        loss_tail = K.mean(K.categorical_crossentropy(input_tail_edge, pred_tail, from_logits=True))
        loss = loss_head + loss_tail

        train_model.add_loss(loss)

        # Build the model
        train_model.compile(optimizer=Adam(LEARNING_RATE))
        train_model.summary()

        return pred_model, train_model
