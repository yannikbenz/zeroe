#  Copyright (c) 2020.
#
#  Yannik Benz

from tensorflow.keras.layers import (Input, Embedding, Lambda, multiply, concatenate, Dense, LSTM, GRU, Bidirectional,
                                     Conv1D, GlobalAveragePooling1D)
from tensorflow.keras import Model
from tensorflow.keras import backend as K


class SimWordModel:
    """

    """

    def __init__(self):
        self.model = None

    def build(self, encoder_type, loss, optimizer, vocab_size, max_word_length):
        # Input shape (x, y)
        input_word_a = Input(shape=(max_word_length,))
        input_word_b = Input(shape=(max_word_length,))
        inputs = [input_word_a, input_word_b]

        # Char Embedding
        embedding = Embedding(input_dim=vocab_size, output_dim=100, input_length=max_word_length, name='embedding')
        word_a_embed = embedding(input_word_a)
        word_b_embed = embedding(input_word_b)

        # Encoding
        word_a_encoded, word_b_encoded = self.word_encoder(word_a_embed, word_b_embed, encoder_type)

        # sub/mul/concat
        sub = Lambda(lambda x: K.abs(x[0] - x[1]))([word_a_encoded, word_b_encoded])
        mul = multiply([word_a_encoded, word_b_encoded])
        concat = concatenate([word_a_encoded, word_b_encoded, sub, mul])

        #
        dense = Dense(units=256, activation='relu', name='concat_layer')(concat)
        output = Dense(4, activation='softmax', name='output_layer')(dense)

        model = Model(inputs, output)
        model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
        return model

    def word_encoder(self, word_a, word_b, encoder_type, embedding_dim=100):
        # Encoded words
        if encoder_type == 'lstm':
            lstm = LSTM(embedding_dim, name='shared-lstm')
            return lstm(word_a), lstm(word_b)
        elif encoder_type == 'gru':
            gru = GRU(name='shared-gru')
            return gru(word_a), gru(word_b)
        elif encoder_type == 'bilstm':
            bilstm = Bidirectional(LSTM(embedding_dim))
            return bilstm(word_a), bilstm(word_b)
        elif encoder_type == 'bigru':
            bigru = Bidirectional(GRU())
            return bigru(word_a), bigru(word_b)
        elif encoder_type == 'bilstm_max_pool':
            bilstm = Bidirectional(LSTM(embedding_dim, return_sequences=True))
            global_max_pooling = Lambda(lambda x: K.max(x, axis=1))  # GlobalMaxPooling1D didn't support masking
            return global_max_pooling(bilstm(word_a)), global_max_pooling(bilstm(word_b))
        elif encoder_type == 'bilstm_mean_pool':
            bilstm = Bidirectional(LSTM(embedding_dim, return_sequences=True))
            return GlobalAveragePooling1D()(bilstm(word_a)), GlobalAveragePooling1D()(bilstm(word_b))
        # elif encoder_type == 'self_attentive':
        #     attention_layers = [SelfAttention() for _ in range(4)]
        #     attend_premise = [attend_layer(word_a) for attend_layer in attention_layers]
        #     attend_hypothesis = [attend_layer(word_b) for attend_layer in attention_layers]
        #     return concatenate(attend_premise), concatenate(attend_hypothesis)
        elif encoder_type == 'h_cnn':
            cnn_word_a, cnn_word_b = [word_a], [word_b]

            filter_lengths = [2, 3, 4, 5]
            for filter_length in filter_lengths:
                conv_layer = Conv1D(filters=25, kernel_size=filter_length, padding='valid',
                                    strides=1, activation='relu')
                cnn_word_a.append(conv_layer(cnn_word_a[-1]))
                cnn_word_b.append(conv_layer(cnn_word_b[-1]))

            global_max_pooling = Lambda(lambda x: K.max(x, axis=1))  # GlobalMaxPooling1D didn't support masking
            cnn_word_a = [global_max_pooling(cnn_word_a[i]) for i in range(1, 5)]
            cnn_word_b = [global_max_pooling(cnn_word_b[i]) for i in range(1, 5)]
            return concatenate(cnn_word_a), concatenate(cnn_word_b)
        else:
            raise ValueError('Encoder Type Not Understood: {}'.format(encoder_type))
