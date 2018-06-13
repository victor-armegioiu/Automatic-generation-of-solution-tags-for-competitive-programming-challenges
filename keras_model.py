from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, Dropout

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split
import numpy as np

from custom_metrics import fbeta_score

import pandas as pd
np.random.seed(7)


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def BidLstm(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(embedding_matrix.shape[0],
                embedding_matrix.shape[1],
                weights=[embedding_matrix],
                mask_zero=True,
                trainable=True)(inp)

    x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.25,
                           recurrent_dropout=0.25))(x)
    x = Attention(maxlen)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(5, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model


def get_embedding_matrix(tokenizer, embedding_size=300):
    embeddings_index = {}
    glove_data = 'glove.6B.300d.txt'

    f = open(glove_data)

    for line in f:
        values = line.split()
        word = values[0]
        value = np.asarray(values[1:], dtype='float')
        embeddings_index[word] = value

    f.close()

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_size))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector[:embedding_size]


    return embedding_matrix


if __name__ == "__main__":
    list_classes = ['dp', 'greedy', 'graphs', 'math', 'implementation']


    data = pd.read_csv('codeforces_problems_csv/data.csv')
    X_data = data['problem_text']
    y_data = []

    unprocessed = data['tags']
    for i in range (len(unprocessed)):
        unprocessed[i] = list(map(int, unprocessed[i][1:-1].split(', ')))
        y_data.append(unprocessed[i])

    X = np.array(X_data)
    y = np.array(y_data)


    Y = y_data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)

    X = tokenizer.texts_to_sequences(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, 
                                        test_size=0.23, random_state=42)


    print('Pad sequences...')
    x_train = sequence.pad_sequences(x_train, maxlen=300)
    x_test = sequence.pad_sequences(x_test, maxlen=300)

    embedding_matrix = get_embedding_matrix(tokenizer=tokenizer)

    model = BidLstm(maxlen=300, max_features=18558, embed_size=300, embedding_matrix=embedding_matrix)
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy', fbeta_score])

    print('Train...')
    model.fit(np.array(x_train), np.array(y_train), 
             batch_size=32,
             epochs=4,
             validation_data=(np.array(x_test), np.array(y_test)))
   