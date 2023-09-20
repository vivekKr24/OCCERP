from keras import Input, Model
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd


class DatasetProcessor:
    def __init__(self, src_path='bbbSpaces.xlsx'):
        self.src_path = src_path

    def read_file(self, label=1):
        MAX_SEQUENCE_LENGTH = 100  # to be set after iteration over all peptides
        MAX_NB_WORDS = 21  # Maximum no. of amino acids

        dataset = pd.read_excel(self.src_path, engine='openpyxl')
        dataset = dataset[dataset['Property'] == label]
        total_peptides = dataset['Sequence'].tolist()
        labels = dataset['Property'].tolist()
        for i in range(len(total_peptides)):
            pept = total_peptides[i]
            spaced_pept = ""
            pept.lstrip()
            pept.rstrip()
            for ch in pept:
                spaced_pept += ch + " "
            total_peptides[i] = spaced_pept[:-1]

        tokenizer = Tokenizer(MAX_NB_WORDS)
        tokenizer.fit_on_texts(total_peptides)
        word_index = tokenizer.word_index  # the dict values start from 1 so this is fine with zeropadding
        print('Found %s unique tokens' % len(word_index))
        sequences = tokenizer.texts_to_sequences(total_peptides)
        data_1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        print('Shape of data tensor:', data_1.shape)

        formatted_data = []
        for i in range(data_1.shape[0]):
            a = data_1[i]
            b = labels[i]
            formatted_data.append((a, b))

        return formatted_data

        X_train = data_1
        X_Test = data_1
        input_dim = X_train.shape[1]
        print(" Input dimension : ", input_dim)
        encoding_dim = 64
        nb_epoch = 100
        batch_size = 32
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
        decoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
        decoder = Dense(input_dim, activation="relu")(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.summary()
        autoencoder.compile(optimizer="adam",
                            loss="mean_squared_error",
                            metrics=["accuracy"])
        autoencoder.fit(X_train, X_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(X_Test, X_Test),
                        verbose=1)
