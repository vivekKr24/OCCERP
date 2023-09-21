from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA


def format_data(data_1, labels):
    formatted_data = []
    for i in range(data_1.shape[0]):
        a = data_1[i]
        b = labels[i]
        formatted_data.append((a, b))

    return formatted_data


class DatasetProcessor:
    def __init__(self, src_path='bbbSpaces.xlsx'):
        self.src_path = src_path

    def read_file(self, label, n_features):
        MAX_SEQUENCE_LENGTH = 100  # to be set after iteration over all peptides
        MAX_NB_WORDS = 21  # Maximum no. of amino acids

        dataset = pd.read_excel(self.src_path, engine='openpyxl')
        if label != 2:
            dataset = dataset[dataset['Property'] == label]
        else:
            dataset = dataset[1:]

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
        # print('Found %s unique tokens' % len(word_index))
        sequences = tokenizer.texts_to_sequences(total_peptides)
        data_1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        # print('Shape of original data tensor:', data_1.shape)

        ################################################################################################################
        data_1 = StandardScaler().fit_transform(data_1)

        pca_data = PCA(n_components=n_features)
        new_features = pca_data.fit_transform(data_1)

        # new_features = pca_data.singular_values_
        # print('Shape of new data tensor:', new_features.shape)
        return format_data(new_features, labels)

