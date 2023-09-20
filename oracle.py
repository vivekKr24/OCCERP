import random

import numpy as np
from sklearn.svm import OneClassSVM


def side_wrt_hyperplane(point, hyperplane):
    point = np.array(point[0]) - np.array(hyperplane)
    if np.dot(point, hyperplane) == 0:
        return 0
    return abs(np.dot(point, hyperplane)) / (np.dot(point, hyperplane))


def divide_data_set(hyperplane, dataset):
    dataset_positive, dataset_negative = [], []
    for point, label in dataset:
        if side_wrt_hyperplane(point, hyperplane) == -1:
            dataset_negative.append((point, label))
        else:
            dataset_positive.append((point, label))

    # print("Divided dataset into %s - %s" % (len(dataset_positive), len(dataset_negative)))

    return dataset_positive, dataset_negative


def hyperplane_from_midpoint(r, s):
    r = np.array(r)
    s = np.array(s)

    z = r + s
    z = np.divide(z, 2)

    # let plane be sum(ai.xi) + b = 0
    return z


class RandomLinearOracle:
    def __init__(self, dataset, oracle_size=500, classifier=OneClassSVM):
        self.oracle = []
        self.size = oracle_size
        self.classifier = classifier
        self.dataset = dataset

    def random_hyperplane(self):
        dataset = self.dataset
        size = len(dataset)

        r = dataset[random.randint(0, size - 1)][0]
        s = dataset[random.randint(0, size - 1)][0]

        # while (np.array(s) == np.array(r)):
        #     s = dataset[random.randint(0, size - 1)]

        return hyperplane_from_midpoint(r, s)

    def train_classifier(self, dataset_positive, dataset_negative):
        classifier = self.classifier

        c_positive = classifier()
        c_positive.fit([x for x, y in dataset_positive], [y for x, y in dataset_positive])

        c_negative = classifier()
        c_negative.fit([x for x, y in dataset_negative], [y for x, y in dataset_negative])

        return c_positive, c_negative

    def build(self, dataset):
        size = self.size
        oracle = self.oracle
        for i in range(size):
            hyperplane = self.random_hyperplane()
            ds_pos, ds_neg = divide_data_set(hyperplane, dataset)

            c_positive, c_negative = self.train_classifier(ds_pos, ds_neg)

            oracle.append((hyperplane, c_positive, c_negative))

    def predict(self, point):
        votes = 0
        for ensemble in self.oracle:
            hyperplane = ensemble[0]
            c_positive = ensemble[1]
            c_negative = ensemble[2]
            if side_wrt_hyperplane(point, hyperplane) == -1:
                votes += c_negative.predict(point.reshape(1, -1))
            else:
                votes += c_positive.predict(point.reshape(1, -1))

        return -1 if votes < 0 else 1
