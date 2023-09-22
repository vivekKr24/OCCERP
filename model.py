import pickle
from sklearn.svm import OneClassSVM

from dataset import DatasetProcessor
from oracle import RandomLinearOracle


class BBBClassifierRunner:
    def __init__(self, _rlo_size, _train_label):
        self.RLO_SIZE = _rlo_size
        self.TRAIN_LABEL = _train_label
        self.best_accuracy = 0
        self.prev_accuracy = 0
        self.RLO = None
        saved_model_file = open('oracle.model', 'rb')
        saved_model = pickle.load(saved_model_file)

        self.best_accuracy = saved_model.best_accuracy
        saved_model_file.close()

    def label(self, x):
        return -1 if x != self.TRAIN_LABEL else 1

    def accuracy(self, test_dataset_, train_dataset_=None, RLO: RandomLinearOracle=None):
        ocsvm = None
        if train_dataset_ is None:
            rlo = RLO
        else:
            rlo = RandomLinearOracle(train_dataset_, self.RLO_SIZE)
            rlo.build()
            ocsvm = OneClassSVM()
            ocsvm.fit([x for x, y in train_dataset_], [y for x, y in train_dataset_])

        r = self.evaluate(test_dataset_, rlo, ocsvm)
        self.prev_accuracy = r[1]

        if self.prev_accuracy > self.best_accuracy:
            self.RLO = rlo
            self.best_accuracy = self.prev_accuracy

        return r

    def evaluate(self, test_dataset_, rlo, ocsvm):
        correct = 0
        correct_oc_svm = 0
        n = 0
        for i in test_dataset_:
            prediction = rlo.predict(i[0])
            if ocsvm is not None:
                ocsvm_prediction = ocsvm.predict(i[0].reshape(1, -1))
                if ocsvm_prediction == self.label(i[1]):
                    correct_oc_svm += 1
            if prediction == self.label(i[1]):
                correct += 1

            n += 1

        return correct_oc_svm / n * 100, correct / n * 100

    def save(self):
        self.best_accuracy = self.prev_accuracy
        saved_model_file = open('oracle.model', 'wb')
        saved_model_file.seek(0)
        saved_model_file.truncate()
        pickle.dump(self, saved_model_file)
        saved_model_file.close()
        print("SAVED MODEL")
