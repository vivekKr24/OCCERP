from sklearn.svm import OneClassSVM

from dataset import DatasetProcessor
from oracle import RandomLinearOracle


def evaluate(test_dataset_, rlo, ocsvm):
    correct = 0
    correct_oc_svm = 0
    n = 0
    for i in test_dataset_:
        prediction = rlo.predict(i[0])
        ocsvm_prediction = ocsvm.predict(i[0].reshape(1, -1))

        def label(x):
            return -1 if x != train_label else 1

        if prediction == label(i[1]):
            correct += 1
        if ocsvm_prediction == label(i[1]):
            correct_oc_svm += 1
        n += 1
    # print('=================================')
    # for plane in rlo.oracle:
    #     print(plane)
    # print('=================================')
    # print("OCSVM: %s, %s, %s" % (correct_oc_svm, n, correct_oc_svm / n * 100) + '%')
    # print("OCCERP: %s, %s, %s" % (correct, n, correct / n * 100) + '%')
    return correct_oc_svm / n * 100, correct / n * 100


class BBBClassifierRunner:
    def __init__(self, _rlo_size):
        self.RLO_SIZE = _rlo_size

    def accuracy(self, train_dataset_, test_dataset_):
        rlo = RandomLinearOracle(train_dataset_, self.RLO_SIZE)
        rlo.build()

        ocsvm = OneClassSVM()
        ocsvm.fit([x for x, y in train_dataset_], [y for x, y in train_dataset_])

        return evaluate(test_dataset_, rlo, ocsvm)


n_features, train_label, rlo_size = 3, 0, 100

dp = DatasetProcessor('bbbSpaces.xlsx')
train_dataset = dp.read_file(label=train_label, n_features=n_features)
test_dataset = dp.read_file(label=0 if train_label == 1 else 1, n_features=n_features)
complete_dataset = dp.read_file(label=2, n_features=n_features)

bbb = BBBClassifierRunner(rlo_size)
acc = bbb.accuracy(train_dataset, test_dataset)

l = "*********************** %s FEATURES ***********************" % n_features
print("=" * len(l))
print(l)
print("=" * len(l))
print("     Accuracy - One Class SVM: " + str(acc[0]) + '%')
print("     Accuracy - OCCERP: " + str(acc[1]) + '%')
print("=" * len(l))
print("*" * len(l))
print("=" * len(l))
