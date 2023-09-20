from sklearn.svm import OneClassSVM

from dataset import DatasetProcessor
from oracle import RandomLinearOracle

dp = DatasetProcessor('bbbSpaces.xlsx')
dataset = dp.read_file()

rlo = RandomLinearOracle(dataset)
rlo.build(dataset)

ocsvm = OneClassSVM()
ocsvm.fit([x for x, y in dataset], [y for x, y in dataset])

testdataprocessor = DatasetProcessor('bbbSpaces.xlsx')
test_dataset = testdataprocessor.read_file(label=0)

def evaluate(test_dataset, label):
    correct = 0
    correct_oc_svm = 0
    n = 0
    for i in test_dataset:
        prediction = rlo.predict(i[0])
        ocsvm_prediction = ocsvm.predict(i[0].reshape(1, -1))
        if prediction == label:
            correct += 1
        if ocsvm_prediction == label:
            correct_oc_svm += 1
        n += 1
    # print('=================================')
    # for plane in rlo.oracle:
    #     print(plane)
    # print('=================================')
    print(correct, n)
    print("OCSVM: %s, %s" % (correct_oc_svm, n))


evaluate(test_dataset, -1)
