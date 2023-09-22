import pickle
from dataset import DatasetProcessor
from model import BBBClassifierRunner


def use_model(file_path, test_dataset_):
    saved_model = open(file_path, 'rb')
    bbb: BBBClassifierRunner = pickle.load(saved_model)
    saved_model.close()
    return bbb.accuracy(test_dataset_=test_dataset_, RLO=bbb.RLO)


def new_model(n_features_, train_label_, rlo_size_):
    dp = DatasetProcessor('bbbSpaces.xlsx')
    train_dataset = dp.read_file(label=train_label_, n_features=n_features_)
    test_dataset = dp.read_file(label=0 if train_label_ == 1 else 1, n_features=n_features_)

    saved_model = open('oracle.model', 'rb')
    bbb_saved: BBBClassifierRunner = pickle.load(saved_model)
    best_acc = bbb_saved.best_accuracy
    print("Accuracy of model:", best_acc)
    saved_model.close()

    bbb = BBBClassifierRunner(rlo_size_, train_label_)
    acc_ = bbb.accuracy(test_dataset_=test_dataset, train_dataset_=train_dataset)

    if acc_[1] > best_acc:
        best_acc = acc_[1]
        bbb.save()

    print("Current:", acc_[1])
    print("Best:", best_acc)

    return acc_


def display_results(acc_):
    header = "*********************** %s FEATURES ***********************" % n_features
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    print("     Accuracy - One Class SVM: " + str(acc_[0]) + '%')
    print("     Accuracy - OCCERP: " + str(acc_[1]) + '%')
    print("=" * len(header))
    print("*" * len(header))
    print("=" * len(header))


n_features, train_label, rlo_size = 4, 0, 50

# Use pretrained model on test_dataset
dp = DatasetProcessor('bbbSpaces.xlsx')
test_dataset = dp.read_file(label=0 if train_label == 1 else 1, n_features=n_features)
acc = use_model('oracle.model', test_dataset)

# # Create New Model
# acc = new_model(n_features, train_label, rlo_size)

display_results(acc)
