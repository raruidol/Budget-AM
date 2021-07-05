import numpy as np
from data import data_processing as preproc

SEED = 7

np.random.seed(seed=SEED)


if __name__ == "__main__":

    data = preproc.argument_classification_data()
    print(data['label'].value_counts())

    folds = preproc.kfold_data(data, 10)

    for fold in folds:
        print(fold['train'])
        print(fold['test'])




