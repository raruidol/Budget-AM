import numpy as np
from data import data_processing as preproc
from nlp import models as mdl
from sklearn.metrics import f1_score as f1

SEED = 7

np.random.seed(seed=SEED)


if __name__ == "__main__":

    data = preproc.argument_classification_data()

    folds = preproc.kfold_data(data, 10)

    for fold in folds:

        clf = mdl.bert_model(7)

        clf.train_model(fold['train'], acc=f1)

        print(fold['train'])
        print(fold['test'])




