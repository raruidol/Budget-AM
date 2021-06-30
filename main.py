import os
from data import data_processing as preproc

train_path = 'data/dry-run-data/train'

if __name__ == "__main__":

    input_data = []

    for _, _, files in os.walk(train_path):
        for file in files:
            input_data.append(preproc.argument_classification_data('data/dry-run-data/train/'+file))

    print(input_data)
