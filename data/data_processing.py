import json
import os
import pandas as pd
import numpy as np
# Create 2 different data structures from input data to approach arg. classification and ID retrieval.


def argument_classification_data():
    train_path = 'data/dry-run-data/train'
    test_path = 'data/dry-run-data/test'

    output = []
    for _, _, files in os.walk(train_path):
        for file in files:
            with open('data/dry-run-data/train/'+file) as f:
                minute_file = json.load(f)
                f.close()

            for session in minute_file:
                for intervention in session['proceeding']:
                    # sentence segmentation of utterances
                    utterance_segment_list = intervention['utterance'].split('\n')
                    # if the intervention contains money expressions
                    if len(intervention['moneyExpressions']) > 0:
                        # look for the segments which contain the money expression
                        for expression in intervention['moneyExpressions']:
                            segments = ''
                            for utterance_segment in utterance_segment_list:
                                # check if the money expression is included in the utterance, and that the money expression is not a subset of a higher value mon. expression
                                if expression['moneyExpression'] in utterance_segment and not utterance_segment[utterance_segment.rfind(expression['moneyExpression'])-1].isdigit():
                                    # build composition of segments where mon. expr. appears
                                    segments = segments + utterance_segment

                            argclass = 'fail'

                            if expression['argumentClass'] == 'Premise : 過去・決定事項':
                                argclass = 0
                            elif expression['argumentClass'] == 'Premise : 未来（現在以降）・見積':
                                argclass = 1
                            elif expression['argumentClass'] == 'Premise : その他（例示・訂正事項など）':
                                argclass = 2
                            elif expression['argumentClass'] == 'Claim : 意見・提案・質問':
                                argclass = 3
                            elif expression['argumentClass'] == 'Claim : その他':
                                argclass = 4
                            elif expression['argumentClass'] == '金額表現ではない':
                                argclass = 5
                            elif expression['argumentClass'] == 'その他':
                                argclass = 6

                            output.append([expression['moneyExpression'], segments, argclass])

    return pd.DataFrame(data= output, columns = ['ID', 'text', 'label'])

def kfold_data(data, k):

    data = data.sample(frac=1)

    fold_chunks = []
    folded_data = []

    # check if 3 or 7 classes are considered for the evaluation of the trained models !!!
    data_c1 = data.loc[data['label'] == 0]
    data_c2 = data.loc[data['label'] == 1]
    data_c3 = data.loc[data['label'] == 2]
    data_c4 = data.loc[data['label'] == 3]
    data_c5 = data.loc[data['label'] == 4]
    data_c6 = data.loc[data['label'] == 5]
    data_c7 = data.loc[data['label'] == 6]

    kfold_c1 = np.array_split(data_c1, k)
    kfold_c2 = np.array_split(data_c2, k)
    kfold_c3 = np.array_split(data_c3, k)
    kfold_c4 = np.array_split(data_c4, k)
    kfold_c5 = np.array_split(data_c5, k)
    kfold_c6 = np.array_split(data_c6, k)
    kfold_c7 = np.array_split(data_c7, k)

    # prepare folds
    for i in range(k):
        fold_list = [kfold_c1[i], kfold_c2[i], kfold_c3[i], kfold_c4[i], kfold_c5[i], kfold_c6[i], kfold_c7[i]]
        fold_chunks.append(pd.concat(fold_list))

    # prepare train-test splits for each fold
    for j in range(k):
        test = fold_chunks[j].sample(frac=1)
        train = pd.concat([x for i,x in enumerate(fold_chunks) if i!=j]).sample(frac=1)
        folded_data.append({'train':train, 'test':test})

    return folded_data


# Rebuild the original json files with the outputs of the algorithm.


