import json
import os
import pandas as pd
import numpy as np
# Create 2 different data structures from input data to approach arg. classification and ID retrieval.


def argument_classification_data():
    train_path = 'data/dry-run-data/PoliInfo3_BAM-minutes-training.json'
    test_path = 'data/dry-run-data/PoliInfo3_BAM-minutes-test.json'

    output = []

    with open(train_path) as f:
        minute_file = json.load(f)
        f.close()

    for session in minute_file['local']:
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
                        argclass = 'Premise : 過去・決定事項'
                    elif expression['argumentClass'] == 'Premise : 未来（現在以降）・見積':
                        argclass = 'Premise : 未来（現在以降）・見積'
                    elif expression['argumentClass'] == 'Premise : その他（例示・訂正事項など）':
                        argclass = 'Premise : その他（例示・訂正事項など）'
                    elif expression['argumentClass'] == 'Claim : 意見・提案・質問':
                        argclass = 'Claim : 意見・提案・質問'
                    elif expression['argumentClass'] == 'Claim : その他':
                        argclass = 'Claim : その他'
                    elif expression['argumentClass'] == '金額表現ではない':
                        argclass = '金額表現ではない'
                    elif expression['argumentClass'] == 'その他':
                        argclass = 'その他'

                    output.append([expression['moneyExpression'], segments, argclass])

    for diet in minute_file['diet']:
        for intervention in diet['speechRecord']:
            # sentence segmentation of utterances
            utterance_segment_list = intervention['speech'].split('\r\n')
            # if the intervention contains money expressions
            if len(intervention['moneyExpressions']) > 0:
                # look for the segments which contain the money expression
                for expression in intervention['moneyExpressions']:
                    segments = ''
                    for utterance_segment in utterance_segment_list:
                        # check if the money expression is included in the utterance, and that the money expression is not a subset of a higher value mon. expression
                        if expression['moneyExpression'] in utterance_segment and not utterance_segment[utterance_segment.rfind(expression['moneyExpression']) - 1].isdigit():
                            # build composition of segments where mon. expr. appears
                            segments = segments + utterance_segment

                    argclass = 'fail'

                    if expression['argumentClass'] == 'Premise : 過去・決定事項':
                        argclass = 'Premise : 過去・決定事項'
                    elif expression['argumentClass'] == 'Premise : 未来（現在以降）・見積':
                        argclass = 'Premise : 未来（現在以降）・見積'
                    elif expression['argumentClass'] == 'Premise : その他（例示・訂正事項など）':
                        argclass = 'Premise : その他（例示・訂正事項など）'
                    elif expression['argumentClass'] == 'Claim : 意見・提案・質問':
                        argclass = 'Claim : 意見・提案・質問'
                    elif expression['argumentClass'] == 'Claim : その他':
                        argclass = 'Claim : その他'
                    elif expression['argumentClass'] == '金額表現ではない':
                        argclass = '金額表現ではない'
                    elif expression['argumentClass'] == 'その他':
                        argclass = 'その他'

                    output.append([expression['moneyExpression'], segments, argclass])

    return pd.DataFrame(data=output, columns=['ID', 'text', 'label'])

def kfold_data(data, k):

    data = data.sample(frac=1)

    fold_chunks = []
    folded_data = []

    data_c1 = data.loc[data['label'] == 0]
    # data_c1['label'] = pd.Series([[1, 0, 0, 0, 0, 0, 0]]*len(data_c1.index), index=data_c1.index)

    data_c2 = data.loc[data['label'] == 1]
    # data_c2['label'] = pd.Series([[0, 1, 0, 0, 0, 0, 0]] * len(data_c2.index), index=data_c2.index)

    data_c3 = data.loc[data['label'] == 2]
    # data_c3['label'] = pd.Series([[0, 0, 1, 0, 0, 0, 0]] * len(data_c3.index), index=data_c3.index)

    data_c4 = data.loc[data['label'] == 3]
    # data_c4['label'] = pd.Series([[0, 0, 0, 1, 0, 0, 0]] * len(data_c4.index), index=data_c4.index)

    data_c5 = data.loc[data['label'] == 4]
    # data_c5['label'] = pd.Series([[0, 0, 0, 0, 1, 0, 0]] * len(data_c5.index), index=data_c5.index)

    data_c6 = data.loc[data['label'] == 5]
    # data_c6['label'] = pd.Series([[0, 0, 0, 0, 0, 1, 0]] * len(data_c6.index), index=data_c6.index)

    data_c7 = data.loc[data['label'] == 6]
    # data_c7['label'] = pd.Series([[0, 0, 0, 0, 0, 0, 1]] * len(data_c7.index), index=data_c7.index)

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
        train = pd.concat([x for i, x in enumerate(fold_chunks) if i != j]).sample(frac=1)
        folded_data.append({'train': train, 'test': test})

    return folded_data


def cascade_data(data):
    label_list = ['Premise : 過去・決定事項', 'Premise : 未来（現在以降）・見積', 'Premise : その他（例示・訂正事項など）', 'Claim : 意見・提案・質問',
                  'Claim : その他', '金額表現ではない', 'その他']

    # premises
    data_pr1 = data.loc[data['label'] == 'Premise : 過去・決定事項']
    data_pr2 = data.loc[data['label'] == 'Premise : 未来（現在以降）・見積']
    data_pr3 = data.loc[data['label'] == 'Premise : その他（例示・訂正事項など）']

    # claims
    data_clm1 = data.loc[data['label'] == 'Claim : 意見・提案・質問']
    data_clm2 = data.loc[data['label'] == 'Claim : その他']

    # non-argumentative
    data_na1 = data.loc[data['label'] == '金額表現ではない']
    data_na2 = data.loc[data['label'] == 'その他']

    premise_data = pd.concat([data_pr1, data_pr2, data_pr3]).sample(frac=1)
    claim_data = pd.concat([data_clm1, data_clm2]).sample(frac=1)
    non_arg_data = pd.concat([data_na1, data_na2]).sample(frac=1)

    data_pr1['label'] = 'Premise'
    data_pr2['label'] = 'Premise'
    data_pr3['label'] = 'Premise'

    data_clm1['label'] = 'Claim'
    data_clm2['label'] = 'Claim'

    premise_claim_data = pd.concat([data_pr1, data_pr2, data_pr3, data_clm1, data_clm2]).sample(frac=1)

    data_pr1['label'] = 'Arg'
    data_pr2['label'] = 'Arg'
    data_pr3['label'] = 'Arg'

    data_clm1['label'] = 'Arg'
    data_clm2['label'] = 'Arg'

    data_na1['label'] = 'None'
    data_na2['label'] = 'None'

    arg_notarg_data = pd.concat([data_pr1, data_pr2, data_pr3, data_clm1, data_clm2, data_na1, data_na2]).sample(frac=1)

    return premise_data, claim_data, non_arg_data, premise_claim_data, arg_notarg_data