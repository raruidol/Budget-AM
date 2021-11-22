import json
import math
import pandas as pd
import numpy as np
import re
import fugashi
# Create 2 different data structures from input data to approach arg. classification and ID retrieval.


def argument_classification_data(n_classes):
    train_path = 'data/dry-run-data/PoliInfo3_BAM-minutes-training.json'

    output = []

    with open(train_path) as f:
        minute_file = json.load(f)
        f.close()

    for session in minute_file['local']:
        for intervention in session['proceeding']:
            # sentence segmentation of utterances
            utterance_segment_list = re.split('\n|。', intervention['utterance'])
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

                    try:
                        # classes: Premise : 過去・決定事項 ; Premise : 未来（現在以降）・見積 ; Premise : その他（例示・訂正事項など）;
                        # Claim : 意見・提案・質問 ; Claim : その他 ; 金額表現ではない ; その他
                        argclass = expression['argumentClass']

                    except:
                        argclass = 'none'

                    if n_classes == 5:
                        if argclass != '金額表現ではない' and argclass != 'その他':
                            output.append([expression['moneyExpression'], segments, argclass])
                    else:
                        output.append([expression['moneyExpression'], segments, argclass])

    for diet in minute_file['diet']:
        for intervention in diet['speechRecord']:
            # sentence segmentation of utterances
            utterance_segment_list = re.split('\r\n|。', intervention['speech'])
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

                    try:
                        # classes: Premise : 過去・決定事項 ; Premise : 未来（現在以降）・見積 ; Premise : その他（例示・訂正事項など）;
                        # Claim : 意見・提案・質問 ; Claim : その他 ; 金額表現ではない ; その他
                        argclass = expression['argumentClass']

                    except:
                        argclass = 'none'

                    if n_classes == 5:
                        if argclass != '金額表現ではない' and argclass != 'その他':
                            output.append([expression['moneyExpression'], segments, argclass])
                    else:
                        output.append([expression['moneyExpression'], segments, argclass])

    return pd.DataFrame(data=output, columns=['ID', 'text', 'labels']).sample(frac=1)


def kfold_data(data, k, preclm):

    data = data.sample(frac=1)

    fold_chunks = []
    folded_data = []

    if preclm:
        data_premise = data.loc[data['labels'] == 'Premise'].sample(frac=1)
        data_claim = data.loc[data['labels'] == 'Claim'].sample(frac=1)

        kfold_premise = np.array_split(data_premise, k)
        kfold_claim = np.array_split(data_claim, k)

        # prepare folds
        for i in range(k):
            fold_list = [kfold_premise[i], kfold_claim[i]]
            fold_chunks.append(pd.concat(fold_list))

    else:
        data_c1 = data.loc[data['labels'] == 'Premise : 過去・決定事項'].sample(frac=1)
        data_c2 = data.loc[data['labels'] == 'Premise : 未来（現在以降）・見積'].sample(frac=1)
        data_c3 = data.loc[data['labels'] == 'Premise : その他（例示・訂正事項など）'].sample(frac=1)
        data_c4 = data.loc[data['labels'] == 'Claim : 意見・提案・質問'].sample(frac=1)
        data_c5 = data.loc[data['labels'] == 'Claim : その他'].sample(frac=1)
        data_c6 = data.loc[data['labels'] == '金額表現ではない'].sample(frac=1)
        data_c7 = data.loc[data['labels'] == 'その他'].sample(frac=1)

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
    data_pr1 = data.loc[data['labels'] == 'Premise : 過去・決定事項']
    data_pr2 = data.loc[data['labels'] == 'Premise : 未来（現在以降）・見積']
    data_pr3 = data.loc[data['labels'] == 'Premise : その他（例示・訂正事項など）']

    # claims
    data_clm1 = data.loc[data['labels'] == 'Claim : 意見・提案・質問']
    data_clm2 = data.loc[data['labels'] == 'Claim : その他']

    # other
    data_o1 = data.loc[data['labels'] == '金額表現ではない']
    data_o2 = data.loc[data['labels'] == 'その他']

    premise_data = pd.concat([data_pr1, data_pr2, data_pr3]).sample(frac=1)
    claim_data = pd.concat([data_clm1, data_clm2]).sample(frac=1)

    balanced_data_allclasses = pd.concat(
        [data_pr1.head(len(data_clm1)), data_pr2.head(len(data_clm1)), data_pr3.head(len(data_clm2)), data_clm1,
         data_clm2, data_o1, data_o2]).sample(frac=1)

    balanced_data_5classes = pd.concat(
        [data_pr1.head(len(data_clm1)), data_pr2.head(len(data_clm1)), data_pr3.head(len(data_clm2)), data_clm1,
         data_clm2]).sample(frac=1)

    data_pr1['labels'] = 'Premise'
    data_pr2['labels'] = 'Premise'
    data_pr3['labels'] = 'Premise'

    data_clm1['labels'] = 'Claim'
    data_clm2['labels'] = 'Claim'

    premise_claim_data = pd.concat([data_pr1, data_pr2, data_pr3, data_clm1, data_clm2]).sample(frac=1)

    c_d = pd.concat([data_clm1, data_clm2]).sample(frac=1)

    n_samples = math.ceil(1.5 * len(c_d.index))

    pr_d = pd.concat([data_pr1.head(math.ceil(n_samples/2)), data_pr2.head(math.ceil(n_samples/4)), data_pr3.head(math.floor(n_samples/4))]).sample(frac=1)

    balanced_data = pd.concat([pr_d, c_d]).sample(frac=1)

    return premise_data, claim_data, premise_claim_data, balanced_data, balanced_data_allclasses, balanced_data_5classes


def related_id_data():
    train_path = 'data/dry-run-data/PoliInfo3_BAM-minutes-training.json'
    budget_path = 'data/dry-run-data/PoliInfo3_BAM-budget.json'

    tagger = fugashi.Tagger()
    output = []

    with open(train_path) as f:
        minute_file = json.load(f)
        f.close()

    with open(budget_path) as g:
        budget_file = json.load(g)
        g.close()

    for session in minute_file['local']:
        for intervention in session['proceeding']:
            utterance_segment_list = re.split('\n|。', intervention['utterance'])
            if len(intervention['moneyExpressions']) > 0:
                for expression in intervention['moneyExpressions']:
                    segments = ''
                    for utterance_segment in utterance_segment_list:
                        if expression['moneyExpression'] in utterance_segment and not utterance_segment[utterance_segment.rfind(expression['moneyExpression'])-1].isdigit():
                            segments = segments + utterance_segment

                    if expression['relatedID'] is None:
                        randBudget = budget_file['local'][session['localGovernmentCode']][np.random.randint(len(budget_file['local'][session['localGovernmentCode']]))]
                        if randBudget['description'] is not None:
                            budget_text = randBudget['budgetItem'] + ' ' + randBudget['description']
                        else:
                            budget_text = randBudget['budgetItem']
                        output.append([segments, budget_text, 0])

                    else:
                        for budget in budget_file['local'][session['localGovernmentCode']]:

                            if budget['description'] is not None:
                                budget_text = budget['budgetItem'] + ' ' + budget['description']
                            else:
                                budget_text = budget['budgetItem']

                            if expression['relatedID'] is not None and budget['budgetId'] in expression['relatedID']:
                                output.append([segments, budget_text, 1])
                                false_samples = 0
                                while false_samples < 2:

                                    randSession = minute_file['local'][np.random.randint(len(minute_file['local']))]
                                    randIntervention = randSession['proceeding'][np.random.randint(len(randSession['proceeding']))]
                                    if len(randIntervention['moneyExpressions']) > 0:
                                        randMex = randIntervention['moneyExpressions'][np.random.randint(len(randIntervention['moneyExpressions']))]
                                    else:
                                        continue

                                    if randMex['relatedID'] is None or budget['budgetId'] not in expression['relatedID']:
                                        utt_sgmt_lst = re.split('\n|。', randIntervention['utterance'])
                                        sgmts = ''
                                        for utt in utt_sgmt_lst:
                                            if randMex['moneyExpression'] in utt and not utt[utt.rfind(expression['moneyExpression']) - 1].isdigit():
                                                sgmts = sgmts + utt

                                        output.append([sgmts, budget_text, 0])
                                        false_samples += 1

    for diet in minute_file['diet']:
        for intervention in diet['speechRecord']:
            utterance_segment_list = re.split('\r\n|。', intervention['speech'])
            if len(intervention['moneyExpressions']) > 0:
                for expression in intervention['moneyExpressions']:
                    segments = ''
                    for utterance_segment in utterance_segment_list:
                        if expression['moneyExpression'] in utterance_segment and not utterance_segment[utterance_segment.rfind(expression['moneyExpression']) - 1].isdigit():
                            segments = segments + utterance_segment

                    if expression['relatedID'] is None:
                        randBudget = budget_file['diet'][np.random.randint(len(budget_file['diet']))]
                        budget_text = randBudget['budgetItem'] + ' ' + randBudget['description']
                        output.append([segments, budget_text, 0])

                    else:
                        for budget in budget_file['diet']:

                            budget_text = budget['budgetItem'] + ' ' + budget['description']

                            if expression['relatedID'] is not None and budget['budgetId'] in expression['relatedID']:
                                output.append([segments, budget_text, 1])
                                false_samples = 0
                                while false_samples < 2:

                                    randSession = minute_file['local'][np.random.randint(len(minute_file['diet']))]
                                    randIntervention = randSession['proceeding'][np.random.randint(len(randSession['proceeding']))]
                                    if len(randIntervention['moneyExpressions']) > 0:
                                        randMex = randIntervention['moneyExpressions'][np.random.randint(len(randIntervention['moneyExpressions']))]
                                    else:
                                        continue

                                    if randMex['relatedID'] is None or budget['budgetId'] not in expression['relatedID']:
                                        utt_sgmt_lst = re.split('\r\n|。', randIntervention['utterance'])
                                        sgmts = ''
                                        for utt in utt_sgmt_lst:
                                            if randMex['moneyExpression'] in utt and not utt[utt.rfind(expression['moneyExpression']) - 1].isdigit():
                                                sgmts = sgmts + utt

                                        output.append([sgmts, budget_text, 0])
                                        false_samples += 1

    return pd.DataFrame(data=output, columns=['text_a', 'text_b', 'labels']).sample(frac=1)
