import numpy as np
from data import data_processing as preproc
from nlp import models as mdl
import sklearn
from sklearn.metrics import f1_score as f1
from datasets.utils.logging import set_verbosity_error
import pandas as pd

set_verbosity_error()
pd.set_option("display.max_rows", None, "display.max_columns", None)

# BASIC, CASCADE
MODEL = 'CASCADE'
NUMBER_CLASSES = 5
SEED = 7
# K-Fold evaluation of the trained models.
FOLD_EVAL = False
# Train definitive model for the Budget-AM task.
TRAIN_DEF = False
# Train related ID identification model.
TRAIN_RID = True

np.random.seed(seed=SEED)


if __name__ == "__main__":

    if TRAIN_DEF:

        data = preproc.argument_classification_data(NUMBER_CLASSES)
        # data_1 -> premise classes only data
        # data_2 -> claim classes only data
        # data_3 -> premise vs. claim data (2 classes)
        # data_4 -> premise vs. claim balanced data (2 classes)
        # data_5 -> balanced data (7 classes)
        # data_6 -> balanced data (5 classes)
        data_1, data_2, data_3, data_4, data_5, data_6 = preproc.cascade_data(data)

        # Classes '金額表現ではない', 'その他' not included due to low representation
        label_list = ['Premise : 過去・決定事項', 'Premise : 未来（現在以降）・見積', 'Premise : その他（例示・訂正事項など）', 'Claim : 意見・提案・質問',
                      'Claim : その他']

        # Model for basic classification (5-class)
        clf_basic = mdl.bert_model(5, label_list)
        clf_basic.train_model(data.drop(['ID'], axis=1))
        clf_basic.save_model(output_dir='nlp/models/basic_model')
        
        # Model for premise classification (3-class)
        clf_premise = mdl.bert_model(3, label_list[0:3])
        clf_premise.train_model(data_1.drop(['ID'], axis=1))
        clf_premise.save_model(output_dir='nlp/models/premise_model')
        
        # Model for claim classification (2-class)
        clf_claim = mdl.bert_model(2, label_list[3:5])
        clf_claim.train_model(data_2.drop(['ID'], axis=1))
        clf_claim.save_model(output_dir='nlp/models/claim_model')
        
        # Model for premise-claim classification (2-class)
        clf_preclm = mdl.bert_model(2, ['Premise', 'Claim'])
        clf_preclm.train_model(data_3.drop(['ID'], axis=1))
        clf_preclm.save_model(output_dir='nlp/models/pre-clm_model')

        # Model for premise-claim classification with balanced data (2-class)
        clf_preclm = mdl.bert_model(2, ['Premise', 'Claim'])
        clf_preclm.train_model(data_4.drop(['ID'], axis=1))
        clf_preclm.save_model(output_dir='nlp/models/balanced-pc_model')

    elif FOLD_EVAL:

        data = preproc.argument_classification_data(NUMBER_CLASSES)
        # data_1 -> premise classes only data
        # data_2 -> claim classes only data
        # data_3 -> premise vs. claim data (2 classes)
        # data_4 -> premise vs. claim balanced data (2 classes)
        # data_5 -> balanced data (7 classes)
        # data_6 -> balanced data (5 classes)
        data_1, data_2, data_3, data_4, data_5, data_6 = preproc.cascade_data(data)

        # label_list = ['Premise : 過去・決定事項', 'Premise : 未来（現在以降）・見積', 'Premise : その他（例示・訂正事項など）', 'Claim : 意見・提案・質問',
        #              'Claim : その他', '金額表現ではない', 'その他']

        label_list = ['Premise : 過去・決定事項', 'Premise : 未来（現在以降）・見積', 'Premise : その他（例示・訂正事項など）', 'Claim : 意見・提案・質問',
                      'Claim : その他']

        folds = preproc.kfold_data(data, 10, False)
        folds_preclm = preproc.kfold_data(data_4, 10, True)
        folds_prem = preproc.kfold_data(data_1, 10, False)
        folds_clm = preproc.kfold_data(data_2, 10, False)

        i = 0

        acc_bert = 0
        f1_bert = 0

        for fold in folds:

            print('------------------------------------------')
            print('FOLD ', i)

            if MODEL == 'BASIC':
                clf_bert = mdl.bert_model(5, label_list)
                clf_bert.train_model(fold['train'].drop(['ID'], axis=1))
                res_bert, out_bert, wrong_bert = clf_bert.eval_model(fold['test'].drop(['ID'], axis=1), acc=sklearn.metrics.accuracy_score)

                acc = res_bert['acc']

                preds = []
                for pred in out_bert:
                    preds.append(pred.tolist().index(max(pred.tolist())))

            elif MODEL == 'CASCADE':

                acc = 0
                tgt_list = fold['test']['label'].tolist()
                span_list = fold['test']['text'].tolist()

                clf = mdl.bert_model(2, ['Premise', 'Claim'])
                clf.train_model(folds_preclm[i]['train'].drop(['ID'], axis=1))
                preds = []
                for sample in span_list:
                    pred, r_o = clf.predict([sample])
                    preds.append(pred[0])

                print('St1 preds: ', preds)

                clf = mdl.bert_model(3, ['Premise : 過去・決定事項', 'Premise : 未来（現在以降）・見積', 'Premise : その他（例示・訂正事項など）'])
                clf.train_model(folds_prem[i]['train'].drop(['ID'], axis=1))

                for j in range(len(span_list)):
                    if preds[j] == 'Premise':
                        pred, r_o = clf.predict([span_list[i]])
                        if tgt_list[j] == pred[0]:
                            acc += 1
                        if pred[0] == 'Premise : 過去・決定事項':
                            preds[j] = 0
                        elif pred[0] == 'Premise : 未来（現在以降）・見積':
                            preds[j] = 1
                        elif pred[0] == 'Premise : その他（例示・訂正事項など）':
                            preds[j] = 2

                print('St2 preds: ', preds)

                clf = mdl.bert_model(2, ['Claim : 意見・提案・質問', 'Claim : その他'])
                clf.train_model(folds_clm[i]['train'].drop(['ID'], axis=1))

                for j in range(len(span_list)):
                    if preds[j] == 'Claim':
                        pred, r_o = clf.predict([span_list[i]])
                        if tgt_list[j] == pred[0]:
                            acc += 1
                        if pred[0] == 'Claim : 意見・提案・質問':
                            preds[j] = 3
                        elif pred[0] == 'Claim : その他':
                            preds[j] = 4

                print('St2 preds: ', preds)

                acc /= len(span_list)

            acc_bert += acc

            targets = []
            for tg in fold['test']['label'].tolist():
                # classes: Premise : 過去・決定事項 ; Premise : 未来（現在以降）・見積 ; Premise : その他（例示・訂正事項など）;
                # Claim : 意見・提案・質問 ; Claim : その他 ; 金額表現ではない ; その他
                if tg == 'Premise : 過去・決定事項':
                    targets.append(0)
                elif tg == 'Premise : 未来（現在以降）・見積':
                    targets.append(1)
                elif tg == 'Premise : その他（例示・訂正事項など）':
                    targets.append(2)
                elif tg == 'Claim : 意見・提案・質問':
                    targets.append(3)
                elif tg == 'Claim : その他':
                    targets.append(4)
                elif tg == '金額表現ではない':
                    targets.append(5)
                elif tg == 'その他':
                    targets.append(6)
                else:
                    print("WATEFOC", tg)
                    targets.append(7)

            print('Real: ', targets)
            print('Pred: ', preds)
            print('Accuracy:', acc)
            print('F1-macro score:', f1(targets, preds[0:len(targets)], average='macro'))
            f1_bert += f1(targets, preds[0:len(targets)], average='macro')

            print('------------------------------------------')

            # clf_bert.save_model(output_dir='nlp/models/basic_model'+str(i))

            i += 1

        print('------------------------------------------')
        print('------------------------------------------')
        print('RESULTS (acc, f1-macro): ', acc_bert/10, f1_bert/10)

    elif TRAIN_RID:

        data_id = preproc.related_id_data()

        clf_basic = mdl.bert_model(2, [0, 1])
        clf_basic.train_model(data_id)
        clf_basic.save_model(output_dir='nlp/models/relID_model')

    else:
        print('Please choose between doing a K-Fold comparison or training the final model.')





