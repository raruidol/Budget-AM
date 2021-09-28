import numpy as np
from data import data_processing as preproc
from nlp import models as mdl
import sklearn
from sklearn.metrics import f1_score as f1

SEED = 7
# K-Fold comparison between transformer-based architectures.
FOLD_EVAL = True
# Train definitive model for the Budget-AM task.
TRAIN_DEF = False

np.random.seed(seed=SEED)


if __name__ == "__main__":

    data = preproc.argument_classification_data()

    data_1, data_2, data_3, data_4, data_5 = preproc.cascade_data(data)

    label_list = ['Premise : 過去・決定事項', 'Premise : 未来（現在以降）・見積', 'Premise : その他（例示・訂正事項など）', 'Claim : 意見・提案・質問',
                  'Claim : その他', '金額表現ではない', 'その他']

    if TRAIN_DEF:

        # Model for 7-class classification
        clf_7 = mdl.bert_model(7, label_list)
        clf_7.train_model(data.drop(['ID'], axis=1))
        clf_7.save_model(output_dir='nlp/models/7class_model')

        # Model for premise classification (3-class)
        clf_premise = mdl.bert_model(3, label_list[0:3])
        clf_premise.train_model(data_1.drop(['ID'], axis=1))
        clf_premise.save_model(output_dir='nlp/models/premise_model')

        # Model for claim classification (2-class)
        clf_claim = mdl.bert_model(2, label_list[3:5])
        clf_claim.train_model(data_2.drop(['ID'], axis=1))
        clf_claim.save_model(output_dir='nlp/models/claim_model')

        # Model for non_arg classification (2-class)
        clf_noarg = mdl.bert_model(2, label_list[5:])
        clf_noarg.train_model(data_3.drop(['ID'], axis=1))
        clf_noarg.save_model(output_dir='nlp/models/noarg_model')

        # Model for premise-claim classification (2-class)
        clf_preclm = mdl.bert_model(2, ['Premise', 'Claim'])
        clf_preclm.train_model(data_4.drop(['ID'], axis=1))
        clf_preclm.save_model(output_dir='nlp/models/pre-clm_model')

        # Model for arg-nonarg classification (2-class)
        clf_ana = mdl.bert_model(2, ['Arg', 'None'])
        clf_ana.train_model(data_5.drop(['ID'], axis=1))
        clf_ana.save_model(output_dir='nlp/models/arg-nonarg_model')

    elif FOLD_EVAL:

        folds = preproc.kfold_data(data, 10)

        i = 0

        acc_bert = 0
        acc_roberta = 0
        acc_xlnet = 0

        for fold in folds:

            print('FOLD ', i)

            clf_bert = mdl.bert_model(7)
            clf_roberta = mdl.roberta_model(7)
            clf_xlnet = mdl.xlnet_model(7)

            clf_bert.train_model(fold['train'].drop(['ID'], axis=1))
            clf_roberta.train_model(fold['train'].drop(['ID'], axis=1))
            clf_xlnet.train_model(fold['train'].drop(['ID'], axis=1))

            res_bert, out_bert, wrong_bert = clf_bert.eval_model(fold['test'].drop(['ID'], axis=1), acc=sklearn.metrics.accuracy_score)
            res_roberta, out_roberta, wrong_roberta = clf_roberta.eval_model(fold['test'].drop(['ID'], axis=1), acc=sklearn.metrics.accuracy_score)
            res_xlnet, out_xlnet, wrong_xlnet = clf_xlnet.eval_model(fold['test'].drop(['ID'], axis=1), acc=sklearn.metrics.accuracy_score)

            acc_bert += res_bert['acc']
            acc_roberta += res_roberta['acc']
            acc_xlnet += res_xlnet['acc']

            print('BERT: ', res_bert)
            print('RoBERTa: ', res_roberta)
            print('XLNET: ', res_xlnet)

            print('------------------------------------------')

            # clf_bert.save_model(output_dir='nlp/models/basic_model'+str(i))

            i += 1

        print('------------------------------------------')
        print('------------------------------------------')
        print('RESULTS:')
        print('------------------------------------------')
        print('------------------------------------------')

        print('BERT: ', acc_bert)
        print('RoBERTa: ', acc_roberta)
        print('XLNET: ', acc_xlnet)

    else:
        print('Please choose between doing a K-Fold comparison or training the final model.')





