# model architectures: bert, roberta and xlnet
from transformers import T5Tokenizer, XLNetTokenizer, RobertaTokenizer, BertTokenizer, AutoTokenizer
from simpletransformers.classification import ClassificationModel
from datasets.utils.logging import set_verbosity_error

set_verbosity_error()


ARGS = {
    "output_dir": "outputs/",
    "cache_dir": "cache_dir/",
    "show_running_loss": False,
    "fp16": True,
    "fp16_opt_level": "O1",
    "max_seq_length": 128,
    "train_batch_size": 64,
    "gradient_accumulation_steps": 1,
    "eval_batch_size": 64,
    "num_train_epochs": 5,
    "weight_decay": 0,
    "learning_rate": 1e-5,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "labels_list": ['Premise : 過去・決定事項', 'Premise : 未来（現在以降）・見積', 'Premise : その他（例示・訂正事項など）', 'Claim : 意見・提案・質問',
                  'Claim : その他', '金額表現ではない', 'その他'],
    "logging_strategy": "No",
    "save_strategy": "No",
    "disable_tqdm": True,
    "overwrite_output_dir": True,
    "reprocess_input_data": False,
    "evaluate_during_training": False,
    "n_gpu": 1
}


# cl-tohoku/bert-large-japanese (https://huggingface.co/cl-tohoku/bert-large-japanese)
def bert_model(nlabels, label_list):
    ARGS['labels_list'] = label_list
    return ClassificationModel("bert", "cl-tohoku/bert-large-japanese", num_labels=nlabels, args=ARGS)

'''
# cl-tohoku/roberta-base-japanese (https://huggingface.co/cl-tohoku/roberta-base-japanese)
roberta_tokenizer = RobertaTokenizer.from_pretrained("cl-tohoku/roberta-base-japanese")
def roberta_model(nlabels):
    return ClassificationModel("roberta", "cl-tohoku/roberta-base-japanese", num_labels=nlabels, tokenizer_type='roberta', tokenizer_name=roberta_tokenizer, args=ARGS)
'''


# rinna/japanese-roberta-base (https://huggingface.co/rinna/japanese-roberta-base)
rinna_tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-roberta-base")
rinna_tokenizer.do_lower_case = True
def roberta_model(nlabels):
    return ClassificationModel("roberta", "rinna/japanese-roberta-base", num_labels=nlabels, tokenizer_name=rinna_tokenizer, args=ARGS)


# hajime9652/xlnet-japanese (https://huggingface.co/hajime9652/xlnet-japanese)
hajime_tokenizer = XLNetTokenizer.from_pretrained("hajime9652/xlnet-japanese")
def xlnet_model(nlabels):
    return ClassificationModel("xlnet", "hajime9652/xlnet-japanese", num_labels=nlabels, tokenizer_type='xlnet', tokenizer_name=hajime_tokenizer, args=ARGS)



