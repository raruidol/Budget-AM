# model architectures: bert, xlnet and T5

from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead, AutoModelForMaskedLM
from simpletransformers.classification import ClassificationModel
from simpletransformers.classification import MultiLabelClassificationModel


ARGS = {
    "output_dir": "outputs/",
    "cache_dir": "cache_dir/",

    "fp16": True,
    "fp16_opt_level": "O1",
    "max_seq_length": 128,
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "eval_batch_size": 8,
    "num_train_epochs": 10,
    "weight_decay": 0,
    "learning_rate": 4e-5,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,

    "logging_steps": 50,
    "save_steps": 2000,

    "overwrite_output_dir": False,
    "reprocess_input_data": False,
    "evaluate_during_training": False,

    "n_gpu": 1,
}

# cl-tohoku/bert-large-japanese (https://huggingface.co/cl-tohoku/bert-large-japanese)

def bert_model(nlabels):
    return MultiLabelClassificationModel("bert", "cl-tohoku/bert-large-japanese", num_labels=nlabels, args=ARGS)

# hajime9652/xlnet-japanese (https://huggingface.co/hajime9652/xlnet-japanese)

def xlnet_model(nlabels):
    return MultiLabelClassificationModel("xlnet", "hajime9652/xlnet-japanese", num_labels=nlabels, args=ARGS)

# sonoisa/t5-base-japanese (https://huggingface.co/sonoisa/t5-base-japanese)

def t5_tokenizer():
    return AutoTokenizer.from_pretrained("sonoisa/t5-base-japanese")

def t5_model():
    return ClassificationModel("t5", "sonoisa/t5-base-japanese", num_labels=7)


