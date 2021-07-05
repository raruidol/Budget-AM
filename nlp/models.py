# model architectures: bert, xlnet and T5

from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead, AutoModelForMaskedLM
from simpletransformers.classification import ClassificationModel

# cl-tohoku/bert-large-japanese (https://huggingface.co/cl-tohoku/bert-large-japanese)

def bert_tokenizer():
    return AutoTokenizer.from_pretrained("cl-tohoku/bert-large-japanese")

def bert_model(nlabels):
    return ClassificationModel("bert", "cl-tohoku/bert-large-japanese", num_labels=nlabels)

# hajime9652/xlnet-japanese (https://huggingface.co/hajime9652/xlnet-japanese)

def xlnet_tokenizer():
    return AutoTokenizer.from_pretrained("hajime9652/xlnet-japanese")

def xlnet_model(nlabels):
    return ClassificationModel("xlnet", "hajime9652/xlnet-japanese", num_labels=nlabels)

# sonoisa/t5-base-japanese (https://huggingface.co/sonoisa/t5-base-japanese)

def t5_tokenizer():
    return AutoTokenizer.from_pretrained("sonoisa/t5-base-japanese")

def t5_model():
    return ClassificationModel("t5", "sonoisa/t5-base-japanese", num_labels=7)


