# model architectures: bert, xlnet and T5

from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead, AutoModelForMaskedLM


# cl-tohoku/bert-large-japanese (https://huggingface.co/cl-tohoku/bert-large-japanese)

bert_tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-large-japanese")

# define own model with a top softmax layer
bert_model = AutoModelForMaskedLM.from_pretrained("cl-tohoku/bert-large-japanese")

# hajime9652/xlnet-japanese (https://huggingface.co/hajime9652/xlnet-japanese)

xlnet_tokenizer = AutoTokenizer.from_pretrained("hajime9652/xlnet-japanese")

# define own model with a top softmax layer
xlnet_model = AutoModelWithLMHead.from_pretrained("hajime9652/xlnet-japanese")

# sonoisa/t5-base-japanese (https://huggingface.co/sonoisa/t5-base-japanese)

t5_tokenizer = AutoTokenizer.from_pretrained("sonoisa/t5-base-japanese")

# define own model with a top softmax layer
t5_model = AutoModel.from_pretrained("sonoisa/t5-base-japanese")
