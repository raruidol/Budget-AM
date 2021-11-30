# NLP algorithms
import fugashi
import re
from operator import itemgetter
from sentence_transformers import util


def budget_score(text, budget):
    tagger = fugashi.Tagger()
    proc_speech = [str(word) for word in tagger(text)]
    # print(proc_speech)

    budget_id = [word for word in tagger(budget.budgetItem)]
    if budget.description is not None:
        budget_description = [word for word in tagger(budget.description)]
    else:
        budget_description = []
    budget_title = [word for word in tagger(budget.budgetTitle)]
    # budget_concept = budget_id + budget_description + budget_title

    score_id, norm_id, score_desc, norm_desc, score_title, norm_title = 0, 0, 0, 0, 0, 0

    for concept in budget_id:
        # check if it is a noun
        if concept.pos.split(',')[0] == '名詞' and concept.pos.split(',')[1] == '普通名詞':

            # nhits = proc_speech.count(str(concept))
            if str(concept) in proc_speech:
                score_desc += 1
            norm_id += 1

    for concept in budget_description:
        # check if it is a noun
        if concept.pos.split(',')[0] == '名詞' and concept.pos.split(',')[1] == '普通名詞':
            # nhits = proc_speech.count(str(concept))
            if str(concept) in proc_speech:
                score_desc += 1
            norm_desc += 1

    for concept in budget_title:
        # check if it is a noun
        if concept.pos.split(',')[0] == '名詞' and concept.pos.split(',')[1] == '普通名詞':

            # nhits = proc_speech.count(str(concept))
            if str(concept) in proc_speech:
                score_desc += 1
            norm_title += 1

    n_id = 0
    n_title = 0
    n_desc = 0

    if norm_id != 0:
        n_id = (score_id / norm_id)
    if norm_title != 0:
        n_title = (score_title / norm_title)
    if norm_desc != 0:
        n_desc = (score_desc / norm_desc)

    if budget.description is not None:
        score = (n_id + n_desc + n_title) / 3
    else:
        score = (n_id + n_title) / 2

    return score


def similar_score(text, budget_text, model):

    emb1 = model.encode(text)
    emb2 = model.encode(budget_text)

    return util.cos_sim(emb1, emb2)[0].item()
