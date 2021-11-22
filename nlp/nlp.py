# NLP algorithms
import fugashi
import re
from operator import itemgetter
from sentence_transformers import util

def budget_score_local(proc, budget_list):
    scores = {}
    tagger = fugashi.Tagger()
    proc_speech = [str(word) for word in tagger(proc.utterance)]
    # print(proc_speech)

    for budget in budget_list:
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
            scores[budget.budgetId] = (n_id + n_desc + n_title) / 3
        else:
            scores[budget.budgetId] = (n_id + n_title) / 2

    print(scores)

    if scores[max(scores, key=scores.get)] == 0:
        return None
    else:
        top2 = sorted(scores.items(), key=itemgetter(1), reverse=True)[:2]
        res = []
        for idt in top2:
            res.append(idt[0])

        return res


def budget_score_diet(sprec, budget_list):
    scores = {}
    tagger = fugashi.Tagger()
    proc_speech = [str(word) for word in tagger(sprec.speech)]

    for budget in budget_list:
        budget_id = [word for word in tagger(budget.budgetItem)]
        if budget.description is not None:
            budget_description = [word for word in tagger(budget.description)]
        else:
            budget_description = []
        budget_title = [word for word in tagger(budget.budgetTitle)]
        # budget_concept = budget_id + budget_description + budget_title

        score_id = 0
        norm_id = 0
        score_desc = 0
        norm_desc = 0
        score_title = 0
        norm_title = 0

        for concept in budget_id:
            # check if it is a noun
            if concept.pos == '名詞':
                nhits = proc_speech.count(str(concept))
                score_id += nhits
                norm_id += 1

        for concept in budget_description:
            # check if it is a noun
            if concept.pos == '名詞':
                nhits = proc_speech.count(str(concept))
                score_desc += nhits
                norm_desc += 1

        for concept in budget_title:
            # check if it is a noun
            if concept.pos == '名詞':
                nhits = proc_speech.count(str(concept))
                score_title += nhits
                norm_title += 1

        if budget.description is not None:
            scores[budget.budgetId] = ((score_id / norm_id) + (score_desc / norm_desc) + (score_title / norm_title)) / 3
        else:
            scores[budget.budgetId] = ((score_id / norm_id) + (score_title / norm_title)) / 2

    if scores[max(scores, key=scores.get)] == 0:
        return None
    else:
        top5 = sorted(scores.items(), key=itemgetter(1), reverse=True)[:5]
        res = []
        for idt in top5:
            res.append(idt[0])

        return res


def similar_score(text, budget_text, model):

    emb1 = model.encode(text)
    emb2 = model.encode(budget_text)

    return util.cos_sim(emb1, emb2)[0].item()
