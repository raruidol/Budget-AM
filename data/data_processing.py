import json

# Create 2 different data structures from input data to approach arg. classification and ID retrieval.


def argument_classification_data(filepath):
    with open(filepath) as file:
        minute_file = json.load(file)
        file.close()

    output = []
    for session in minute_file:
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

                    output.append([expression['moneyExpression'], expression['argumentClass'], segments])

    return output


# Rebuild the original json files with the outputs of the algorithm.


