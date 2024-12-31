import re
import copy
import random
import pandas as pd
from pathlib import Path

from settings import benchmark_settings
from preprocessing import wordsplit

seed = 61
random.seed(seed)

def random_replace(string):
    '''
        variable → regular expression → imitated variable
    '''
    letters = re.findall('[a-zA-Z]', string)
    numbers = re.findall('[0-9]', string)

    replaced_letters = random.choices(letters, k=len(letters))
    replaced_numbers = random.choices(numbers, k=len(numbers))

    replaced_string = re.sub('[a-zA-Z]', lambda _: replaced_letters.pop(0), string)
    replaced_string = re.sub('[0-9]', lambda _: replaced_numbers.pop(0), replaced_string)

    return replaced_string


def sim(seq1, seq2):
    long  = copy.copy(seq1) if len(seq1) > len(seq2) else copy.copy(seq2)
    short = copy.copy(seq2) if len(seq1) > len(seq2) else copy.copy(seq1)
    simTokens = 0
    for token in short:
        if token in long:
            simTokens+=1
    retVal = simTokens / len(long)
    return retVal


def find_continuous_subsequences(nums):
    subsequences = []
    current_subsequence = []

    for i in range(len(nums)):
        if i > 0 and nums[i] != nums[i - 1] + 1:
            if len(current_subsequence) > 1:
                subsequences.append(current_subsequence)
            current_subsequence = []

        current_subsequence.append(nums[i])

    if len(current_subsequence) > 1:
        subsequences.append(current_subsequence)

    return subsequences

def convert_to_regex_pattern(string):
    regex_pattern = ''
    for char in string:
        if char.isalpha():
            regex_pattern += r'[a-zA-Z]'
        elif char.isdigit():
            regex_pattern += r'-?\d*'
        else:
            if char != '-':
                regex_pattern += re.escape(char)
    return regex_pattern


def is_number(s):
    for func in (float, lambda x: int(x, 0)):
        try:
            func(s)
            return True
        except ValueError:
            continue
    return False

def wordcomp(prediction, template):
    _template = list(template)
    for i, _token in enumerate(_template):
        token = re.sub(r'\W+', '', _token)
        _template[i] = token
    for i, _token in enumerate(prediction):
        token = re.sub(r'\W+', '', _token)
        if token != '' and token not in _template:
            # print("---")
            # print(token, _template)
            if not is_number(token) and i < len(template) and "<*>" not in template[i]:
                return False
    return True


def has_consecutive_variable(log,key):
    long  = copy.copy(log) if len(log)>len(key) else copy.copy(key)
    short = copy.copy(key) if len(log)>len(key) else copy.copy(log)
    long, short = list(long), list(short)
    save=[]
    for word in long:
        if word not in short:
           save.append(long.index(word))
    consecutive=find_continuous_subsequences(save)
    for i in range(len(consecutive)):
                start=consecutive[i][0]
                end=consecutive[i][len(consecutive[i])-1]
                comppare=0
                for word in long[start:end+1]:
                    if comppare == 0:
                        regex = convert_to_regex_pattern(word)
                        comppare = 1
                    else:
                        if not re.findall(regex, word):
                            return False
                long[start:end + 1] = ['<*>']
    if len(long)==len(short):
        return True
    else:
        return False


def get_templ(template, logsplit):
    template = list(template)
    for i, token in enumerate(template):
        if token not in logsplit:
            template[i] = "<*>"
    # from itertools import groupby
    # template = [key for key, _group in groupby(template)]
    return tuple(template)

def group_update(project, lg, log, prediction, appendvalue):
    # print(prediction)
    pred = prediction.replace("<*>", " ")
    pred = tuple(wordsplit(pred,project))

    # pred = wordsplit(prediction,project)
    # for i,p in enumerate(pred):
    #     if "<*>" in p:
    #         pred[i] = p.replace("<*>", "")

    logsplit = tuple(wordsplit(log,project))
    candidates = []

    # print(appendvalue, log)
    # print(logsplit)
    # print(pred)

    keys = sorted(lg.keys(), key=lambda x: len(x))
    for templ in keys:
        # if wordcomp(pred, templ) or wordcomp(logsplit, templ):
        if wordcomp(logsplit, templ) or wordcomp(pred, templ):
            _templ = get_templ(templ, logsplit)
            if set(_templ) == {'<*>'}: continue
            if not has_consecutive_variable(logsplit, _templ): continue
            # if len(_templ) != len(logsplit): continue
            candidates.append([templ, _templ])

    if len(candidates) >= 1:
        sim_list=[]
        for cand in candidates:
            sim_list.append(sim(cand[1], logsplit))
        maxsim = sim_list.index(max(sim_list))
        templ  = candidates[maxsim][0]
        _templ = candidates[maxsim][1]
        if _templ != templ:
            lg[_templ] = lg.pop(templ)
        lg[_templ].append(appendvalue)
        # print(candidates)
        # print("_templ:", _templ)
        plg(lg)
    else:
        if logsplit not in lg:
            lg[logsplit] = [appendvalue]
        else:
            lg[logsplit].append(appendvalue)
        # print("group_update: else>else")
        plg(lg)

    return lg

def exclude_digits(string):
    pattern = r'\d'
    digits = re.findall(pattern, string)
    if len(digits)==0:
        return False
    return len(digits)/len(string) >= 0.3


def template_update(template_group,log_sentence):
    new_template_group={}
    for key in template_group.keys():
        group_list = template_group[key]
        if group_list[0] == -1:
            continue
        len_save = list()
        for id in group_list:
            lenth = len(log_sentence[id])
            len_save.append(lenth)
        max_lenth = max(len_save, key=len_save.count)
        filter = {}
        for id in group_list:
            group_member = log_sentence[id]
            if len(group_member) != max_lenth:
                continue
            else:
                ind = 0
                for word in group_member:
                    filter.setdefault(ind, []).append(word)
                    ind += 1
        templat = list()
        for key in filter.keys():
            all_words = filter[key]
            all_words = list(set(all_words))
            if len(all_words) == 1:
                if not exclude_digits(all_words[0]):
                    templat.append(all_words[0])
                else:
                    templat.append('<*>')
            else:
                templat.append('<*>')
        templat = ' '.join(templat)
        for id in group_list:
            new_template_group.setdefault(templat, []).append(id)
    return new_template_group





def plg(lg):
    # print("{\n" + "\n".join("{!r}: {!r},".format(k, v) for k, v in lg.items()) + "\n}")
    pass


def prepare_data(rootdir, project):

    train_data = pd.read_csv("mutes/" + project + "/train.csv", index_col=False).values.tolist()
    parse_data = pd.read_csv("mutes/" + project + "/test.csv",  index_col=False).values.tolist()

    from openprompt.data_utils import InputExample

    train_dataset = []
    for i in range(len(train_data)):
        input_example = InputExample(
            text_a=train_data[i][0],
            meta={"EventTemplate": train_data[i][1]},
            label=0,
        )
        train_dataset.append(input_example)

    parse_dataset = []
    for i in range(len(parse_data)):
        input_example = InputExample(
            text_a=parse_data[i][0],
            meta={"EventTemplate": parse_data[i][1]},
            label=0,
        )
        parse_dataset.append(input_example)

    templates = [item[1] for item in parse_data]
    ground_list = {}
    ground_list_values = pd.Series(templates).value_counts().index.tolist()
    for value in ground_list_values:
        indices = [i for i, x in enumerate(templates) if x == value]
        ground_list[value] = indices

    print("LEN:", len(train_dataset), len(parse_dataset))
    return train_dataset, parse_dataset, parse_data


if __name__ == '__main__':
    from pathlib import Path
    rootdir = Path(__file__).absolute().parent.parent
    print(f"Project root: {rootdir}")
    prepare_data(rootdir, 'Windows')
